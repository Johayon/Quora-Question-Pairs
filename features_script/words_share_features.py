import argparse
import itertools
import pandas as pd
import pickle
from nltk.corpus import stopwords

stops = stopwords.words("english")
stops_set = set(stops)

def word_share(words1,words2):
    
    w1 = set(words1)
    w2 = set(words2)
    
    shared = w1 & w2
    diff = w1 ^ w2
    
    return pd.Series([shared, diff])

def get_weight(count, eps=10000, min_count=2):
    return 0 if count < min_count else 1 / (count + eps)

def create_weights(train_data):
    train_qs = pd.Series(train_data[typ+'_question1'].tolist() + train_data[typ+'_question2'].tolist()).astype(str)
    words = (" ".join(train_qs)).split()
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    return weights

def bigrams(words):
    if len(words) > 1:
        return [i for i in zip(words[:-1], words[1:])]
    else:
        return []
    
def trigrams(words):
    if len(words) > 2:
        return [i for i in zip(words[:-2], words[1:-1], words[2:])]
    else:
        return [] 

def word_share_bigrams(words1,words2):
    
    b1 = bigrams(words1)
    b2 = bigrams(words2)
    
    w1 = set(b1)
    w2 = set(b2)
    
    shared = len(w1 & w2)
    diff = len(w1 ^ w2)
    
    return pd.Series([shared/max(shared+diff,1), shared/max(len(b1)+len(b2),1)])

def word_share_trigrams(words1,words2):
    
    b1 = trigrams(words1)
    b2 = trigrams(words2)
    
    w1 = set(b1)
    w2 = set(b2)
    
    shared = len(w1 & w2)
    diff = len(w1 ^ w2)
    
    return pd.Series([shared/max(shared+diff,1), shared/max(len(b1)+len(b2),1)])

def word_hamming(words1,words2):
    return sum(1 for i in zip(words1, words2) if i[0]==i[1])/max(len(words1), len(words2),1)



if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='stem/lem compute')
    parser.add_argument('--data',default="train", choices=['train','test'])
    parser.add_argument('--type',default="clean", choices=['clean','stem','lem'],type=str)    
    
    main_args = parser.parse_args()
    typ = main_args.type
    data_type = main_args.data    
    
    print("processing {} data".format(data_type))
    index_col = 'id'
    if data_type == 'test':
        index_col = 'test_id'
    data = pd.read_csv('../data/{}_{}.csv'.format(typ,data_type), index_col=index_col)
        
    for i in range(1,3):
        print(typ+" len and nb words"+str(i))
        data[typ+'_question'+str(i)] = data[typ+'_question'+str(i)].astype(str)
        data[typ+'_words'+str(i)] = data[typ+'_question'+str(i)].str.split()
        data[typ+'_words_stop'+str(i)] = data[typ+'_words'+str(i)].apply(
             lambda words: [w for w in words if not w in stops])
        
    print(typ+' set word ')
    data[[typ+'_common_set',typ+'_diff_set']] = data.apply(
        lambda x: word_share(x[typ+'_words1'],x[typ+'_words2']), axis=1)
    
    print(typ+' set word stop')
    data[[typ+'_common_set_stop', typ+'_diff_set_stop']] = data.apply(
        lambda x: word_share(x[typ+'_words_stop1'],x[typ+'_words_stop2']), axis=1)
    
    added_features = []
    
    for stop in ['','_stop']:
        print(typ+' set word '+stop)
        
        added_features.append(typ+'_common_set_len'+stop)
        data[typ+'_common_set_len'+stop] = data[typ+'_common_set'+stop].apply(len)
        
        added_features.append(typ+'_diff_set_len'+stop)
        data[typ+'_diff_set_len'+stop] = data[typ+'_diff_set'+stop].apply(len)
    
    added_features.append(typ+'_stop_common_ratio')
    data[typ+'_stop_common_ratio'] = data[typ+'_common_set_len_stop'] / data[typ+'_common_set_len']
    
    added_features.append(typ+'_stop_diff_ratio')
    data[typ+'_stop_diff_ratio'] = data[typ+'_diff_set_len_stop'] / data[typ+'_diff_set_len']
    
    added_features.append(typ+'_word_share_stop')
    data[typ+'_word_share_stop'] = (data[typ+'_common_set_len_stop'] / 
                                    (data[typ+'_diff_set_len_stop'] + data[typ+'_common_set_len_stop']))
    
    added_features.append(typ+'_word_share')
    data[typ+'_word_share'] = data[typ+'_common_set_len'] / (data[typ+'_diff_set_len'] + data[typ+'_common_set_len'])
    
    added_features.extend([typ+'_shared_bigrams_jac',typ+'_shared_bigram_dice'])
    data[[typ+'_shared_bigrams_jac',typ+'_shared_bigram_dice']] = data.apply(
        lambda x: word_share_bigrams(x[typ+'_words1'],x[typ+'_words2']), axis=1)
    
    added_features.extend([typ+'_shared_trigrams_jac',typ+'_shared_trigram_dice'])
    data[[typ+'_shared_trigrams_jac',typ+'_shared_trigram_dice']] = data.apply(
        lambda x: word_share_trigrams(x[typ+'_words1'],x[typ+'_words2']), axis=1)
    
    added_features.extend([typ+'_word_hamming'])
    data[typ+'_word_hamming'] = data.apply(
        lambda x: word_hamming(x[typ+'_words1'],x[typ+'_words2']), axis=1)
    
    if data_type == "train":
        weights = create_weights(data)
        with open('pickles/{}_train_weights.pickle'.format(typ), 'wb') as handle:
            pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        with open('pickles/{}_train_weights.pickle'.format(typ), 'rb') as handle:
            weights = pickle.load(handle)
            
    def idf_word_share(common_set, diff_set):
        common_weight =[weights.get(w, 0) for w in common_set]
        diff_weight =[weights.get(w, 0) for w in diff_set]
        denominator = np.sum(common_weight)+np.sum(diff_weight)
        
        if denominator == 0:
            return 0
        
        return np.sum(common_weight)/denominator
    
    added_features = []

    print('idf word share')
    added_features.extend([typ+'_idf_word_share_stop',typ+'_idf_word_share'])
    data[typ+'_idf_word_share_stop'] = data.apply(lambda x: idf_word_share(x[typ+'_common_set_stop'],
                                                                                      x[typ+'_diff_set_stop']),axis=1)
    data[typ+'_idf_word_share'] = data.apply(lambda x: idf_word_share(x[typ+'_common_set'], x[typ+'_diff_set']),axis=1)
    
    
    data[added_features].to_csv('../features/{}_{}_word_share_features.csv'.format(typ,data_type))
