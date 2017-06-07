import difflib, Levenshtein, distance
import argparse
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import itertools

def multiple_distance(string1,string2):
    try:
        diff = difflib.SequenceMatcher(None, string1, string2).ratio()
        leve = Levenshtein.ratio(string1, string2)
        sore = 1 - distance.sorensen(string1, string2)
        jacc = 1 - distance.jaccard(string1, string2)
    except:
        diff,leve,sore,jacc = 0,0,0,0
    return pd.Series([diff, leve, sore, jacc])


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
    stops = stopwords.words("english")
    added_features = []
    print(">>>>>>>>>>>> size: {}".format(data.shape[0]))

    for i in range(1,3):
        print(typ+" len and nb words"+str(i))
        data[typ+'_question'+str(i)] = data[typ+'_question'+str(i)].astype(str)
        data[typ+'_len'+str(i)] = data[typ+'_question'+str(i)].str.len()
        data[typ+'_words'+str(i)] = data[typ+'_question'+str(i)].str.split()
        data[typ+'_nb_words'+str(i)] = data[typ+'_words'+str(i)].apply(len)

        print(typ+" question stop" +str(i))
        data[typ+'_words_stop'+str(i)] = data[typ+'_words'+str(i)].apply(
            lambda words: [w for w in words if not w in stops])
        data[typ+'_nb_words_stop'+str(i)] = data[typ+'_words_stop'+str(i)].apply(len)
        data[typ+'_question_stop'+str(i)] = data[typ+'_words_stop'+str(i)].apply(lambda x: " ".join(x))
        data[typ+'_len_stop'+str(i)] = data[typ+'_question_stop'+str(i)].str.len()

    for cat,stop in itertools.product(['len','nb_words'],['','_stop']):
        added_features.append(typ+'_'+'min_'+cat+stop)
        data[typ+'_'+'min_'+cat+stop] = np.minimum(data[typ+'_'+cat+stop+'1'],
                                                         data[typ+'_'+cat+stop+'2'])
        added_features.append(typ+'_'+'max_'+cat+stop)
        data[typ+'_'+'max_'+cat+stop] = np.maximum(data[typ+'_'+cat+stop+'1'],
                                                         data[typ+'_'+cat+stop+'2'])
        added_features.append(typ+'_'+'abs_diff_'+cat+stop)
        data[typ+'_'+'abs_diff_'+cat+stop] = np.abs(data[typ+'_'+cat+stop+'1'] - 
                                                          data[typ+'_'+cat+stop+'2'])
        added_features.append(typ+'_'+'abs_diff_norm'+cat+stop)
        data[typ+'_'+'abs_diff_norm'+cat+stop] = (np.abs(data[typ+'_'+cat+stop+'1'] - 
                                                               data[typ+'_'+cat+stop+'2']) /
                                                        np.maximum(1,(data[typ+'_'+cat+stop+'1'] + 
                                                                      data[typ+'_'+cat+stop+'2'])))
    for stop in ['','_stop']:
        print(typ+' multiple distance '+stop)
        added_features.extend([typ+'_diff'+stop,typ+'_leve'+stop,typ+'_sore'+stop,typ+'_jacc'+stop])
        data[[typ+'_diff'+stop,typ+'_leve'+stop,typ+'_sore'+stop,typ+'_jacc'+stop]] = data[
            [typ+'_question'+stop+'1',typ+'_question'+stop+'2']].apply(
            lambda x: multiple_distance(x[typ+'_question'+stop+'1'], x[typ+'_question'+stop+'2']),axis=1)

    data[added_features].to_csv('../features/{}_{}_distance.csv'.format(typ,data_type))

    
    
    
    
    