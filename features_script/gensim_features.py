import pandas as pd
import numpy as np
import gensim
from scipy.stats import skew, kurtosis
from scipy.spatial.distance import cosine, cityblock, jaccard, canberra, euclidean, minkowski, braycurtis
import argparse
from nltk.corpus import stopwords
from nltk import word_tokenize
stop_words = stopwords.words('english')

def wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return model.wmdistance(s1, s2)


def norm_wmd(s1, s2):
    s1 = str(s1).lower().split()
    s2 = str(s2).lower().split()
    s1 = [w for w in s1 if w not in stop_words]
    s2 = [w for w in s2 if w not in stop_words]
    return norm_model.wmdistance(s1, s2)


def sent2vec(s):
    words = str(s).lower()
    words = word_tokenize(words)
    words = [w for w in words if not w in stop_words]
    words = [w for w in words if w.isalpha()]
    M = []
    for w in words:
        try:
            M.append(model[w])
        except:
            continue
    M = np.array(M)
    v = M.sum(axis=0)
    return v / np.sqrt((v ** 2).sum())

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='stem/lem compute')
    parser.add_argument('--data',default="train", choices=['train','test'])
    
    main_args = parser.parse_args()
    typ = 'clean'
    data_type = main_args.data
        
    print("processing {} data".format(data_type))
    index_col = 'id'
    if data_type == 'test':
        index_col = 'test_id'
    data = pd.read_csv('../data/{}_{}.csv'.format('corrected',data_type), index_col=index_col)
    stops = stopwords.words("english")
    added_features = []
    print(">>>>>>>>>>>> size: {}".format(data.shape[0]))

    data[typ+'_question1'] = data[typ+'_question1'].astype(str)
    data[typ+'_question2'] = data[typ+'_question2'].astype(str)
    added_features = []
    
    print("creating gensin model")
    added_features.append(typ+'_'+'wmd')
    model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    data[typ+'_'+'wmd'] = data.apply(lambda x: wmd(x[typ+'_'+'question1'], x[typ+'_'+'question2']), axis=1)

    print("creating gensin norm model")
    added_features.append(typ+'_'+'norm_wmd')
    norm_model = gensim.models.KeyedVectors.load_word2vec_format('data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    norm_model.init_sims(replace=True)
    data[typ+'_'+'norm_wmd'] = data.apply(lambda x: norm_wmd(x[typ+'_'+'question1'], x[typ+'_'+'question2']), axis=1)

    print("creating sent2vec model")
    question1_vectors = np.zeros((data.shape[0], 300))
    error_count = 0

    for i, q in enumerate(data[typ+'_'+'question1'].values):
        question1_vectors[i, :] = sent2vec(q)

    question2_vectors  = np.zeros((data.shape[0], 300))
    for i, q in enumerate(data[typ+'_'+'question2'].values):
        question2_vectors[i, :] = sent2vec(q)

    added_features.append(typ+'_'+'cosine_distance')
    data[typ+'_'+'cosine_distance'] = [cosine(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    added_features.append(typ+'_'+'cityblock_distance')
    data[typ+'_'+'cityblock_distance'] = [cityblock(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    added_features.append(typ+'_'+'jaccard_distance')
    data[typ+'_'+'jaccard_distance'] = [jaccard(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    added_features.append(typ+'_'+'canberra_distance')
    data[typ+'_'+'canberra_distance'] = [canberra(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    added_features.append(typ+'_'+'euclidean_distance')
    data[typ+'_'+'euclidean_distance'] = [euclidean(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    added_features.append(typ+'_'+'minkowski_distance')
    data[typ+'_'+'minkowski_distance'] = [minkowski(x, y, 3) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    added_features.append(typ+'_'+'braycurtis_distance')
    data[typ+'_'+'braycurtis_distance'] = [braycurtis(x, y) for (x, y) in zip(np.nan_to_num(question1_vectors),
                                                              np.nan_to_num(question2_vectors))]
    added_features.extend([typ+'_'+'skew_q1vec',typ+'_'+'skew_q2vec',typ+'_'+'kur_q1vec',typ+'_'+'kur_q2vec'])
    data[typ+'_'+'skew_q1vec'] = [skew(x) for x in np.nan_to_num(question1_vectors)]
    data[typ+'_'+'skew_q2vec'] = [skew(x) for x in np.nan_to_num(question2_vectors)]
    data[typ+'_'+'kur_q1vec'] = [kurtosis(x) for x in np.nan_to_num(question1_vectors)]
    data[typ+'_'+'kur_q2vec'] = [kurtosis(x) for x in np.nan_to_num(question2_vectors)]
    
    data[added_features].to_csv('../features/{}_{}_gensim_sent_features.csv'.format(typ,data_type))