import pandas as pd
import numpy as np
import timeit
import argparse
from collections import defaultdict
from nltk.corpus import stopwords
from collections import Counter

stops = set(stopwords.words("english"))

def word_match_share(q1, q2):
    q1 = str(q1).lower().split()
    q2 = str(q2).lower().split()
    q1words = {}
    q2words = {}
    for word in q1:
        if word not in stops:
            q1words[word] = 1
    for word in q2:
        if word not in stops:
            q2words[word] = 1
    if len(q1words) == 0 or len(q2words) == 0:
        # The computer-generated chaff includes a few questions that are nothing but stopwords
        return 0.
    shared_words_in_q1 = [w for w in q1words.keys() if w in q2words]
    shared_words_in_q2 = [w for w in q2words.keys() if w in q1words]
    R = (len(shared_words_in_q1) + len(shared_words_in_q2))/(len(q1words) + len(q2words))
    return R


def get_weight(count, eps=10000, min_count=2):
    if count < min_count:
        return 0
    else:
        return 1 / (count + eps)
    

if __name__ == "__main__":
    
    typ = 'raw'

    train_orig =  pd.read_csv('../data/{}_train.csv'.format(typ), header=0)
    train_orig['{}_question1'.format(typ)] = train_orig['{}_question1'.format(typ)].astype(str)
    train_orig['{}_question2'.format(typ)] = train_orig['{}_question2'.format(typ)].astype(str)
    test_orig =  pd.read_csv('../data/{}_test.csv'.format(typ), header=0)
    test_orig['{}_question1'.format(typ)] = test_orig['{}_question1'.format(typ)].astype(str)
    test_orig['{}_question2'.format(typ)] = test_orig['{}_question2'.format(typ)].astype(str)

    df1 = train_orig[['{}_question1'.format(typ)]].copy()
    df2 = train_orig[['{}_question2'.format(typ)]].copy()
    df1_test = test_orig[['{}_question1'.format(typ)]].copy()
    df2_test = test_orig[['{}_question2'.format(typ)]].copy()

    df2.rename(columns = {'{}_question2'.format(typ):'{}_question1'.format(typ)},inplace=True)
    df2_test.rename(columns = {'{}_question2'.format(typ):'{}_question1'.format(typ)},inplace=True)

    train_questions = df1.append(df2)
    train_questions = train_questions.append(df1_test)
    train_questions = train_questions.append(df2_test)
    #train_questions.drop_duplicates(subset = ['qid1'],inplace=True)
    train_questions.drop_duplicates(subset = ['{}_question1'.format(typ)],inplace=True)

    train_questions.reset_index(inplace=True,drop=True)
    questions_dict = pd.Series(train_questions.index.values,index=train_questions['{}_question1'.format(typ)].values).to_dict()
    train_cp = train_orig.copy()
    test_cp = test_orig.copy()
    try:
        train_cp.drop(['qid1','qid2'],axis=1,inplace=True)
    except:
        pass

    test_cp['is_duplicate'] = -1
    test_cp.rename(columns={'test_id':'id'},inplace=True)
    comb = pd.concat([train_cp,test_cp])
    comb = comb.reset_index()

    comb['q1_hash'] = comb['{}_question1'.format(typ)].map(questions_dict)
    comb['q2_hash'] = comb['{}_question2'.format(typ)].map(questions_dict)
    
    print('creating weights')
    train_qs = pd.Series(comb['{}_question1'.format(typ)].tolist() + comb['{}_question2'.format(typ)].tolist())
    words = [x for y in train_qs for x in y]
    counts = Counter(words)
    weights = {word: get_weight(count) for word, count in counts.items()}
    
    def tfidf_word_match_share(q1, q2):
        q1words = {}
        q2words = {}
        for word in q1:
            if word not in stops:
                q1words[word] = 1
        for word in q2:
            if word not in stops:
                q2words[word] = 1
        if len(q1words) == 0 or len(q2words) == 0:
            # The computer-generated chaff includes a few questions that are nothing but stopwords
            return 0

        shared_weights = [weights.get(w, 0) for w in q1words.keys() if w in q2words] + [weights.get(w, 0) for w in q2words.keys() if w in q1words]
        total_weights = [weights.get(w, 0) for w in q1words] + [weights.get(w, 0) for w in q2words]

        R = np.sum(shared_weights) / np.sum(total_weights)
        return R

    print('creating dictionaries')
    q_dict_wm = defaultdict(dict)
    q_dict_wm_idf = defaultdict(dict)
    
    for i in range(comb.shape[0]):
        wm_idf = tfidf_word_match_share(comb.ix[i,'{}_question1'.format(typ)], comb.ix[i,'{}_question2'.format(typ)])
        wm = word_match_share(comb.ix[i,'{}_question1'.format(typ)], comb.ix[i,'{}_question2'.format(typ)])
        q_dict_wm_idf[comb.ix[i,'q1_hash']][comb.ix[i,'q2_hash']] = wm_idf
        q_dict_wm_idf[comb.ix[i,'q2_hash']][comb.ix[i,'q1_hash']] = wm_idf
        q_dict_wm[comb.ix[i,'q1_hash']][comb.ix[i,'q2_hash']] = wm
        q_dict_wm[comb.ix[i,'q2_hash']][comb.ix[i,'q1_hash']] = wm
            
    def q1_q2_wm_ratio(row):
        q1 = q_dict_wm[row['q1_hash']]
        q2 = q_dict_wm[row['q2_hash']]
        inter_keys = set(q1.keys()).intersection(set(q2.keys()))
        if(len(inter_keys) == 0): return 0.
        inter_wm = 0.
        total_wm = 0.
        for q,wm in q1.items():
            if q in inter_keys:
                inter_wm += wm
            total_wm += wm
        for q,wm in q2.items():
            if q in inter_keys:
                inter_wm += wm
            total_wm += wm
        if(total_wm == 0.): return 0.
        return inter_wm/total_wm
    
    def q1_q2_wm_idf_ratio(row):
        q1 = q_dict_wm_idf[row['q1_hash']]
        q2 = q_dict_wm_idf[row['q2_hash']]
        inter_keys = set(q1.keys()).intersection(set(q2.keys()))
        if(len(inter_keys) == 0): return 0.
        inter_wm = 0.
        total_wm = 0.
        for q,wm in q1.items():
            if q in inter_keys:
                inter_wm += wm
            total_wm += wm
        for q,wm in q2.items():
            if q in inter_keys:
                inter_wm += wm
            total_wm += wm
        if(total_wm == 0.): return 0.
        return inter_wm/total_wm
        
    print('assign weights')
    comb['q1_q2_wm_ratio'] = comb.apply(q1_q2_wm_ratio, axis=1, raw=True)
    comb['q1_q2_wm_idf_ratio'] = comb.apply(q1_q2_wm_ratio, axis=1, raw=True)
    
    train_comb = comb[comb['is_duplicate'] >= 0][['id','q1_q2_wm_ratio','q1_q2_wm_idf_ratio']].set_index('id')
    test_comb = comb[comb['is_duplicate'] < 0][['id','q1_q2_wm_ratio','q1_q2_wm_idf_ratio']].rename(
                                                         columns={'id':'test_id'}).set_index('test_id')

    train_comb.to_csv('../features/{}_train_magic_features_wm.csv'.format(typ))
    test_comb.to_csv('../features/{}_test_magic_features_wm.csv'.format(typ))
    
            
            
            