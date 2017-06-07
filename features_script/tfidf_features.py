import argparse
import itertools
import pandas as pd
import pickle
from nltk import pos_tag
import numpy as np
from collections import Counter

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD, NMF



def sparse_cosine_vectors(vec1,vec2):
    norm_v1 = np.array(np.sqrt(vec1.multiply(vec1).sum(axis=1)))[:, 0]
    norm_v2 = np.array(np.sqrt(vec2.multiply(vec2).sum(axis=1)))[:, 0]
    num = np.array(vec1.multiply(vec2).sum(axis=1))[:, 0]
    den = norm_v1 * norm_v2
    num[np.where(den == 0)] = 1
    den[np.where(den == 0)] = 1
    v_score = 1 - num/den
    return v_score.astype(float)

def sparse_l1_norm(vec1,vec2):
    return np.array(np.abs(vec1 - vec2).sum(axis=1))[:, 0].astype(float)

def sparse_l2_norm(vec1,vec2):
    vec_diff = vec1 - vec2
    return np.array(vec_diff.multiply(vec_diff).sum(axis=1))[:, 0].astype(float)

def cosine_vectors(vec1,vec2):
    norm_v1 = np.sqrt((vec1*vec1).sum(axis=1))
    norm_v2 = np.sqrt((vec2*vec2).sum(axis=1))
    num = (vec1*vec2).sum(axis=1)
    den = norm_v1 * norm_v2
    num[np.where(den == 0)] = 1
    den[np.where(den == 0)] = 1
    v_score = 1 - num/den
    return v_score.astype(float)

def l1_norm(vec1,vec2):
    return np.array(np.abs(vec1 - vec2).sum(axis=1)).astype(float)

def l2_norm(vec1,vec2):
    vec_diff = vec1 - vec2
    return np.array((vec_diff*vec_diff).sum(axis=1)).astype(float)
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='stem/lem compute')
    parser.add_argument('--data',default="train", choices=['train','test'])
    parser.add_argument('--type',default="stem", choices=['clean','stem','lem'],type=str)
    parser.add_argument('--method',default='word',choices=['char','word'])
    parser.add_argument('--lsa',type=int,default=50)
    
    main_args = parser.parse_args()
    typ = main_args.type
    data_type = main_args.data
    lsa_dim = main_args.lsa
    method = main_args.method
    if method == 'word':
        NGRAMS = {'1':(1,1),'2':(2,2),'3':(3,3), '1-3':(1,3)}
    else:
        NGRAMS = {'1':(1,3),'2':(1,5)}
    
    print("processing {} data".format(data_type))
    index_col = 'id'
    if data_type == 'test':
        index_col = 'test_id'
    data = pd.read_csv('../data/{}_{}.csv'.format(typ,data_type), index_col=index_col)
        
    for i in range(1,3):
        print(typ+" len and nb words"+str(i))
        data[typ+'_question'+str(i)] = data[typ+'_question'+str(i)].astype(str)
        
    questions = pd.concat([data[typ+'_question1'],data[typ+'_question2']]).unique()
    
    if data_type == 'train':
        
        tfidfs = {}
        lsas = {}
        nmfs = {}
        
        print("generating tfidfs and lsas {}".format(lsa_dim))
        for gram in NGRAMS:
            
            print(">>> processing {}".format(gram))
            
            print('tfidf')
            tfidf = TfidfVectorizer(analyzer=method,use_idf=True,
                                    smooth_idf=True,min_df=3,
                                    ngram_range=NGRAMS[gram])

            matrix = tfidf.fit_transform(questions)

            tfidfs[gram] = tfidf

            print('svd')
            svd = TruncatedSVD(n_components=lsa_dim,random_state=42).fit(matrix)

            lsas[gram] = svd
            
            #print('nmf')
            #nmf = NMF(n_components=lsa_dim, random_state=1, alpha=.1, l1_ratio=.5).fit(matrix)
            #nmfs[gram] = nmf
        
        with open('../pickles/{}_train_tfidfs.pickle'.format(typ), 'wb') as handle:
            pickle.dump(tfidfs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        with open('../pickles/{}_train_lsas_{}.pickle'.format(typ,lsa_dim), 'wb') as handle:
            pickle.dump(lsas, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
        #with open('../pickles/{}_train_nmfs_{}.pickle'.format(typ,lsa_dim), 'wb') as handle:
        #    pickle.dump(nmfs, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
    else:
        
        print("loading tfidfs and lsas {}".format(lsa_dim))
        
        with open('../pickles/{}_train_tfidfs.pickle'.format(typ), 'rb') as handle:
            tfidfs = pickle.load(handle)
            
        with open('../pickles/{}_train_lsas_{}.pickle'.format(typ,lsa_dim), 'rb') as handle:
            lsas = pickle.load(handle)
        
        #with open('../pickles/{}_train_nmfs_{}.pickle'.format(typ,lsa_dim), 'rb') as handle:
        #    nmfs = pickle.load(handle)
            
    added_features = []
            
    for gram in NGRAMS:
        
        print(">>> creating feature tfidfs {}".format(gram))
        
        vec1 = tfidfs[gram].transform(data[typ+'_question1'])
        vec2 = tfidfs[gram].transform(data[typ+'_question2'])
        
        added_features.extend([typ+'_tfidf_{}_gram_cosine'.format(gram),
                               typ+'_tfidf_{}_gram_norml'.format(gram),
                               typ+'_tfidf_{}_gram_normll'.format(gram)])
        
        data[typ+'_tfidf_{}_gram_cosine'.format(gram)] = sparse_cosine_vectors(vec1,vec2)
        data[typ+'_tfidf_{}_gram_norml'.format(gram)] = sparse_l1_norm(vec1,vec2)
        data[typ+'_tfidf_{}_gram_normll'.format(gram)] = sparse_l2_norm(vec1,vec2)
        
        print(">>> creating feature {} lsa {}".format(gram,lsa_dim))
        
        svd_vec1 = lsas[gram].transform(vec1)
        svd_vec2 = lsas[gram].transform(vec2)
        
        added_features.extend([typ+'_lsa_{}_{}_gram_cosine'.format(lsa_dim,gram),
                               typ+'_lsa_{}_{}_gram_norml'.format(lsa_dim,gram),
                               typ+'_lsa_{}_{}_gram_normll'.format(lsa_dim,gram)])
        
        data[typ+'_lsa_{}_{}_gram_cosine'.format(lsa_dim,gram)] = cosine_vectors(svd_vec1,svd_vec2)
        data[typ+'_lsa_{}_{}_gram_norml'.format(lsa_dim,gram)] = l1_norm(svd_vec1,svd_vec2)
        data[typ+'_lsa_{}_{}_gram_normll'.format(lsa_dim,gram)] = l2_norm(svd_vec1,svd_vec2)
        
        added_features.extend([typ+'_lsa_{}_{}_gram_lsa0'.format(lsa_dim,gram),
                               typ+'_lsa_{}_{}_gram_lsa1'.format(lsa_dim,gram)])
        
        data[typ+'_lsa_{}_{}_gram_lsa0'.format(lsa_dim,gram)] = np.abs(svd_vec1[:,0] - svd_vec2[:,0])
        data[typ+'_lsa_{}_{}_gram_lsa1'.format(lsa_dim,gram)] = np.abs(svd_vec1[:,1] - svd_vec2[:,1])
        
        #nmf_vec1 = nmfs[gram].transform(vec1)
        #nmf_vec2 = nmfs[gram].transform(vec2)
        
        #added_features.extend([typ+'_nmf_{}_{}_gram_cosine'.format(lsa_dim,gram),
        #                       typ+'_nmf_{}_{}_gram_norml'.format(lsa_dim,gram),
        #                       typ+'_nmf_{}_{}_gram_normll'.format(lsa_dim,gram)])
        
        #data[typ+'_nmf_{}_{}_gram_cosine'.format(lsa_dim,gram)] = cosine_vectors(svd_vec1,svd_vec2)
        #data[typ+'_nmf_{}_{}_gram_norml'.format(lsa_dim,gram)] = l1_norm(svd_vec1,svd_vec2)
        #data[typ+'_nmf_{}_{}_gram_normll'.format(lsa_dim,gram)] = l2_norm(svd_vec1,svd_vec2)
    
    data[added_features].to_csv('../features/{}_{}_tfidf_lsa_{}_features.csv'.format(typ,data_type,lsa_dim))
            
    
            
            
            
            
        
        
    
    
        
    