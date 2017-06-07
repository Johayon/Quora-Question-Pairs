import argparse
import itertools
import pandas as pd
import pickle
from nltk import pos_tag
import numpy as np
from nltk.corpus import stopwords

stops = stopwords.words("english")
stops_set = set(stops)

def word_share(words1,words2):
    
    w1 = set(words1)
    w2 = set(words2)
    
    shared = w1 & w2
    diff = w1 ^ w2
    
    return pd.Series([shared, diff])



def count_tags(pos_tags):
    types = [tag[1][:2] for tag in pos_tags]
    res = [types.count(tag) for tag in ['NN','JJ','RB','VB']]
    return pd.Series(res)



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

    print(typ+' common pos word ')
    data[typ+'_common_pos'] = data[typ+'_common_set_stop'].apply(pos_tag)
    
    print(typ+' diff pos word ')
    data[typ+'_diff_pos'] = data[typ+'_diff_set_stop'].apply(pos_tag)
    
    print(typ+' count pos word diff')
    data[[typ+'_NN_diff',typ+'_JJ_diff',
          typ+'_RB_diff',typ+'_VB_diff']] = data[typ+'_diff_pos'].apply(count_tags)
    
    print(typ+' count pos word common')
    data[[typ+'_NN_common',typ+'_JJ_common',
          typ+'_RB_common',typ+'_VB_common']] = data[typ+'_common_pos'].apply(count_tags)
    
    for stop in ['','_stop']:
        print(typ+' set word '+stop)
        data[typ+'_common_set_len'+stop] = data[typ+'_common_set'+stop].apply(len)
        data[typ+'_diff_set_len'+stop] = data[typ+'_diff_set'+stop].apply(len)
    
    
    for tag in ['NN','JJ','RB','VB']:
        added_features.extend([typ+'_'+tag+'_diff_norm',typ+'_'+tag+'_diff_norm_ll',typ+'_'+tag+'_common_norm_ll',
                               typ+'_'+tag+'_diff_norm_lll',typ+'_'+tag+'_common_norm_lll'])
        data[typ+'_'+tag+'_diff_norm'] = (data[typ+'_'+tag+'_diff'] / 
                                          np.maximum(1.0, data[typ+'_'+tag+'_diff'] + 
                                                         data[typ+'_'+tag+'_common']))
        data[typ+'_'+tag+'_diff_norm_ll'] = (data[typ+'_'+tag+'_diff'] /  
                                             np.maximum(1.0,data[typ+'_diff_set_len_stop']))
        data[typ+'_'+tag+'_common_norm_ll'] = (data[typ+'_'+tag+'_common'] /  
                                               np.maximum(1.0, data[typ+'_common_set_len_stop']))
        data[typ+'_'+tag+'_diff_norm_lll'] = (data[typ+'_'+tag+'_diff'] / 
                                              np.maximum(1.0, data[typ+'_diff_set_len_stop'] + 
                                                              data[typ+'_common_set_len_stop']))
        data[typ+'_'+tag+'_common_norm_lll'] = (data[typ+'_'+tag+'_common'] / 
                                                np.maximum(1.0, data[typ+'_diff_set_len_stop'] + 
                                                          data[typ+'_common_set_len_stop']))
        
    data[added_features].to_csv('../features/{}_{}_word_pos_features.csv'.format(typ,data_type))