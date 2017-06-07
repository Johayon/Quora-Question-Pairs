from fuzzywuzzy import fuzz
import argparse
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
import itertools

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

    data[typ+'_question1'] = data[typ+'_question1'].astype(str)
    data[typ+'_question2'] = data[typ+'_question2'].astype(str)
        
    added_features = [typ+'_'+'fuzz_qratio', typ+'_'+'fuzz_WRatio', typ+'_'+'fuzz_partial_ratio',
                      typ+'_'+'fuzz_partial_token_set_ratio', typ+'_'+'fuzz_partial_token_sort_ratio',
                      typ+'_'+'fuzz_token_set_ratio', typ+'_'+'fuzz_token_sort_ratio']
    
    print('fuzzy features')
    data[typ+'_'+'fuzz_qratio'] = data.apply(
        lambda x: fuzz.QRatio(x[typ+'_question1'], x[typ+'_question2']), axis=1)
    data[typ+'_'+'fuzz_WRatio'] = data.apply(
        lambda x: fuzz.WRatio(x[typ+'_question1'], x[typ+'_question2']), axis=1)
    data[typ+'_'+'fuzz_partial_ratio'] = data.apply(
        lambda x: fuzz.partial_ratio(x[typ+'_question1'], x[typ+'_question2']), axis=1)
    data[typ+'_'+'fuzz_partial_token_set_ratio'] = data.apply(
        lambda x: fuzz.partial_token_set_ratio(x[typ+'_question1'], x[typ+'_question2']), axis=1)
    data[typ+'_'+'fuzz_partial_token_sort_ratio'] = data.apply(
        lambda x: fuzz.partial_token_sort_ratio(x[typ+'_question1'], x[typ+'_question2']), axis=1)
    data[typ+'_'+'fuzz_token_set_ratio'] = data.apply(
        lambda x: fuzz.token_set_ratio(x[typ+'_question1'], x[typ+'_question2']), axis=1)
    data[typ+'_'+'fuzz_token_sort_ratio'] = data.apply(
        lambda x: fuzz.token_sort_ratio(x[typ+'_question1'], x[typ+'_question2']), axis=1)
    
    data[added_features].to_csv('../features/{}_{}_fuzzy_features.csv'.format(typ,data_type))