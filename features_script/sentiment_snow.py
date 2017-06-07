from snownlp import SnowNLP
import pandas as pd
import argparse
import numpy as np

def sentence_sentiment_diff(row):
    try:
        s1=SnowNLP(str(row['clean_question1'])).sentiments
        s2=SnowNLP(str(row['clean_question2'])).sentiments
        return (s1-s2)*(s1-s2)
    except:
        return -1
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='stem/lem compute')
    parser.add_argument('--data',default="train", choices=['train','test'])

    main_args = parser.parse_args()
    data_type = main_args.data    

    # Execution of the generating features
    from tqdm import tqdm

    ## Generate features for the train set first

    df_train = pd.read_csv('../data/clean_{}.csv'.format(data_type), dtype={
                           'clean_question1': np.str,
                           'clean_question2': np.str
                       })
    
    results = []
    print("processing:", df_train.shape[0])
    for index, row in tqdm(df_train.iterrows()):
        res = {}
        res['sent'] = sentence_sentiment_diff(row)
        
        results.append(res)     

    train_locations = pd.DataFrame.from_dict(results)
    
    if data_type == 'train':
        train_locations['id'] = df_train.id
        train_locations.to_csv('../features/clean_train_sent.csv',index=None)
        
    if data_type == 'test':
        train_locations['test_id'] = df_train.test_id
        train_locations.to_csv('../features/clean_test_sent.csv',index=None)