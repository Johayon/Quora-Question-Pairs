import pandas as pd
import numpy as np
import argparse
import re 
from geotext import GeoText

def text_cleaner(text):
    
    try:
        # Clean the text
        text = re.sub(r"[^A-Za-z]", " ", text)
        # Remove unused space
        text = text.strip()
        return text
    except:
        return ""

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='stem/lem compute')
    parser.add_argument('--data',default="train", choices=['train','test'])

    main_args = parser.parse_args()
    data_type = main_args.data    

    # Execution of the generating features
    from tqdm import tqdm

    ## Generate features for the train set first

    df_train = pd.read_csv('../data/raw_{}.csv'.format(data_type), dtype={
                           'raw_question1': np.str,
                           'raw_question2': np.str
                       })

    subset = df_train.shape[0] # Remove the subsetting 

    results = []
    print("processing:", df_train[0:subset].shape)
    for index, row in tqdm(df_train[0:subset].iterrows()):
        q1 = text_cleaner(row['raw_question1']).title()
        q2 = text_cleaner(row['raw_question2']).title()

        rr = {}

        ct1 = []
        co1 = []
        cm1 = []

        if len(q1) > 0 :
            for word in q1.split():
                q1_place = GeoText(word)
                if len(q1_place.cities) > 0:
                    ct1.extend(q1_place.cities)
                if len(q1_place.countries) > 0:
                    co1.extend(q1_place.countries)
                if len(q1_place.country_mentions) > 0:
                    cm1.extend(q1_place.country_mentions)

        ct2 = []
        co2 = []
        cm2 = []
        if len(q2) > 0 :
            for word in q2.split():
                q2_place = GeoText(word)
                if len(q2_place.cities) > 0:
                    ct2.extend(q2_place.cities)
                if len(q2_place.countries) > 0:
                    co2.extend(q2_place.countries)
                if len(q2_place.country_mentions) > 0:
                    cm2.extend(q2_place.country_mentions)



        rr['cities_min'] = min(len(ct1),len(ct2))

        rr['cities_max'] = max(len(ct1),len(ct2))

        rr['cities_num'] = len(set(ct1).intersection(set(ct2)))

        rr['cities_num_mismatch_num'] = len(set(ct1).difference(set(ct2)))

        rr['countries_min'] = min(len(co1),len(co2))

        rr['countries_max'] = max(len(co1),len(co2))

        rr['countries_num'] = len(set(co1).intersection(set(co2)))

        rr['countries_mismatch_num'] = len(set(co1).difference(set(co2)))

        rr['countries_m_num'] = len(set(cm1).intersection(set(cm2)))

        rr['countries_m_mismatch_num'] = len(set(cm1).difference(set(cm2)))

        results.append(rr)     

    train_locations = pd.DataFrame.from_dict(results)
    
    if data_type == 'train':
        train_locations['id'] = df_train.id
        train_locations.to_csv('../features/raw_train_locations.csv',index=None)
        
    if data_type == 'test':
        train_locations['test_id'] = df_train.test_id
        train_locations.to_csv('../features/raw_test_locations.csv',index=None)