import pandas as pd
import networkx as nx
import numpy as np
import timeit
import argparse
from networkx import eccentricity
from networkx.algorithms import approximation as approx

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='magic features graph compute')
    parser.add_argument('--type',default="clean", choices=['clean', 'stem', 'lem', 'raw'],type=str)

    main_args = parser.parse_args()
    typ = main_args.type

    train_orig =  pd.read_csv('data/{}_train.csv'.format(typ), header=0)
    test_orig =  pd.read_csv('data/{}_test.csv'.format(typ), header=0)

    tic0=timeit.default_timer()
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

    comb['q1_hash'] = comb['{}_question1'.format(typ)].map(questions_dict)
    comb['q2_hash'] = comb['{}_question2'.format(typ)].map(questions_dict)
    
    added_features = []
    
    q1_vc = comb.q1_hash.value_counts().to_dict()
    q2_vc = comb.q2_hash.value_counts().to_dict()

    print("computing frequency")
    def try_apply_dict(x,dict_to_apply):
        try:
            return dict_to_apply[x]
        except KeyError:
            return 0
        
    # map to frequency space
    comb['q1_freq'] = comb['q1_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
    comb['q2_freq'] = comb['q2_hash'].map(lambda x: try_apply_dict(x,q1_vc) + try_apply_dict(x,q2_vc))
    comb['min_freq'] = np.minimum(comb['q1_freq'],comb['q2_freq'])
    comb['max_freq'] = np.maximum(comb['q1_freq'],comb['q2_freq'])
    comb['ratio_freq'] = comb['min_freq'] / np.maximum(1,comb['max_freq'])
    added_features.extend(['min_freq', 'max_freq', 'ratio_freq'])
    
    
    print("creating graph")
    g = nx.Graph()
    g.add_nodes_from(comb.q1_hash)
    g.add_nodes_from(comb.q2_hash)
    edges = list(comb[['q1_hash', 'q2_hash']].to_records(index=False))
    g.add_edges_from(edges)
    
    
    print("computing connected component")
    res = connected_components(g)
    comp_dict = {}
    for comp in res:
        size = len(comp)
        for elem in comp:
            comp_dict[elem] = size
            
    comb['cc_size'] = comb.apply(lambda row: comp_dict[row.q1_hash], axis=1)
    added_features.extend(['cc_size'])
    
    print("computing page rank")
    pr = nx.pagerank(g, alpha=0.9)
    
    comb['pr_q1'] = comb.apply(lambda row: pr[row.q1_hash], axis=1)
    comb['pr_q2'] = comb.apply(lambda row: pr[row.q2_hash], axis=1)
    comb['min_pr'] = np.minimum(comb['ecc_q1'],comb['pr_q2'])
    comb['max_pr'] = np.maximum(comb['ecc_q1'],comb['pr_q2'])
    comb['ratio_pr'] = comb['min_pr'] / np.maximum(1,comb['max_pr'])
    added_features.extend(['min_pr', 'max_pr', 'ratio_pr'])
    
    print("computing intersection count")
    def get_intersection_count(row):
        return(len(set(g.neighbors(row.q1_hash)).intersection(set(g.neighbors(row.q2_hash)))))
    
    comb['intersection_count'] = comb.apply(lambda row: get_intersection_count(row), axis=1)
    comb['ecc_q1'] = comb.apply(lambda row: len(set(g.neighbors(row.q1_hash))), axis=1)
    comb['ecc_q2'] = comb.apply(lambda row: len(set(g.neighbors(row.q2_hash))), axis=1)
    comb['min_ecc'] = np.minimum(comb['ecc_q1'],comb['ecc_q2'])
    comb['max_ecc'] = np.maximum(comb['ecc_q1'],comb['ecc_q2'])
    comb['ratio_ecc'] = comb['min_ecc'] / np.maximum(1,comb['max_ecc'])
    added_features.extend(['min_ecc', 'max_ecc', 'ratio_ecc'])
    
    train_comb = comb[comb['is_duplicate'] >= 0][['id'] + added_features].set_index('id')
    test_comb = comb[comb['is_duplicate'] < 0][['id'] + added_features].rename(columns={'id':'test_id'}).set_index('test_id')

    train_comb.to_csv('../features/{}_train_magic_features_graph.csv'.format(typ))
    test_comb.to_csv('../features/{}_test_magic_features_graph.csv'.format(typ))