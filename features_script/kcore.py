import pandas as pd
import networkx as nx
import numpy as np
import timeit
import argparse
from networkx import eccentricity
from networkx.algorithms import approximation as approx

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='magic features graph compute')
    parser.add_argument('--type',default="clean", choices=['clean','stem','lem','raw'],type=str)
    parser.add_argument('--use_cache', action='store_true')

    main_args = parser.parse_args()
    typ = main_args.type
    
    train_orig =  pd.read_csv('../data/{}_train.csv'.format(typ), header=0)
    test_orig =  pd.read_csv('../data/{}_test.csv'.format(typ), header=0)

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
    
    if not main_args.use_cache:
        g = nx.Graph()
        g.add_nodes_from(comb.q1_hash)
        g.add_nodes_from(comb.q2_hash)
        edges = list(comb[['q1_hash', 'q2_hash']].to_records(index=False))
        g.add_edges_from(edges)
        g.remove_edges_from(g.selfloop_edges())

        print(g.number_of_nodes()) # 4789604
        print(g.number_of_edges()) # 2743365 (after self-edges)

        df_output = pd.DataFrame(data=g.nodes(), columns=["qid"])

        print("df_output.shape:", df_output.shape)

        NB_CORES = 20

        for k in range(2, NB_CORES + 1):
            fieldname = "kcore{}".format(k)
            print("fieldname = ", fieldname)
            ck = nx.k_core(g, k=k).nodes()
            print("len(ck) = ", len(ck))
            df_output[fieldname] = 0
            df_output.ix[df_output.qid.isin(ck), fieldname] = k

        df_output.to_csv("pickles/question_kcores.csv", index=None)
        df_cores = pd.read_csv("pickles/question_kcores.csv", index_col="qid")
        df_cores.index.names = ["qid"]
        df_cores['max_kcore'] = df_cores.apply(lambda row: max(row), axis=1)
        df_cores[['max_kcore']].to_csv("pickles/question_max_kcores.csv") # with index
    
    
    cores_dict = pd.read_csv("pickles/question_max_kcores.csv", index_col="qid").to_dict()["max_kcore"]
    
    def gen_qid1_max_kcore(row):
        return cores_dict[row['q1_hash']]
    
    def gen_qid2_max_kcore(row):
        return cores_dict[row['q2_hash']]

    comb["qid1_max_kcore"] = comb.apply(gen_qid1_max_kcore, axis=1)
    comb["qid2_max_kcore"] = comb.apply(gen_qid2_max_kcore, axis=1)
    
    comb['min_kcore'] = np.minimum(comb['qid1_max_kcore'],comb['qid2_max_kcore'])
    comb['max_kcore'] = np.maximum(comb['qid1_max_kcore'],comb['qid2_max_kcore'])
    comb['ratio_kcore'] = comb['min_kcore'] / np.maximum(1,comb['max_kcore'])
    
    train_comb = comb[comb['is_duplicate'] >= 0][['id','min_kcore','max_kcore','ratio_kcore']].set_index('id')
    test_comb = comb[comb['is_duplicate'] < 0][['id','min_kcore','max_kcore','ratio_kcore']].rename(
                                                      columns={'id':'test_id'}).set_index('test_id')

    train_comb.to_csv('../features/{}_train_magic_features_kcore.csv'.format(typ))
    test_comb.to_csv('../features/{}_test_magic_features_kcore.csv'.format(typ))
