import re
from nltk.stem import PorterStemmer, WordNetLemmatizer
import pandas as pd
import argparse
from nltk import word_tokenize

def text_cleaner(text,stem=None):
    # Lower case
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)
    text = re.sub(r"(^| ).( |$)", " ", text)
    
    # Remove unused space
    text = text.strip()
    
    if stem == "lem":
        words = word_tokenize(text)
        # lemma
        lemmatized_words = [try_lem(word) for word in words]
        lemmatized_text = " ".join(lemmatized_words)
        return lemmatized_text
        
    if stem == "stem":
        words = word_tokenize(text)
        # stemmer
        stemmed_words = [try_stem(word) for word in words]
        stemmed_text = " ".join(stemmed_words)
        return stemmed_text
    
    return text

stemmer = PorterStemmer()
lemmatiser = WordNetLemmatizer()

def try_stem(word):
    try:
        return stemmer.stem(word)
    except:
        return word
    
def try_lem(word):
    try:
        return lemmatiser.lematize(word)
    except:
        return word
    
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
    data = pd.read_csv('../data/{}.csv'.format(data_type), index_col=index_col)
    data = data.fillna("")
    print(">>>>>>>>>> size: {}".format(data.shape[0]))
    for i in range(1,3):
        # clean
        print("cleanning question " +str(i))
        data[typ + '_question'+str(i)] = data['question'+str(i)].apply(lambda x: text_cleaner(x,typ))
        data.drop('question'+str(i),axis=1,inplace=True)
    data.to_csv('../data/{}_{}.csv'.format(typ,data_type))
    print("")
        