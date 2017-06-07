########################################
## import packages
########################################
import os
import re
import csv
import codecs
import numpy as np
import pandas as pd

from string import punctuation
from collections import defaultdict

from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers.normalization import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint

from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import StratifiedKFold

import sys

########################################
## set directories and parameters
########################################
EMBEDDING_FILE = 'data/glove.840B.300d.txt'
TRAIN_DATA_FILE = 'data/corrected_train.csv'
TEST_DATA_FILE = 'data/corrected_test.csv'
MAX_SEQUENCE_LENGTH = 30
MAX_NB_WORDS = 300000
EMBEDDING_DIM = 300
N_FOLDS = 5

num_lstm = np.random.randint(175, 275)
num_dense = np.random.randint(100, 150)
rate_drop_lstm = 0.15 + np.random.rand() * 0.25
rate_drop_dense = 0.15 + np.random.rand() * 0.25

act = 'relu'
re_weight = False # whether to re-weight classes to fit the 17.5% share in test set

STAMP = 'lstm_%d_%d_%.2f_%.2f'%(num_lstm, num_dense, rate_drop_lstm, \
        rate_drop_dense)

########################################
## index word vectors
########################################
print('Indexing word vectors')

embeddings_index = {}
f = open(EMBEDDING_FILE,'r')
for line in f:
    try:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    except:
        continue
f.close()

print('Found %d word vectors of glove.' % len(embeddings_index))

########################################
## process texts in datasets
########################################
print('Processing text dataset')

train_df = pd.read_csv(TRAIN_DATA_FILE, index_col='id')
train_df.fillna('',inplace=True)
train_df['clean_question1'] = train_df['clean_question1'].astype(str)
train_df['clean_question2'] = train_df['clean_question2'].astype(str)

texts_1 = list(train_df['clean_question1'])
texts_2 = list(train_df['clean_question2'])
labels = np.array(train_df['is_duplicate'])
print('Found %s texts in train.csv' % len(texts_1))



test_df = pd.read_csv(TEST_DATA_FILE, index_col='test_id')
test_df.fillna('',inplace=True)
test_df['clean_question1'] = test_df['clean_question1'].astype(str)
test_df['clean_question2'] = test_df['clean_question2'].astype(str)

test_texts_1 = list(test_df['clean_question1'])
test_texts_2 = list(test_df['clean_question2'])
test_ids = test_df.index
print('Found %s texts in test.csv' % len(test_texts_1))

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts_1 + texts_2 + test_texts_1 + test_texts_2)

sequences_1 = tokenizer.texts_to_sequences(texts_1)
sequences_2 = tokenizer.texts_to_sequences(texts_2)
test_sequences_1 = tokenizer.texts_to_sequences(test_texts_1)
test_sequences_2 = tokenizer.texts_to_sequences(test_texts_2)

word_index = tokenizer.word_index
print('Found %s unique tokens' % len(word_index))

data_1 = pad_sequences(sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
data_2 = pad_sequences(sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
labels = np.array(labels)
print('Shape of data tensor:', data_1.shape)
print('Shape of label tensor:', labels.shape)

test_data_1 = pad_sequences(test_sequences_1, maxlen=MAX_SEQUENCE_LENGTH)
test_data_2 = pad_sequences(test_sequences_2, maxlen=MAX_SEQUENCE_LENGTH)
test_ids = np.array(test_ids)


########################################
## prepare embeddings
########################################
print('Preparing embedding matrix')

nb_words = min(MAX_NB_WORDS, len(word_index))+1

embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
for word, i in word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
print('Null word embeddings: %d' % np.sum(np.sum(embedding_matrix, axis=1) == 0))

########################################
## sample train/validation data
########################################
#np.random.seed(1234)
train_df['lstm_feature'] = 0
splits = StratifiedKFold(train_df['is_duplicate'],n_folds=N_FOLDS, shuffle=True)
preds_models = []

test_sum = (test_data_1 + test_data_2)/2
test_diff = np.abs(test_data_1 - test_data_2)/2

for i,(idx_train, idx_val) in enumerate(splits):

    data_1_train = (data_1[idx_train] + data_2[idx_train])/2
    data_2_train = np.abs(data_2[idx_train] - data_1[idx_train])/2
    labels_train = labels[idx_train]

    data_1_val = (data_1[idx_val] + data_2[idx_val])/2
    data_2_val = np.abs(data_2[idx_val] - data_1[idx_val])/2
    labels_val = labels[idx_val]

    weight_val = np.ones(len(labels_val))
    if re_weight:
        weight_val *= 0.472001959
        weight_val[labels_val==0] = 1.309028344

    ########################################
    ## define the model structure
    ########################################

    embedding_layer = Embedding(nb_words,
            EMBEDDING_DIM,
            weights=[embedding_matrix],
            input_length=MAX_SEQUENCE_LENGTH,
            trainable=False)

    lstm_layer = LSTM(num_lstm, dropout=rate_drop_lstm, recurrent_dropout=rate_drop_lstm)

    sequence_1_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_1 = embedding_layer(sequence_1_input)
    x1 = lstm_layer(embedded_sequences_1)

    sequence_2_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences_2 = embedding_layer(sequence_2_input)
    y1 = lstm_layer(embedded_sequences_2)

    merged = concatenate([x1, y1])
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    merged = Dense(num_dense, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)
    
    merged = Dense(10, activation=act)(merged)
    merged = Dropout(rate_drop_dense)(merged)
    merged = BatchNormalization()(merged)

    preds = Dense(1, activation='sigmoid')(merged)

    ########################################
    ## add class weight
    ########################################
    if re_weight:
        class_weight = {0: 1.309028344, 1: 0.472001959}
    else:
        class_weight = None

    ########################################
    ## train the model
    ########################################
    model = Model(inputs=[sequence_1_input, sequence_2_input], \
            outputs=preds)
    model.compile(loss='binary_crossentropy',
            optimizer='nadam',
            metrics=['acc'])
    #model.summary()
    print(STAMP)

    early_stopping =EarlyStopping(monitor='val_loss', patience=3)
    bst_model_path = 'lstms/' + STAMP + str(i) + '.h5'
    model_checkpoint = ModelCheckpoint(bst_model_path, save_best_only=True, save_weights_only=True)

    hist = model.fit([data_1_train, data_2_train], labels_train, \
            validation_data=([data_1_val, data_2_val], labels_val, weight_val), \
            epochs=200, batch_size=1024, shuffle=True, \
            class_weight=class_weight, callbacks=[early_stopping, model_checkpoint])

    model.load_weights(bst_model_path)
    bst_val_score = min(hist.history['val_loss'])
    print(bst_val_score)
    
    train_df.ix[idx_val,'lstm_feature'] = model.predict([data_1_val, data_2_val], batch_size=1024, verbose=1)   
    preds = model.predict([test_sum, test_diff], batch_size=1024, verbose=1)
    preds_models.append(preds)

########################################
## make the submission
########################################
    
test_df['lstm_feature']  = np.array(preds_models).T[0].mean(axis=1)

train_df[['lstm_feature']].to_csv('features/clean_train_lstm_glove_feature.csv')
test_df[['lstm_feature']].to_csv('features/clean_test_lstm_glove_feature.csv')