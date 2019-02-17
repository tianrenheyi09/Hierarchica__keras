import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D,GRU,LSTM
from keras.layers import Bidirectional, GlobalMaxPool1D,GlobalAveragePooling1D
from keras.layers import concatenate,BatchNormalization
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from model_function import Attention

train_df = pd.read_csv('input/labeledTrainData.tsv', sep='\t')
test_df = pd.read_csv("input/testData.tsv", sep='\t')
print("Train shape : ",train_df.shape)
print("Test shape : ",test_df.shape)

## split to train and val
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=2018)

## some config values
embed_size =300 # how big is each word vector
max_features = 10000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 70 # max number of words in a question to use

## fill up the missing values
train_X = train_df["review"].fillna("_##_").values
val_X = val_df["review"].fillna("_##_").values
test_X = test_df["review"].fillna("_##_").values

## Tokenize the sentences
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(train_X))
train_X = tokenizer.texts_to_sequences(train_X)
val_X = tokenizer.texts_to_sequences(val_X)
test_X = tokenizer.texts_to_sequences(test_X)

## Pad the sentences
train_X = pad_sequences(train_X, maxlen=maxlen)
val_X = pad_sequences(val_X, maxlen=maxlen)
test_X = pad_sequences(test_X, maxlen=maxlen)

## Get the target values
train_y = train_df['sentiment'].values
val_y = val_df['sentiment'].values

#shuffling the data
np.random.seed(2018)
trn_idx = np.random.permutation(len(train_X))
val_idx = np.random.permutation(len(val_X))

train_X = train_X[trn_idx]
val_X = val_X[val_idx]
train_y = train_y[trn_idx]
val_y = val_y[val_idx]


EMBEDDING_FILE = 'input/glove.840B.300d.txt'
word_index = tokenizer.word_index
print("found %s uniqe toens" %len(word_index))
##词嵌入矩阵
embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

print("found %s word vectors" %len(embeddings_index))

embedding_matrix = np.zeros((max_features,embed_size))
for word ,i in word_index.items():
    if(i<max_features):
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

def get_model():#####双向LSTM/gru加上attention
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features,embed_size,weights=[embedding_matrix],trainable=False)(inp)
    y = Bidirectional(LSTM(128, return_sequences=True))(x)
    z = Bidirectional(LSTM(32, return_sequences=True))(y)
    atten = Attention(maxlen)(z)
    d = Dense(64, activation="relu")(atten)
    preds = Dense(1, activation="sigmoid")(d)
    model = Model(inputs=inp,outputs=preds)
    model.summary()
    return model

def get_textcnn():
    inp = Input(shape=(maxlen,))
    emb = Embedding(max_features,embed_size,weights=[embedding_matrix],trainable=False)(inp)
    c1 = Conv1D(32,2)(emb)
    c2 = Conv1D(32,3)(emb)
    c3 = Conv1D(32,4)(emb)

    p1 = GlobalMaxPool1D()(c1)
    p2 = GlobalMaxPool1D()(c2)
    p3 = GlobalMaxPool1D()(c3)
    textcnn_list = [p1,p2,p3]

    cnn = concatenate(textcnn_list,axis=1)
    cnn = BatchNormalization()(cnn)
    cnn = Dropout(0.3)(cnn)
    fc1 = Dense(64,activation='relu')(cnn)
    output = Dense(1,activation='sigmoid')(fc1)
    model = Model(inputs=inp,outputs=output)
    model.summary()
    return model
def get_rcnn():
    inp = Input(shape=(maxlen,))
    emb = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    gru = Bidirectional(GRU(128,return_sequences=True))(emb)
    c1 = Conv1D(32,2)(gru)
    c2 = Conv1D(32,3)(gru)
    c3 = Conv1D(32,4)(gru)
    # c4 = Attention(maxlen)(gru)
    p1 = GlobalMaxPool1D()(c1)
    p2 = GlobalMaxPool1D()(c2)
    p3 = GlobalMaxPool1D()(c3)
    textcnn_list = [p1,p2,p3]
    # textcnn_list = [p1,p2,p3,c4]
    cnn = concatenate(textcnn_list, axis=1)

    # cnn = Attention(maxlen*3)(cnn)
    # cnn = BatchNormalization()(cnn)
    cnn = Dropout(0.3)(cnn)
    fc1 = Dense(64, activation='relu')(cnn)
    output = Dense(1, activation='sigmoid')(fc1)
    model = Model(inputs=inp, outputs=output)
    model.summary()
    return model
def get_fasttext():
    inp = Input(shape=(maxlen,))
    emb = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)

    x = GlobalAveragePooling1D()(emb)
    pred  = Dense(1,activation='sigmoid')(x)
    model = Model(inputs=inp,outputs=pred)
    model.summary()
    return model

model = get_model()

model.compile(loss='binary_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])

history = model.fit(train_X, train_y, batch_size=128, epochs=10, validation_data=(val_X, val_y))

import matplotlib.pyplot as plt
acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epoches = range(1,len(acc)+1)
plt.plot(epoches,acc,'bo',label='Training acc')
plt.plot(epoches,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epoches,loss,'bo',label='Training loss')
plt.plot(epoches,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()