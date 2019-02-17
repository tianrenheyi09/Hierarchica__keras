'''Trains a Hierarchical Attention Model on the IMDB sentiment classification task.
Modified from keras' examples/imbd_lstm.py.
'''
from __future__ import print_function
import numpy as np
import pandas as pd
# from model import createHierarchicalAttentionModel
np.random.seed(1337)  # for reproducibility

from keras.preprocessing import sequence
from keras.datasets import imdb
from keras.callbacks import EarlyStopping,ModelCheckpoint

MAX_NUM_WORDS = 20000
max_sen_len = 80  # cut texts after this number of words (among top max_features most common words)
batch_size = 32

print('Loading data...')
(X_train, y_train), (X_test, y_test) = imdb.load_data(nb_words=MAX_NUM_WORDS)
print(len(X_train), 'train sequences')
print(len(X_test), 'test sequences')

print('Pad sequences (samples x time)')
X_train = sequence.pad_sequences(X_train, maxlen=max_sen_len)
X_test = sequence.pad_sequences(X_test, maxlen=max_sen_len)
#add one extra dimention as the sentence (1 sentence per doc!)
X_train = np.expand_dims(X_train, axis=1)
X_test = np.expand_dims(X_test, axis=1)
print('X_train shape:', X_train.shape)
print('X_test shape:', X_test.shape)

print('Build model...')

from han_model import createHierarchicalAttentionModel
model = createHierarchicalAttentionModel(max_sen_len, embWeights=None,
                                                        embeddingSize = 300, vocabSize = MAX_NUM_WORDS)

filepath="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
callbacks_list = [checkpoint, early]

model.fit(X_train, y_train, batch_size=64, epochs=4,validation_data=(X_test, y_test),callbacks = callbacks_list,verbose=1)

# print('score ',score)
# print('score ',loss)
model.load_weights(filepath)


test_df = pd.read_csv('input/imdb/testData.tsv', sep='\t')

pred = model.predict(X_test,batch_size=1024)

final_pred = []
for i in pred:
    if i>0.9:
        final_pred.append(1)
    else: final_pred.append(0)

subm = pd.DataFrame(data=[],columns=['id','sentiment'])#read_csv('sampleSubmission.csv')
subm['sentiment'] = final_pred
subm['id'] = X_test['id']
subm.to_csv('submission.csv',index=False)


