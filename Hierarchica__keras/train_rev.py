import numpy as np
import pandas as pd
import re
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping,ModelCheckpoint

max_sen_len = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
EMBEDDING_FILE = 'F:\deep_leaning_action\Kaggle_jigsaw-toxic-comment-classification-challenge\glove\glove.840B.300d.txt'


def clean_str(string):
    """
    Tokenization/string cleaning for dataset
    Every dataset is lower cased except
    """
    string = re.sub(r"\\", "", string)
    string = re.sub(r"\'", "", string)
    string = re.sub(r"\"", "", string)
    return string.strip().lower()

train_df = pd.read_csv('input/labeledTrainData.tsv', sep='\t')

texts, labels = [], []
for i in range(train_df['review'].shape[0]):
    text = train_df['review'][i]
    texts.append(clean_str(text))
    labels.append(train_df['sentiment'][i])

labels = np.array(labels)

tok = Tokenizer(num_words=MAX_NUM_WORDS)
tok.fit_on_texts(texts)
seq = tok.texts_to_sequences(texts)
data = pad_sequences(seq,maxlen=max_sen_len)

data = np.expand_dims(data, axis=1)
# data.shape
# max_doc = data.shape[0]
idx = tok.word_index
labels_cat = np.expand_dims(labels,axis=1)
# labels_cat = to_categorical(np.asarray(labels))

x_train,x_val,y_train,y_val = train_test_split(data,labels_cat,test_size=0.12)
embeddings_index = {}
with open(EMBEDDING_FILE,encoding='utf8') as f:
    for line in f:
        values = line.rstrip().rsplit(' ')
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs

num_words = min(MAX_NUM_WORDS, len(idx) + 1)
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
for word, i in idx.items():
    if i >= MAX_NUM_WORDS:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

from han_model import *
model = createHierarchicalAttentionModel(max_sen_len, embWeights=embedding_matrix,
                                                        embeddingSize = 300, vocabSize = min(MAX_NUM_WORDS, len(idx) + 1))

filepath="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
callbacks_list = [checkpoint, early]

model.fit(x_train, y_train, batch_size=64, epochs=4,validation_data=(x_val, y_val),callbacks = callbacks_list,verbose=1)

# print('score ',score)
# print('score ',loss)
model.load_weights(filepath)

test_df = pd.read_csv('input/imdb/testData.tsv', sep='\t')

texts_test = []
for i in range(test_df['review'].shape[0]):
    text = test_df['review'][i]
    texts_test.append(clean_str(text))

test_seq = tok.texts_to_sequences(texts_test)
test_data = pad_sequences(test_seq,maxlen=max_sen_len)
test_data = np.expand_dims(test_data, axis=1)

pred = model.predict(test_data,batch_size=1024)

final_pred = []
for i in pred:
    if i>0.9:
        final_pred.append(1)
    else: final_pred.append(0)

subm = pd.DataFrame(data=[],columns=['id','sentiment'])#read_csv('sampleSubmission.csv')
subm['sentiment'] = final_pred
subm['id'] = test_df['id']
subm.to_csv('submission.csv',index=False)

