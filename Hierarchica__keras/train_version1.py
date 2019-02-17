import numpy as np
import pandas as pd
import re
import pickle
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping,ModelCheckpoint
import nltk
from nltk.tokenize import WordPunctTokenizer
from collections import defaultdict
###nltk分词
sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
word_tokenizer = WordPunctTokenizer()
##记录每个单词出现的频率
word_freq = defaultdict(int)

# max_sen_len = 100
MAX_NUM_WORDS = 20000
EMBEDDING_DIM = 300
VALIDATION_SPLIT = 0.1
EMBEDDING_FILE = 'input/glove.840B.300d.txt'


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
####数据预处理
texts, labels = [], []
for i in range(train_df['review'].shape[0]):
    text = train_df['review'][i]
    texts.append(clean_str(text))
    labels.append(train_df['sentiment'][i])

labels = np.array(labels)

#####统计每个单词出现
for text in texts:
    words = word_tokenizer.tokenize(text)
    for word in words:
        word_freq[word] +=1
print("establish vacb fished")
##保存词频表
with open('word_freq.pickle','wb') as g:
    pickle.dump(word_freq,g)
    print(len(word_freq))
    print("word_freq save finshed")
######构建vicab,次数小于5的去除
vocab = {}
i = 1
vocab['unknow'] = 0
for word,freq in word_freq.items():
    if freq>5:
        vocab[word]  = i
        i +=1
print(i)

max_sent_in_doc = 10
max_word_in_sen = 20
embedding_size = 300


def filt_my(X):
    data_x = []
    data_s = np.zeros((len(X),max_sent_in_doc,max_word_in_sen))
    sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
    for k,para in enumerate(X):
        doc = []
        sents = sent_tokenizer.tokenize(para)
        for i,sent in enumerate(sents):
            if i<max_sent_in_doc:
                word_to_index = []
                for j, word in enumerate(word_tokenizer.tokenize(sent)):
                    if j < max_word_in_sen:
                        if vocab.get(word,0)<MAX_NUM_WORDS:
                            data_s[k, i, j] = vocab.get(word, 0)
                            word_to_index.append(vocab.get(word, 0))

                doc.append(word_to_index)


        data_x.append(doc)

    return data_x,data_s

data_x,data_s = filt_my(texts)

# .reshape((-1,max_sent_in_doc,max_word_in_sen,max_sent_in_doc))
print("段落变句子二维变三维：",data_s.shape)

labels_cat = np.expand_dims(labels,axis=1)

x_train,x_val,y_train,y_val = train_test_split(data_s,labels_cat,test_size=0.12)
# embeddings_index = {}
# with open(EMBEDDING_FILE,encoding='utf8') as f:
#     for line in f:
#         values = line.rstrip().rsplit(' ')
#         word = values[0]
#         coefs = np.asarray(values[1:], dtype='float32')
#         embeddings_index[word] = coefs
#
# num_words = min(MAX_NUM_WORDS, len(vocab.items()) + 1)
# embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))
# for word, i in vocab.items():
#     if i >= MAX_NUM_WORDS:
#         continue
#     embedding_vector = embeddings_index.get(word)
#     if embedding_vector is not None:
#         # words not found in embedding index will be all-zeros.
#         embedding_matrix[i] = embedding_vector


from han_model import create_model#########多分类情况需要进入函数将dense改为n个类别
model = create_model(MAX_NUM_WORDS,embedding_size,max_sent_in_doc,max_word_in_sen,embWeights=None)

filepath="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
early = EarlyStopping(monitor="val_acc", mode="max", patience=5)
callbacks_list = [checkpoint, early]


history = model.fit(x_train, y_train, batch_size=128, epochs=10,validation_data=(x_val, y_val),callbacks = callbacks_list,verbose=1)


