import numpy as np
from keras.models import Model
from keras.layers import Input,Multiply
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import GRU
from keras.layers.wrappers import Bidirectional, TimeDistributed
from keras.layers.core import Dropout, Dense
from keras.engine.topology import Layer

from keras import backend as K
from keras import initializers,regularizers,constraints

def dot_product(x, kernel):
    """
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class Attention(Layer):
    def __init__(self,
                 W_regularizer=None,u_regularizer=None,b_regularizer=None,
                 W_constraint=None,u_constraint=None,b_constraint=None,
                 bias=True,**kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.u_regularizer = regularizers.get(u_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.u_constraint = constraints.get(u_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        super(Attention,self).__init__(**kwargs)
    def build(self,input_shape):
        assert len(input_shape)==3

        self.W = self.add_weight((input_shape[-1], input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)

        self.u = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_u'.format(self.name),
                                 regularizer=self.u_regularizer,
                                 constraint=self.u_constraint)

        super(Attention, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None

    def call(self, x, mask=None):
        uit = dot_product(x, self.W)

        if self.bias:
            uit += self.b

        uit = K.tanh(uit)
        ait = dot_product(uit, self.u)

        a = K.exp(ait)

        # apply mask after the exp. will be re-normalized next
        if mask is not None:
            # Cast the mask to floatX to avoid float64 upcasting in theano
            a *= K.cast(mask, K.floatx())

        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0], input_shape[-1]

def createHierarchicalAttentionModel(maxSeq, embWeights=None, embeddingSize = None, vocabSize = None,wordRnnSize=100, sentenceRnnSize=100,dropWordEmb = 0.2, dropWordRnnOut = 0.2, dropSentenceRnnOut = 0.5):
    wordInp = Input(shape=(maxSeq,),dtype='int32')
    if embWeights is None:
        x = Embedding(vocabSize,embeddingSize,input_length=maxSeq,trainable=True)(wordInp)
    else:
        x = Embedding(embWeights.shape[0],embWeights.shape[1],weights=[embWeights],trainable=False)(wordInp)
    # if dropWordEmb !=0.0:
    #     x = Dropout(dropWordEmb)(x)
    wordRNN = Bidirectional(GRU(wordRnnSize,return_sequences=True))(x)
    # if dropWordRnnOut>0.0:
    #     wordRNN = Dropout(dropWordRnnOut)(wordRNN)
    word_dense = TimeDistributed(Dense(200))(wordRNN)
    word_attention = Attention()(word_dense)
    modelSentEncoder = Model(wordInp,word_attention)
    ### oncument level logic
    docInp = Input(shape=(1,maxSeq),dtype='int32')
    sentEncoder = TimeDistributed(modelSentEncoder)(docInp)
    sentRNN = Bidirectional(GRU(sentenceRnnSize,return_sequences=True))(sentEncoder)
    # if dropSentenceRnnOut!=0:
    #     sentRNN = Dropout(dropSentenceRnnOut)(sentRNN)
    sent_dense = TimeDistributed(Dense(200))(sentRNN)
    sent_attention = Attention()(sent_dense)
    docOut = Dense(1,activation='sigmoid')(sent_attention)

    model = Model(input=docInp,output=docOut)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model



def create_model(vocabSize,EMBEDDING_DIM,max_sent_in_doc,max_word_in_sen,embWeights=None):
    char_input = Input(shape=(max_word_in_sen,), dtype='int32')
    if embWeights is None:
        embedding_layer = Embedding(vocabSize,EMBEDDING_DIM,input_length=max_word_in_sen,trainable=True)
    else:
        embedding_layer = Embedding(embWeights.shape[0],embWeights.shape[1],weights=[embWeights],trainable=False)

    char_sequences = embedding_layer(char_input)
    char_lstm = Bidirectional(GRU(100, return_sequences=True))(char_sequences)
    char_dense = TimeDistributed(Dense(200))(char_lstm)
    char_att = Attention()(char_dense)
    charEncoder = Model(char_input, char_att)

    words_input = Input(shape=(max_sent_in_doc,max_word_in_sen), dtype='int32')
    words_encoder = TimeDistributed(charEncoder)(words_input)
    words_lstm = Bidirectional(GRU(100, return_sequences=True))(words_encoder)
    words_dense = TimeDistributed(Dense(200))(words_lstm)
    words_att = Attention()(words_dense)
    preds = Dense(1, activation='sigmoid')(words_att)
    model = Model(words_input, preds)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model