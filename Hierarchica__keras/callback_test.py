import keras
from keras import layers
from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.layers import Dense,Input,Conv1D,MaxPool1D,GlobalMaxPooling1D,Embedding
from keras import Model
from  keras.callbacks import EarlyStopping,ModelCheckpoint

max_features = 10000
max_len = 300
embedding_dim = 128

(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)
x_train = sequence.pad_sequences(x_train,maxlen=max_len)
x_test = sequence.pad_sequences(x_test,maxlen=max_len)

def get_model():
    inp = Input(shape=(max_len,))
    x = Embedding(max_features,embedding_dim,input_length=max_len)(inp)
    y = Conv1D(32,7,activation='relu')(x)
    y = MaxPool1D(5)(y)
    y = Conv1D(32,7,activation='relu')(y)
    y = GlobalMaxPooling1D()(y)
    z = Dense(1,activation='sigmoid')(y)
    model = Model(inp,z)
    model.summary()
    return model

model = get_model()
call_back_list = [EarlyStopping(monitor='val_loss',patience=5),
                  ModelCheckpoint(filepath='my_model.h5',monitor='val_loss',save_best_only=True)]
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(x_train,y_train,epochs=10,batch_size=128,callbacks=call_back_list,validation_data=(x_test,y_test))

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