
from keras.datasets import boston_housing
(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()
train_data.shape
test_data.shape
########数据标准化
mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -=mean
test_data /=std

from keras import models
from keras import layers
from keras.layers import Input,Dense
from keras import Model
def build_model():
    inp = Input(shape=(train_data.shape[1],))
    x = Dense(64,activation='relu')(inp)
    y = Dense(64,activation='relu')(inp)
    pred = Dense(1)(y)########没有用激活函数所以可以拟合人任意范围的值
    model = Model(inp,pred)
    #
    # model = models.Sequential()
    # model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    # model.add(layers.Dense(4,activation='relu'))
    # model.add(layers.Dense(1))
    model.summary()
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model
#######k折验证


import numpy as np
from sklearn.cross_validation import KFold
# k=5
# num_val_samples = len(train_data)//k
num_epoches = 100
all_scores = []
all_mae_histories = []
folds = 5
seed = 2018
skf = KFold(train_data.shape[0], n_folds=folds, shuffle=True, random_state=seed)

for ii,(idx_train,idx_val) in enumerate(skf):
    val_data = train_data[idx_val]
    val_targets = train_targets[idx_val]
    partial_train_data = train_data[idx_train]
    partial_train_targets = train_targets[idx_train]

    model = build_model()
    history = model.fit(partial_train_data, partial_train_targets, validation_data=(val_data, val_targets),
                        epochs=num_epoches, batch_size=1, verbose=1)
    mae_his = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_his)
    # val_mse,val_mae = model.evaluate(val_data,val_targets,verbose=0)
    # all_scores.append(val_mae)

ave_mae_his = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epoches)]


import matplotlib.pyplot as plt

plt.plot(range(1,len(ave_mae_his)+1),ave_mae_his)
plt.xlabel('Epoches')
plt.ylabel('Validation mae')
plt.show()

def smooth_curve(points,factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous*factor+point*(1-factor))
        else:
            smoothed_points.append(point)

    return smoothed_points

smooth_mae_his = smooth_curve(ave_mae_his[10:])
plt.plot(range(1,len(smooth_mae_his)+1),smooth_mae_his)
plt.xlabel('Epoches')
plt.ylabel('Validation mae')
plt.show()

#########训练最终模型
model = build_model()
model.fit(train_data,train_targets,epochs=80,batch_size=16,verbose=0)
test_mse_score,test_mae_score = model.evaluate(test_data,test_targets)


