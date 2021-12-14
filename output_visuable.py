import sys
sys.path.append(r'.')
import scipy.io as scio
import numpy as np
import math
import keras
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Conv2D, GlobalAveragePooling1D
from keras.layers import Reshape
from keras.layers import Dropout
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from scipy import signal

from keras import backend as K
from keras import Model
from keras.callbacks import LearningRateScheduler
import tensorflow as tf
import matplotlib.pyplot as plt


dataFile = '.\data\EEG_blue_down4.mat'
data = scio.loadmat(dataFile)
train_input = data['train_input']
train_output = data['train_output']

height, width, length = train_input.shape
# 输入数据归一化
train_input = np.reshape(train_input, (-1, length))
print('train_input的维度：', train_input.shape)
ss = StandardScaler()
scaler = ss.fit(train_input)
scaled_train = scaler.transform(train_input)
# print('scaled_train的维度：', scaled_train.shape)
train_input = np.reshape(scaled_train, (-1, width, length))
print('train_input的维度：', train_input.shape)
# print(train_input.shape)

# 打乱数据
permutation = np.random.permutation(train_input.shape[0])
train_input = train_input[permutation, :, :]
train_output = train_output[permutation]

print('train_input的维度：', train_input.shape)
print('train_output的维度：', train_output.shape)
# print('train_output：', train_output)

# 标签one hot化
lb = LabelBinarizer()
# train_output = lb.fit_transform(train_output) # transfer label to binary value
train_output = to_categorical(train_output) # transfer binary label to one-hot. IMPORTANT

print('train_input的维度：', train_input.shape)
print('train_output的维度：', train_output.shape)


height, width, length = train_input.shape

kk = math.ceil(height * 0.8)
print('kk=:', kk)
x_test = train_input[kk:, :, :]
y_test = train_output[kk:]
train_input = train_input[0:kk, :, :]
train_output = train_output[0:kk]
# x_test = train_input
# y_test = train_output



model = load_model('.\model\music_250_model.h5')
layer_model = Model(model.input, model.layers[11].output)
layer_model.summary()
feature1 = layer_model.predict(train_input, batch_size=50)
feature2 = layer_model.predict(x_test, batch_size=50)
print('feature shape:', feature1.shape)

A = [2, 5, 8, 11]
plt.figure()
k = 1
for i in A:
    layer_model = Model(model.input, model.layers[i].output)
    feature1 = layer_model.predict(train_input, batch_size=50)
    height, width, length = feature1.shape
    plt.subplot(2, 2, k)
    k = k+1
    for j in range(10):
        plt.plot(feature1[1, :, j])
# plt.savefig('./orange_down4_layer.jpg')
plt.show()