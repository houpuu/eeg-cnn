#load library
import numpy as np
from scipy import signal
from sklearn import preprocessing
import scipy.io as scio
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
from keras.layers import Conv2D
import tensorflow as tf

from sklearn.preprocessing import StandardScaler

dataFile = 'eegdata_256.mat'
data = scio.loadmat(dataFile)
train_input = data['eegdata_x']
train_input = train_input.transpose(2, 0, 1)

y = np.zeros([1440])
for i in range(720):
    y[i] = 1

plt.plot(train_input[1, 1, :])
plt.show()


height, width, length = train_input.shape


# 输入数据归一化
train_input = np.reshape(train_input, (-1, length))
print('train_input的维度：', train_input.shape)
ss = StandardScaler()
scaler=ss.fit(train_input)
print(scaler)
print(scaler.mean_)
scaled_train = scaler.transform(train_input)
print('scaled_train的维度：', scaled_train.shape)
train_input = np.reshape(scaled_train, (-1,width,length))
print('train_input的维度：', train_input.shape)

print(train_input.shape)

plt.plot(train_input[1, 1, :])
plt.show()

# 打乱数据
permutation = np.random.permutation(train_input.shape[0])
train_input = train_input[permutation, :, :]
y = y[permutation, ]

#print('train_data_awake的维度：',x.shape)

x_train=train_input[0:1400, :, :]
y_train=y[0:1400]
x_test=train_input[1400:, :, :]
y_test=y[1400:]

print('x_train的维度：', x_train.shape)
print('y_train的维度：', y_train.shape)

import tensorflow as tf

model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(13, 256)),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#训练网络：
model.fit(x_train, y_train, batch_size=50, epochs=10, verbose=2, validation_data=(x_test, y_test))

#测试网络
score = model.evaluate(x_test, y_test, batch_size=40, verbose=2, sample_weight=None)
print('test score:', score[0])
print('test accuracy:', score[1])
