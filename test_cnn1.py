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
from keras import backend as K


dataFile = './EEGdata_data.mat'
data = scio.loadmat(dataFile)
data_awake = data["EEGdata_awake"]
data_sleep = data["EEGdata_sleep"]

x = np.zeros([13, 2048, 180])
y = np.zeros([180])
y[0:90] = 1

x[:, :, 0:90] = np.array(data_awake, dtype='float32')
x[:, :, 90:] = np.array(data_sleep, dtype='float32')

x = x.transpose(2, 1, 0)
#[180,13,2048]

# standardizing
scaler = preprocessing.StandardScaler()
print('train_data_awake的维度：', x.shape)

x = x.reshape(180, 2048*13)

print('train_data_awake的维度：', x.shape)
x = scaler.fit_transform(x)

x = x.reshape(180, 2048, 13)

# 打乱数据
permutation = np.random.permutation(x.shape[0])
x = x[permutation, :, :]
y = y[permutation, ]

print('train_data的维度：', x.shape)

x_train = x[0:140, :, :]
y_train = y[0:140, ]
x_test = x[140:, :, :]
y_test = y[140:, ]

print('x_train的维度：', x_train.shape)
print('y_train的维度：', y_train.shape)



model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(2048, 13)),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(100, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
#训练网络：
history = model.fit(x_train, y_train, batch_size=50, epochs=10, verbose=2, validation_data=(x_test, y_test))

# print(history.history.keys())
# # 绘制训练 & 验证的准确率值
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()
# ##
# ##
# ### 绘制训练 & 验证的损失值
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Test'], loc='upper left')
# plt.show()


#测试网络
score = model.evaluate(x_test, y_test, batch_size=10, verbose=2, sample_weight=None)
print('test score:', score[0])
print('test accuracy:', score[1])
