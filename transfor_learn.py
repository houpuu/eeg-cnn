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


dataFile = 'C:\PycharmProjects\eeg\data\music\music_data\happy_3.mat'
data = scio.loadmat(dataFile)
train_input = data['train_input']
train_output = data['train_output']

height, width, length = train_input.shape
# 输入数据归一化
train_input = np.reshape(train_input, (-1, length))
print('train_input的维度：', train_input.shape)
'''
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



model = load_model('.\model\picture250_model.h5')
layer_model = Model(model.input, model.layers[11].output)
layer_model.summary()
feature1 = layer_model.predict(train_input, batch_size=100)
feature2 = layer_model.predict(x_test, batch_size=100)
print('feature shape:', feature1.shape)

# #定义模型的架构
layer_model1 = Sequential()

layer_model1.add(Conv1D(200, 3, activation='relu', input_shape=(11, 200)))
layer_model1.add(MaxPooling1D(2))
layer_model1.add(Dropout(0.2))
layer_model1.add(GlobalAveragePooling1D())

#layer_model1.add(GlobalAveragePooling1D(input_shape=(11, 200)))
layer_model1.add(Dense(5, activation='softmax'))
print(layer_model1.summary())
#
#编译模型并训练
layer_model1.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = layer_model1.fit(feature1, train_output,
                           batch_size=100,
                           epochs=200,
                           validation_data=(feature2, y_test),
                           verbose=1,
                           shuffle=True)


print(history.history.keys())
# 绘制训练 & 验证的准确率值
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
##
##
### 绘制训练 & 验证的损失值
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()


# # # #加载模型并计算测试集的分类准确性
# # # score = layer_model.evaluate(x_test, y_test, batch_size=50, verbose=1)
# # # print('test_loss', score[0], 'test_accuracy', score[1])
# #
# # # # 预测
# # # feature = layer_model.predict(x_test, batch_size=50)
# # # print('predict feature:', feature[1, :], 'y_true', y_test[1])
# # # # print('feature shape:', feature.shape)
# # # # print('y_test shape:', y_test.shape)
'''