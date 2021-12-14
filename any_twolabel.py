import scipy.io as scio
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

import sys
sys.path.append(r'.')
import keras
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling1D

import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.optimizers import RMSprop

from keras.utils import to_categorical
from scipy import signal

from keras import backend as K
from keras.callbacks import LearningRateScheduler


def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

dataFile1 = '.\data\music\EEG_music.mat'
data1 = scio.loadmat(dataFile1)
train_input1 = data1['train_input']
train_output1 = data1['train_output']
height1, width, length = train_input1.shape

dataFile2 = '.\data\drelax\EEG_relax.mat'
data2 = scio.loadmat(dataFile2)
train_input2 = data2['train_input']
train_output2 = data2['train_output']
height2, width, length = train_input2.shape

train_input = np.zeros([height2+height1, width, length])
train_output = np.zeros([height2+height1, 1])

train_input[0:height1, :, :] = train_input1
train_output[0:height1, ] = np.ones([height1, 1])
train_input[height1:height1+height2, :, :] = train_input2
train_output[height1:height1+height2, ] = np.zeros([height2, 1])

print('train_input的维度：', train_input.shape)
print('train_output的维度：', train_output.shape)

# plt.plot(train_input[1, :, 1])
# plt.show()


height, width, length = train_input.shape
# 输入数据归一化
train_input = np.reshape(train_input, (-1, length))
print('train_input的维度：', train_input.shape)
ss = StandardScaler()
scaler = ss.fit(train_input)
print(scaler)
print(scaler.mean_)
scaled_train = scaler.transform(train_input)
print('scaled_train的维度：', scaled_train.shape)
train_input = np.reshape(scaled_train, (-1, width, length))
print('train_input的维度：', train_input.shape)
# print(train_input.shape)

# plt.plot(train_input[1, :, 1])
# plt.show()

# 打乱数据
permutation = np.random.permutation(train_input.shape[0])
train_input = train_input[permutation, :, :]
train_output = train_output[permutation]

# print('train_input的维度：', train_input.shape)
# print('train_output的维度：', train_output.shape)
# print('train_output：', train_output)

# 标签one hot化
lb = LabelBinarizer()
# train_output = lb.fit_transform(train_output)  # transfer label to binary value
train_output = to_categorical(train_output)  # transfer binary label to one-hot. IMPORTANT

print('train_input的维度：', train_input.shape)
print('train_output的维度：', train_output.shape)
print('train_output：', train_output)


# 设置测试集

kk = math.ceil(height * 0.8)
print('kk=:', kk)
x_test = train_input[kk:, :, :]
y_test = train_output[kk:]
train_input = train_input[0:kk, :, :]
train_output = train_output[0:kk]
height, width, length = train_input.shape



model_m = Sequential()
model_m.add(Conv1D(20, 10, activation='relu', input_shape=(width, length)))
model_m.add(Conv1D(20, 10, activation='relu'))
model_m.add(MaxPooling1D(2))
model_m.add(Conv1D(40, 8, activation='relu'))
model_m.add(Conv1D(40, 8, activation='relu'))
model_m.add(MaxPooling1D(2))
# model_m.add(Conv1D(100, 8, activation='relu'))
# model_m.add(Conv1D(100, 8, activation='relu'))
# model_m.add(MaxPooling1D(2))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(2, activation='softmax'))
print(model_m.summary())

# 回调函数Callbacks
callbacks_list = [
#     keras.callbacks.ModelCheckpoint(
#         filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
#         monitor='val_loss', save_best_only=True),
#     keras.callbacks.EarlyStopping(monitor='val_loss', patience=0)
]
# 配置模型，损失函数
model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy', precision, recall])

BATCH_SIZE = 30
EPOCHS = 1

# 训练模型
history = model_m.fit(train_input, train_output, batch_size=BATCH_SIZE, epochs=EPOCHS,
                      validation_data=(x_test, y_test),
                      verbose=2, shuffle=True)
                      # callbacks=callbacks_list,
                      # validation_split=0.2,


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
##
### 评估模型,输出预测结果
score = model_m.evaluate(x_test, y_test, batch_size=30, verbose=1, sample_weight=None)
print('Test score:', score[0])
print("test accuracy:", score[1])
# print('accuracy', accuracy)
# print('\ntest loss', loss)

# # 预测
# feature = model_m.predict(x_test, batch_size=10)
# print('predict feature:', feature, 'y_true', y_test)
