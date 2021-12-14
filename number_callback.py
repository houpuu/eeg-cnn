import sys
sys.path.append(r'.')

import scipy.io as scio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
from scipy import signal


import keras
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling1D
from keras.utils import plot_model
from keras.optimizers import RMSprop
from keras import backend as K
from keras.callbacks import LearningRateScheduler
from keras.callbacks import TensorBoard
from keras.callbacks import ModelCheckpoint
from keras.utils import to_categorical



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


dataFile1 = 'E:\PycharmProjects\eeg\data\EEGi1.mat'
data = scio.loadmat(dataFile1)
data1 = data['EEG_data']
# print('data0:', data1.shape)
height1, width, length = data1.shape

dataFile2 = 'E:\PycharmProjects\eeg\data\EEGi2.mat'
data = scio.loadmat(dataFile2)
data2 = data['EEG_data']
# print('data1:', data2.shape)
height2, width, length = data2.shape

train_input = np.zeros([height1+height2, width, length])
train_output = np.zeros([height1+height2])
for i in range(height1):
    train_output[i] = 1

train_input[0:height1, :, :] = data1
train_input[height1:height1+height2, :, :] = data2
train_input = train_input.transpose(0, 2, 1)


plt.plot(train_input[1, :, 1])
plt.show()


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
print(train_input.shape)

plt.plot(train_input[1, :, 1])
plt.show()

# 打乱数据
permutation = np.random.permutation(train_input.shape[0])
train_input = train_input[permutation, :, :]
train_output = train_output[permutation]

print('train_input的维度：', train_input.shape)
print('train_output的维度：', train_output.shape)
# print('train_output：', train_output)

# 标签one hot化
lb = LabelBinarizer()
train_output = lb.fit_transform(train_output) # transfer label to binary value
train_output = to_categorical(train_output) # transfer binary label to one-hot. IMPORTANT

print('train_input的维度：', train_input.shape)
print('train_output的维度：', train_output.shape)
print('train_output：', train_output)
# np.set_printoptions(threshold=np.inf)
# print(train_output)

# 设置测试集

x_test = train_input[100:130, :, :]
y_test = train_output[100:130]
train_input = train_input[0:100, :, :]
train_output = train_output[0:100]

height, width, length = train_input.shape

model_m = Sequential()
model_m.add(Conv1D(100, 10, activation='relu', input_shape=(width, length)))
model_m.add(Conv1D(100, 10, activation='relu'))
model_m.add(MaxPooling1D(2))
model_m.add(Conv1D(100, 8, activation='relu'))
model_m.add(Conv1D(100, 8, activation='relu'))
model_m.add(MaxPooling1D(2))
model_m.add(GlobalAveragePooling1D())
model_m.add(Dropout(0.5))
model_m.add(Dense(2, activation='softmax'))
print(model_m.summary())

# 回调函数Callbacks
callbacks_list = [
    keras.callbacks.TensorBoard(log_dir='./logs'),
    keras.callbacks.ModelCheckpoint(
        filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
        monitor='val_loss', verbose=0,  save_best_only=True,
        save_weights_only=True, mode='auto', period=1),
    # keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10, verbose=0, mode='auto',
    #                                   min_delta=0.0001, cooldown=0, min_lr=0),
    # keras.callbacks.EarlyStopping(monitor='val_loss', patience=0)
]
# 损失函数
model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy', precision, recall])

BATCH_SIZE = 20
EPOCHS = 100


history = model_m.fit(train_input,
                      train_output,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=1)


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
score = model_m.evaluate(x_test, y_test, batch_size=10, verbose=1, sample_weight=None)
print('Test score:', score[0])
print("test accuracy:", score[1])

# 预测
feature = model_m.predict(x_test, batch_size=10)
model_m.save('my_model.h5')