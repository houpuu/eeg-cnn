import sys
sys.path.append(r'.')
import keras
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense
from keras.layers import Reshape
from keras.layers import Dropout
from keras.layers import GlobalAveragePooling1D
import numpy as np

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.optimizers import RMSprop

from sklearn.preprocessing import LabelBinarizer
from keras.utils import to_categorical
from scipy import signal
import scipy.io as scio
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


dataFile = './EEGdata_data.mat'
data = scio.loadmat(dataFile)
data_awake = data["EEGdata_awake"]
data_sleep = data["EEGdata_sleep"]

train_input = np.zeros([1440, 256, 13])
x_1 = np.zeros([13, 256, 1440])

for i in range(13):
    x = np.reshape(data_awake[i, :, :], [256, 8, 90])
    x = np.reshape(x, [256, 720])
    x_1[i, :, 0:720] = x
    x = np.reshape(data_sleep[i, :, :], [256, 8, 90])
    x = np.reshape(x, [256, 720])
    x_1[i, :, 720:1440] = x

train_input = x_1.transpose(2, 1, 0)

# plt.plot(x_1[1, :, 1])
# plt.show()
#
# plt.plot(train_input[1, :, 1])
# plt.show()

train_output = np.zeros([1440])
for i in range(720):
    train_output[i] = 1

# plt.plot(train_input[1, :, 1])
# plt.show()

height, width,length = train_input.shape
# ?????????????????????
train_input = np.reshape(train_input, (-1, length))
print('train_input????????????', train_input.shape)
ss = StandardScaler()
scaler=ss.fit(train_input)
print(scaler)
print(scaler.mean_)
scaled_train = scaler.transform(train_input)
print('scaled_train????????????',scaled_train.shape)
train_input = np.reshape(scaled_train,(-1,width,length))
print('train_input????????????',train_input.shape)
print(train_input.shape)

# plt.plot(train_input[1, :, 1])
# plt.show()

# ????????????
permutation = np.random.permutation(train_input.shape[0])
train_input = train_input[permutation, :, :]
train_output = train_output[permutation]

print('train_input????????????', train_input.shape)
print('train_output????????????', train_output.shape)
print('train_output???', train_output)

# ??????one hot???
lb = LabelBinarizer()
train_output = lb.fit_transform(train_output) # transfer label to binary value
train_output = to_categorical(train_output) # transfer binary label to one-hot. IMPORTANT

print('train_input????????????', train_input.shape)
print('train_output????????????', train_output.shape)
print('train_output???', train_output)

# np.set_printoptions(threshold=np.inf)
# print(train_output)

# ???????????????
x_test = train_input[1400:1440, :, :]
y_test = train_output[1400:1440]
train_input = train_input[0:1400, :, :]
train_output = train_output[0:1400]
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

# ????????????Callbacks
callbacks_list = [
#     keras.callbacks.ModelCheckpoint(
#         filepath='best_model.{epoch:02d}-{val_loss:.2f}.h5',
#         monitor='val_loss', save_best_only=True),
#     keras.callbacks.EarlyStopping(monitor='val_loss', patience=0)
]
# ????????????
model_m.compile(loss='categorical_crossentropy',
                optimizer='adam', metrics=['accuracy', precision, recall])

BATCH_SIZE = 80
EPOCHS = 10


history = model_m.fit(train_input,
                      train_output,
                      batch_size=BATCH_SIZE,
                      epochs=EPOCHS,
                      callbacks=callbacks_list,
                      validation_split=0.2,
                      verbose=2)


print(history.history.keys())
# ???????????? & ?????????????????????
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
##
##
### ???????????? & ??????????????????
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
##
### ????????????,??????????????????
print(model_m.evaluate(x_test, y_test, batch_size=10))
# print('accuracy', accuracy)
# print('\ntest loss', loss)
