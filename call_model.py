import sys
sys.path.append(r'.')
import scipy.io as scio
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer
from keras import backend as K
from keras.utils import to_categorical
from keras.models import load_model

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

model = load_model('my_model.h5', custom_objects={'precision': precision, 'recall': recall})

dataFile1='E:\PycharmProjects\eeg\data\EEGi9.mat'
data = scio.loadmat(dataFile1)
data1 = data['EEG_data']
# print('data0:', data1.shape)
height1, width, length = data1.shape

dataFile2='E:\PycharmProjects\eeg\data\EEGi8.mat'
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


height, width, length = train_input.shape
# 输入数据归一化
train_input = np.reshape(train_input, (-1, length))
# print('train_input的维度：', train_input.shape)
ss = StandardScaler()
scaler = ss.fit(train_input)
# print(scaler)
# print(scaler.mean_)
scaled_train = scaler.transform(train_input)
# print('scaled_train的维度：', scaled_train.shape)
train_input = np.reshape(scaled_train, (-1, width, length))
# print('train_input的维度：', train_input.shape)
# print(train_input.shape)

# 打乱数据
permutation = np.random.permutation(train_input.shape[0])
train_input = train_input[permutation, :, :]
train_output = train_output[permutation]

# print('train_input的维度：', train_input.shape)
# print('train_output的维度：', train_output.shape)
# print('train_output：', train_output)

# 标签one hot化
lb = LabelBinarizer()
train_output = lb.fit_transform(train_output) # transfer label to binary value
train_output = to_categorical(train_output) # transfer binary label to one-hot. IMPORTANT

# print('train_input的维度：', train_input.shape)
# print('train_output的维度：', train_output.shape)
# print('train_output：', train_output)
# np.set_printoptions(threshold=np.inf)
# print(train_output)

x_test = train_input
y_test = train_output


score = model.evaluate(x_test, y_test, batch_size=10, verbose=1)

print('test_loss', score[0], 'test_accuracy', score[1])
