import scipy.io as scio
import numpy as np
import math
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelBinarizer

import sys
sys.path.append(r'.')
import keras
from keras.models import Sequential
from keras.layers import Embedding, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Conv2D
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
from keras.models import Model
from keras.models import load_model


model = load_model('.\model\EEGdata_ver2_model.h5')
print(model.summary())

# 权值
A = [0, 3, 6, 9]
plt.figure()
k = 1
for i in A:
    weight_Dense_1, bias_Dense_1 = model.get_layer(index=i).get_weights()
    height, width, length = weight_Dense_1.shape
    plt.subplot(2, 2, k)
    k = k+1
    for j in range(length):
        plt.plot(weight_Dense_1[:, 1, j])
plt.savefig('./EEG5_down4_model.jpg')
plt.show()

