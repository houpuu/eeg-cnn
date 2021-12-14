import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

dataFile1 = 'C:\PycharmProjects\eeg\data\drelax\EEG_relax_down4.mat'
data1 = scio.loadmat(dataFile1)
train_input = data1['train_input']
height1, width, length = train_input.shape
data_h = width

print('train_input的维度：', train_input.shape)
plt.plot(train_input[0, 0:width, 0])
plt.show()

filter_delta_file = 'C:\FangCloudV2\personal_space\eeg_program\eeg_function\FIR_beta_16.mat'
data2 = scio.loadmat(filter_delta_file)
filter_coe = data2['h']
height1, width = filter_coe.shape

print('filter_coe的维度：', filter_coe.shape)
plt.plot(filter_coe[0, 0:width])
plt.show()

data_output = np.convolve(filter_coe[0, 0:width], train_input[0:data_h, 0, 0], 'same')
print('data_putput的维度：', data_output.shape)
plt.plot(data_output)
plt.show()

np.save('data1.npy',data_output)