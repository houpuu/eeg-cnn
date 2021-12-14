import matplotlib.pyplot as plt
import numpy as np
import scipy.io as scio

dataFile = 'gen_data.npy'
data = np.load(dataFile)

print('gen_imgs shape', data.shape)
plt.plot(data[3, :, 0, 0])
plt.show()

filter_delta_file = 'C:\FangCloudV2\personal_space\eeg_program\eeg_function\FIR_beta_16.mat'
data2 = scio.loadmat(filter_delta_file)
filter_coe = data2['h']
height1, width = filter_coe.shape

data_output = np.convolve(filter_coe[0, 0:width], data[3, :, 0, 0], 'same')
plt.plot(data_output)
plt.show()