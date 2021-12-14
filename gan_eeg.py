from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import pandas as pd

import sys
import os
import numpy as np
import scipy.io as scio

class GAN():
    def __init__(self):
        # --------------------------------- #
        #   行28，列28，也就是mnist的shape
        # --------------------------------- #
        self.img_rows = 250
        self.img_cols = 16
        self.channels = 1
        # 28,28,1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 10
        # adam优化器
        optimizer = Adam(0.0002, 0.5)

        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy',
           optimizer=optimizer,
            metrics=['accuracy'])

        self.generator = self.build_generator()
        gan_input = Input(shape=(self.latent_dim,))
        img = self.generator(gan_input)
        # 在训练generate的时候不训练discriminator
        self.discriminator.trainable = False
        # 对生成的假图片进行预测
        validity = self.discriminator(img)
        self.combined = Model(gan_input, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)

    def build_generator(self):
        # --------------------------------- #
        #   生成器，输入一串随机数字
        # --------------------------------- #
        model = Sequential()

        model.add(Dense(250, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(500))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))

        model.add(Dense(1000))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def build_discriminator(self):
        # ----------------------------------- #
        #   评价器，对输入进来的图片进行评价
        # ----------------------------------- #
        model = Sequential()
        # 输入一张图片
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(500))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(250))
        model.add(LeakyReLU(alpha=0.2))
        # 判断真伪
        model.add(Dense(1, activation='sigmoid'))

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)


    def train(self, epochs, batch_size=100, sample_interval=50):
    # 获得数据
        dataFile1 = 'C:\PycharmProjects\eeg\data\drelax\EEG_relax_down4.mat'
        data1 = scio.loadmat(dataFile1)
        X_train = data1['train_input']
        # 进行标准化
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        print('X_train shape:', X_train.shape)


        # 创建标签
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # --------------------------- #
            #   随机选取batch_size个图片
            #   对discriminator进行训练
            # --------------------------- #
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            #plt.plot(imgs[0, :, 0, 0])
            #plt.show()

            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            #print('noise:', noise.shape)


            gen_imgs = self.generator.predict(noise)

            d_loss_real = self.discriminator.train_on_batch(imgs, valid)
            d_loss_fake = self.discriminator.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # --------------------------- #
            #  训练generator
            # --------------------------- #
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))
            g_loss = self.combined.train_on_batch(noise, valid)
            print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):

        r, c = 2, 2
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.generator.predict(noise)

        gen_imgs = 0.5 * gen_imgs + 0.5
        print('gen_imgs shape:', gen_imgs.shape)

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1
        fig.savefig("images/%d.png" % epoch)
        plt.close()
        np.save('gen_data.npy', gen_imgs)

        #plt.plot(gen_imgs[0, :, 0, 0])
        #plt.show()

if __name__ == '__main__':
    if not os.path.exists("./images"):
        os.makedirs("./images")
    gan = GAN()
    gan.train(epochs=3000, batch_size=100, sample_interval=50)