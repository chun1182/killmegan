# -*- coding: utf-8 -*-
from __future__ import print_function, division
import os
import tensorflow as tf
import keras.backend as K
import numpy as np
import glob
import cv2
from keras.models import Sequential, Model  # load_model
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D, ZeroPadding2D
from keras.layers import Flatten, Dense, Dropout, Input, UpSampling2D, Reshape, Activation, LeakyReLU
from keras.optimizers import Adam  # SGD, Adagrad
from keras.preprocessing.image import ImageDataGenerator
# import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
np.random.seed(777)


class GAN:
    def __init__(self):
        # BVHデータ用の入力データサイズ
        self.img_rows = 128
        self.img_cols = 128
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        
        # 潜在変数の次元数 
        self.z_dim = 100

        # model最適化関数
        self.d_optimizer = Adam(lr=1e-4, beta_1=0.3)
        self.c_optimizer = Adam(lr=2e-4, beta_1=0.5)

        # discriminatorモデル
        self.discriminator = self.build_discriminator()
        self.discriminator.compile(loss='binary_crossentropy', optimizer=self.d_optimizer, metrics=['accuracy'])
        if os.path.exists('weights/discriminator_ganBVH.h5'):
            self.discriminator.load_weights('weights/discriminator_ganBVH.h5')
            print('load discriminator model')

        # Generatorモデル
        self.generator = self.build_generator()
        # generatorは単体で学習しないのでコンパイルは必要ない
        # self.generator.compile(loss='binary_crossentropy', optimizer=optimizer)

        self.combined = self.build_combined1()
        # self.combined = self.build_combined2()
        self.combined.compile(loss='binary_crossentropy', optimizer=self.c_optimizer)
        if os.path.exists('weights/generator_ganBVH.h5'):
            self.generator.load_weights('weights/generator_ganBVH.h5')
            print('load generator model')

    def build_generator(self):
        # generator model 深くする
        noise_shape = (self.z_dim,)
        model = Sequential()
        model.add(Dense(128 * 32 * 32, activation="relu", input_shape=noise_shape))
        model.add(Reshape((32, 32, 128)))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(128, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))

        model.add(UpSampling2D())
        model.add(Conv2D(64, kernel_size=3, padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(1, kernel_size=3, padding="same"))
        model.add(Activation("tanh"))
        model.summary()

        return model

    def build_discriminator(self):
        # discriminatorはあまり深くしすぎない。
        img_shape = (self.img_rows, self.img_cols, self.channels)
        model = Sequential()
        model.add(Conv2D(32, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
        model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Conv2D(256, kernel_size=3, strides=1, padding="same"))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))
        model.summary()
        return model

    # generatorとdiscriminatorの結合
    def build_combined1(self):
        self.discriminator.trainable = False
        model = Sequential([self.generator, self.discriminator])
        return model

    # generatorとdiscriminatorの結合functional版
    def build_combined2(self):
        z = Input(shape=(self.z_dim,))
        img = self.generator(z)
        self.discriminator.trainable = False
        valid = self.discriminator(img)
        model = Model(z, valid)
        model.summary()
        return model

    # 入力画像にIDGをかける
    def image_data_generate(self, images):
        # img.shape=(num, hight, weide, channel)
        datagen = ImageDataGenerator(width_shift_range=0.1, height_shift_range=0.1, horizontal_flip=True)
        batch = datagen.flow(images, batch_size=images.shape[0], shuffle=False).next()
        return batch

    def train(self, epochs, batch_size=128, save_interval=50):
        # samplesからデータの読み込み
        spath = './pic'

        all_images = [cv2.resize(cv2.imread(x, 0), (self.img_rows, self.img_rows)) for x in glob.glob(spath+'/*/*.png')]
        X_train = np.float32(all_images)
        print(X_train.shape)

        # 値を-1 to 1に規格化
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        # グレーの時は次元を増やすために使用
        X_train = np.expand_dims(X_train, axis=3)

        half_batch = int(batch_size / 2)
        num_batches = int(X_train.shape[0] / half_batch)
        print('Number of batches:', num_batches)
        
        for epoch in range(epochs):
            for iteration in range(num_batches):

                # ---------------------
                #  Discriminatorの学習
                # ---------------------

                # バッチサイズの半数をGeneratorから生成
                noise = np.random.normal(0, 1, (half_batch, self.z_dim))
                gen_imgs = self.generator.predict(noise)

                # バッチサイズの半数を教師データからピックアップ
                idx = np.random.randint(0, X_train.shape[0], half_batch)
                imgs = X_train[idx]
                imgs = self.image_data_generate(imgs)

                # discriminatorを学習
                # 本物データと偽物データは別々に学習させる
                # print('img,shape',imgs.shape, gen_imgs.shape)
                d_loss_real = self.discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
                d_loss_fake = self.discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))
                # それぞれの損失関数を平均
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

                # ---------------------
                #  Generatorの学習
                # ---------------------
                noise = np.random.normal(0, 1, (batch_size, self.z_dim))
                # 生成データの正解ラベルは本物（1） 
                # valid_y = np.array([1] * batch_size)

                # Train the generator
                # g_loss = self.combined.train_on_batch(noise, valid_y)
                g_loss = self.combined.train_on_batch(noise, np.ones((batch_size, 1)))

                # 10iterationごとに進捗の表示
                iters = iteration + epoch * num_batches
                if iters % (save_interval//10) == 0:
                    print("epoch:%d, iteration:%d,  [D loss: %f, acc.: %.2f%%] [G loss: %f]"
                          % (epoch, iteration, d_loss[0], 100 * d_loss[1], g_loss))
                # 指定した間隔で生成画像を保存
                if iters % save_interval == 0:
                    self.save_imgs(iters)
                    self.generator.save_weights('generator_ganBVH.h5')
                    self.discriminator.save_weights('discriminator_ganBVH.h5')

    def save_imgs(self, epoch):
        if not os.path.exists('./images'):
            os.mkdir('./images')
        # 生成画像を敷き詰めるときの行数、列数
        r, c = 5, 5
        # 画像生成、生成画像を0-1に再スケール
        noise = np.random.normal(0, 1, (r * c, self.z_dim))
        gen_imgs = self.generator.predict(noise)
        gen_imgs = 0.5 * gen_imgs + 0.5

        cnt = 0
        yoko_list = []
        for i in range(r):
            yoko = []
            for j in range(c):
                # gray_gen_imgs = cv2.cvtColor(gen_imgs[cnt] * 255, cv2.COLOR_BGR2GRAY)
                gray_gen_imgs = gen_imgs[cnt] * 255
                yoko.append(gray_gen_imgs)
                # 4000以上なら画像を保存
                if epoch >= 100000:
                    cv2.imwrite('images/BVH_iteration_{}_{}.png'.format(epoch, cnt), gray_gen_imgs)
                cnt += 1
            yoko_list.append(np.concatenate(yoko, axis=1))
        cv2.imwrite('images/BVH_matrix_iters_{}.png'.format(epoch), np.concatenate(yoko_list, axis=0))


if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=2000, batch_size=64, save_interval=100)
