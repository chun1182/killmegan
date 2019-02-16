import os
import argparse
import numpy as np
import cv2
import h5py
import time
import keras
import keras.backend as K
from keras.utils import generic_utils
from keras.optimizers import Adam  # , SGD
import models_kilmebaybe as models
import tensorflow as tf
print('import2')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
K.set_session(sess)
np.random.seed(777)


def my_normalization(x):
    return x / 127.5 - 1


def my_inverse_normalization(x):
    return (x + 1.) / 2.


def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)


# グレーを3d化
def to3d(X):
    if X.shape[-1] == 3:
        return X
    b = X.transpose(3, 1, 2, 0)
    c = np.array([b[0], b[0], b[0]])
    return c.transpose(3, 1, 2, 0)


def plot_generated_batch(x_proc, x_raw, generator_model, batch_size, count, suffix):
    # 5枚づつ確認
    x_gen = generator_model.predict(x_raw)
    x_raw = my_inverse_normalization(x_raw)
    x_proc = my_inverse_normalization(x_proc)
    x_gen = my_inverse_normalization(x_gen)

    xs = to3d(x_raw[:5])
    xg = to3d(x_gen[:5])
    xr = to3d(x_proc[:5])
    xs = np.concatenate(xs, axis=1)
    xg = np.concatenate(xg, axis=1)
    xr = np.concatenate(xr, axis=1)
    xx = np.concatenate((xs, xg, xr), axis=0)

    cv2.imwrite("./figures/current_iter_"+str(count)+suffix+".png", xx*255)


def plot_generated_batch_test(x_raw, generator_model, batch_size, count):
    # 5枚づつ確認
    x_gen = generator_model.predict(x_raw)
    x_raw = my_inverse_normalization(x_raw)
    x_gen = my_inverse_normalization(x_gen)

    xs = to3d(x_raw[:5])
    xg = to3d(x_gen[:5])
    xs = np.concatenate(xs, axis=1)
    xg = np.concatenate(xg, axis=1)
    xx = np.concatenate((xs, xg), axis=0)
    cv2.imwrite("./figures/current_iter_"+str(count)+"test.png", xx*255)


def my_load_data(datasetpath):
    with h5py.File(datasetpath, "r") as hf:
        x_full_train = hf["train_data_gen"][:].astype(np.float32)
        x_full_train = my_normalization(x_full_train)
        x_sketch_train = hf["train_data_raw"][:].astype(np.float32)
        x_sketch_train = my_normalization(x_sketch_train)
        if len(x_sketch_train.shape) == 3:
            x_sketch_train = x_sketch_train.reshape(x_sketch_train.shape[0], x_sketch_train.shape[1], -1, 1)

        x_full_val = hf["val_data_gen"][:].astype(np.float32)
        x_full_val = my_normalization(x_full_val)
        x_sketch_val = hf["val_data_raw"][:].astype(np.float32)
        x_sketch_val = my_normalization(x_sketch_val)
        if len(x_sketch_val.shape) == 3:
            x_sketch_val = x_sketch_val.reshape(x_sketch_val.shape[0], x_sketch_val.shape[1], -1, 1)

        x_sketch_test = hf["test_data_raw"][:].astype(np.float32)
        x_sketch_test = my_normalization(x_sketch_test)
        if len(x_sketch_test.shape) == 3:
            x_sketch_test = x_sketch_test.reshape(x_sketch_test.shape[0], x_sketch_test.shape[1], -1, 1)

        return x_full_train, x_sketch_train, x_full_val, x_sketch_val, x_sketch_test


def extract_patches(x, patch_size):
    list_x = []
    list_row_idx = [(i*patch_size, (i+1)*patch_size) for i in range(x.shape[1] // patch_size)]
    list_col_idx = [(i*patch_size, (i+1)*patch_size) for i in range(x.shape[2] // patch_size)]
    for row_idx in list_row_idx:
        for col_idx in list_col_idx:
            list_x.append(x[:, row_idx[0]:row_idx[1], col_idx[0]:col_idx[1], :])
    return list_x


def get_disc_batch(procImage, rawImage, generator_model, batch_counter, patch_size):
    if batch_counter % 2 == 0:
        # produce an output
        X_disc = generator_model.predict(rawImage)
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)
        y_disc[:, 0] = 1
    else:
        X_disc = procImage
        y_disc = np.zeros((X_disc.shape[0], 2), dtype=np.uint8)

    X_disc = extract_patches(X_disc, patch_size)
    return X_disc, y_disc


def my_train(args):
    # create output finder
    if not os.path.exists(os.path.expanduser(args.datasetpath)):
        os.mkdir(args.datasetpath)
    # create figures
    if not os.path.exists('./figures'):
        os.mkdir('./figures')

    # load data
    procImage, rawImage, procImage_val, rawImage_val, rawImage_test = my_load_data(args.datasetpath)
    print('procImage.shape : ', procImage.shape)
    print('rawImage.shape : ', rawImage.shape)
    print('procImage_val : ', procImage_val.shape)
    print('rawImage_val : ', rawImage_val.shape)

    # パッチサイズと画像サイズを指定
    img_shape = rawImage.shape[-3:]
    print('img_shape : ', img_shape)
    patch_num = (img_shape[0] // args.patch_size) * (img_shape[1] // args.patch_size)
    disc_img_shape = (args.patch_size, args.patch_size, procImage.shape[-1])
    print('disc_img_shape : ', disc_img_shape)

    # train
    opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

    # load generator model
    generator_model = models.my_load_generator(img_shape, disc_img_shape)
    # load discriminator model
    discriminator_model = models.my_load_DCGAN_discriminator(img_shape, disc_img_shape, patch_num)

    generator_model.compile(loss='mae', optimizer=opt_discriminator)
    discriminator_model.trainable = False

    DCGAN_model = models.my_load_DCGAN(generator_model, discriminator_model, img_shape, args.patch_size)

    loss = [l1_loss, 'binary_crossentropy']
    loss_weights = [1E1, 1]
    DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

    discriminator_model.trainable = True
    discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

    # start training
    print('start training')
    for e in range(args.epoch):
        # 時間計測
        starttime = time.time()
        # シャッフル iterを作る
        perm = np.random.permutation(rawImage.shape[0])
        X_procImage = procImage[perm]
        X_rawImage = rawImage[perm]
        X_procImageIter = [X_procImage[i:i+args.batch_size] for i in range(0, rawImage.shape[0], args.batch_size)]
        X_rawImageIter = [X_rawImage[i:i+args.batch_size] for i in range(0, rawImage.shape[0], args.batch_size)]
        b_it = 0
        # 経過確認用
        progbar = generic_utils.Progbar(len(X_procImageIter)*args.batch_size)
        for (X_proc_batch, X_raw_batch) in zip(X_procImageIter, X_rawImageIter):
            b_it += 1

            X_disc, y_disc = get_disc_batch(X_proc_batch, X_raw_batch, generator_model, b_it, args.patch_size)
            raw_disc, _ = get_disc_batch(X_raw_batch, X_raw_batch, generator_model, 1, args.patch_size)
            x_disc = X_disc + raw_disc
            # update the discriminator
            disc_loss = discriminator_model.train_on_batch(x_disc, y_disc)

            # create a batch to feed the generator model 順番入れ替え
            idx = np.random.choice(procImage.shape[0], args.batch_size)
            X_gen_target, X_gen = procImage[idx], rawImage[idx]
            y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
            y_gen[:, 1] = 1

            # Freeze the discriminator
            discriminator_model.trainable = False
            gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
            # Unfreeze the discriminator
            discriminator_model.trainable = True

            progbar.add(args.batch_size, values=[
                ("D logloss", disc_loss),
                ("G tot", gen_loss[0]),
                ("G L1", gen_loss[1]),
                ("G logloss", gen_loss[2])
            ])

            # save images for visualization
            if b_it % (procImage.shape[0]//args.batch_size//2) == 0:
                plot_generated_batch(X_proc_batch, X_raw_batch, generator_model, args.batch_size, b_it, "training")
                idx = np.random.choice(procImage_val.shape[0], args.batch_size, replace=False)
                X_gen_target, X_gen = procImage_val[idx], rawImage_val[idx]
                plot_generated_batch(X_gen_target, X_gen, generator_model, args.batch_size, b_it, "validation")

                idx = np.random.choice(rawImage_test.shape[0], rawImage_test.shape[0], replace=False)
                X_gen = rawImage_test[idx]
                plot_generated_batch_test(X_gen, generator_model, args.batch_size, b_it)

        print("")
        print('Epoch %s/%s, Time: %s' % (e + 1, args.epoch, time.time() - starttime))


def main():
    parser = argparse.ArgumentParser(description='Train Font GAN')
    parser.add_argument('--datasetpath', '-d', type=str, required=True)
    parser.add_argument('--patch_size', '-p', type=int, default=64)
    parser.add_argument('--batch_size', '-b', type=int, default=32)
    parser.add_argument('--epoch', '-e', type=int, default=100)
    args = parser.parse_args()
    print('load start')
    K.set_image_data_format("channels_last")
    print('set image data format')
    my_train(args)


if __name__ == '__main__':
    main()
