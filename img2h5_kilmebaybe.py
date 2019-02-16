import numpy as np
import glob
import argparse
import h5py

from keras.preprocessing.image import load_img, img_to_array
import cv2


def main():
    # データセットが入っているフォルダを指定
    # inpath/finders/files
    # args.outpath+'.hdf5'ができる
    # canny-エッジ検出、gray=グレー色塗り
    parser = argparse.ArgumentParser()
    parser.add_argument('--inpath', '-i', required=True)
    parser.add_argument('--outpath', '-o', required=True)
    parser.add_argument('--trans', '-t', default='canny')
    args = parser.parse_args()

    finders = glob.glob(args.inpath+'/*')
    print(finders)
    imgs = []
    gimgs = []

    for finder in finders:
        files = glob.glob(finder+'/*')
        print('files:', len(glob.glob(finder+'/*')))
        for imgfile in files:
            # imgsに元画像を追加
            imgarray = cv2.resize(cv2.imread(imgfile), (128, 128))
            imgs.append(imgarray)
            if args.trans == 'gray':
                # gimsにグレー画像追加
                grayimgarray = cv2.cvtColor(cv2.resize(cv2.imread(imgfile), (128, 128)), cv2.COLOR_BGR2GRAY)
                gimgs.append(grayimgarray)
            elif args.trans == 'canny':
                grayimg = cv2.cvtColor(cv2.imread(imgfile), cv2.COLOR_BGR2GRAY)
                gray_canny_xy = cv2.Canny(grayimg, 30, 30)
                gray_canny_xy = cv2.bitwise_not(gray_canny_xy)
                # gimsにエッジ画像追加
                gimgs.append(gray_canny_xy.reshape(30, 30, 1))
        print('imgs:', len(imgs), 'gimgs:', len(gimgs))

    timgs = []
    t_files = glob.glob('test/*')
    for imgfile in t_files:
        # imgsに元画像を追加
        timgarray = cv2.resize(cv2.imread(imgfile, 0), (128, 128))
        timgs.append(timgarray)

    print('imgs:', len(imgs), 'gimgs:', len(gimgs))

    # シャッフル
    perm = np.random.permutation(len(imgs))
    imgs = np.array(imgs)[perm]
    gimgs = np.array(gimgs)[perm]
    timgs = np.array(timgs)
    # 90枚ごとにvalファイルを振り分け
    threshold = len(imgs)//10*9
    vimgs = imgs[threshold:]
    vgimgs = gimgs[threshold:]
    imgs = imgs[:threshold]
    gimgs = gimgs[:threshold]
    print('shapes')
    print('gen imgs : ', imgs.shape)
    print('raw imgs : ', gimgs.shape)
    print('val gen  : ', vimgs.shape)
    print('val raw  : ', vgimgs.shape)
    print('test raw  : ', timgs.shape)

    outh5 = h5py.File(args.outpath+'.hdf5', 'w')
    outh5.create_dataset('train_data_gen', data=imgs)
    outh5.create_dataset('train_data_raw', data=gimgs)
    outh5.create_dataset('val_data_gen', data=vimgs)
    outh5.create_dataset('val_data_raw', data=vgimgs)
    outh5.create_dataset('test_data_raw', data=timgs)
    outh5.flush()
    outh5.close()


if __name__ == '__main__':
    main()
