from __future__ import division
from __future__ import absolute_import
import os
import random
import fnmatch
import numpy as np
from PIL import Image

class DataLoader():
    def __init__(self, data_dir, dataset_name, img_res=(256, 256), test_only=False):
        self.img_res = img_res
        self.DATA = dataset_name #Unpaired
        self.data_dir = data_dir  # ../data/test/A/Unpaired

        if not test_only:
            print("Kyu nahi aati hhon tum")
            self.trainA_paths = getPaths(os.path.join(self.data_dir, "trainA"))  # distorted
            # print(self.trainA_paths)
            self.trainB_paths = getPaths(os.path.join(self.data_dir, "trainB"))# enhanced
            # print(self.trainA_paths)
            print("Length of trainA folder:",len(self.trainA_paths))
            print("Length of trainB folder:",len(self.trainB_paths))
            if len(self.trainA_paths) < len(self.trainB_paths):
                self.trainB_paths = self.trainB_paths[:len(self.trainA_paths)]
            elif len(self.trainA_paths) > len(self.trainB_paths):
                self.trainA_paths = self.trainA_paths[:len(self.trainB_paths)]
            else:
                pass
            self.val_paths = getPaths(os.path.join(self.data_dir, "validation"))
            print(self.val_paths)
            print("Length of validation folder:",len(self.val_paths))
            self.num_train, self.num_val = len(self.trainA_paths), len(self.val_paths)
            print("{0} training pairs\n".format(self.num_train))
        else:
            self.test_paths = getPaths(os.path.join(self.data_dir, "test"))
            print("{0} test images\n".format(len(self.test_paths)))

    def get_test_data(self, batch_size=1):
        idx = np.random.choice(np.arange(len(self.test_paths)), batch_size, replace=False)
        paths = self.test_paths[idx]
        imgs = []
        for p in paths:
            img = read_and_resize(p, self.img_res)
            imgs.append(img)
        imgs = preprocess(np.array(imgs))
        return imgs

    def load_val_data(self, batch_size=1):
        # if batch_size > self.num_val:
        #     batch_size = self.num_val  # Set batch_size to the number of available samples

        idx = np.random.choice(np.arange(self.num_val), batch_size, replace=False)
        pathsA = self.trainA_paths[idx]
        pathsB = self.trainB_paths[idx]
        imgs_A, imgs_B = [], []
        for idx in range(len(pathsB)):
            img_A, img_B = read_and_resize_pair(pathsA[idx], pathsB[idx], self.img_res)
            imgs_A.append(img_A)
            imgs_B.append(img_B)
        imgs_A = preprocess(np.array(imgs_A))
        imgs_B = preprocess(np.array(imgs_B))
        return imgs_A, imgs_B

    def load_batch(self, batch_size=1, data_augment=True):
        self.n_batches = self.num_train // batch_size
        for i in range(self.n_batches - 1):
            batch_A = self.trainA_paths[i * batch_size:(i + 1) * batch_size]
            batch_B = self.trainB_paths[i * batch_size:(i + 1) * batch_size]
            imgs_A, imgs_B = [], []
            for idx in range(len(batch_A)):
                img_A, img_B = read_and_resize_pair(batch_A[idx], batch_B[idx], self.img_res)
                if data_augment:
                    img_A, img_B = augment(img_A, img_B)
                imgs_A.append(img_A)
                imgs_B.append(img_B)
            imgs_A = preprocess(np.array(imgs_A))
            imgs_B = preprocess(np.array(imgs_B))
            yield imgs_A, imgs_B

def deprocess(x, np_uint8=True):
    # [-1,1] -> [0, 255]
    x = (x + 1.0) * 127.5
    return np.uint8(x) if np_uint8 else x

def preprocess(x):
    # [0,255] -> [-1, 1]
    return (x / 127.5) - 1.0

def augment(a_img, b_img):
    a = random.random()  # randomly interpolate
    a_img = a_img * (1 - a) + b_img * a
    if random.random() < 0.25:
        a_img = np.fliplr(a_img)
        b_img = np.fliplr(b_img)
    if random.random() < 0.25:
        a_img = np.flipud(a_img)
        b_img = np.flipud(b_img)
    return a_img, b_img

def getPaths(data_dir):
    exts = ['*.png', '*.PNG', '*.jpg', '*.JPG', '*.JPEG']
    image_paths = []
    for pattern in exts:
        for d, s, fList in os.walk(data_dir):
            for filename in fList:
                if fnmatch.fnmatch(filename, pattern):
                    fname_ = os.path.join(d, filename)
                    image_paths.append(fname_)
    return np.asarray(image_paths)

def read_and_resize(path, img_res):
    im = Image.open(path).resize(img_res)
    if im.mode == 'L':
        copy = np.zeros((img_res[1], img_res[0], 3))
        copy[:, :, 0] = im
        copy[:, :, 1] = im
        copy[:, :, 2] = im
        im = copy
    return np.array(im).astype(np.float32)

def read_and_resize_pair(pathA, pathB, img_res):
    img_A = read_and_resize(pathA, img_res)
    img_B = read_and_resize(pathB, img_res)
    return img_A, img_B

def get_local_test_data(data_dir, img_res=(256, 256)):
    assert os.path.exists(data_dir), "local image path doesn't exist"
    imgs = []
    for p in getPaths(data_dir):
        img = read_and_resize(p, img_res)
        imgs.append(img)
    imgs = preprocess(np.array(imgs))
    return imgs
