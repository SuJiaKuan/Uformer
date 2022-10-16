import os
import pickle
from abc import ABC
from abc import abstractmethod
from glob import glob

import cv2
import numpy as np
import torch

IMAGE_EXTENSIONS = (
    "bmp",
    "png",
    "jpg",
    "jpeg",
    "tif",
)

def is_numpy_file(filename):
    return any(filename.endswith(extension) for extension in [".npy"])

def is_image_file(filename):
    return any(filename.endswith(extension) for extension in [".jpg"])

def is_png_file(filename):
    return any(filename.endswith(extension) for extension in [".png"])

def is_pkl_file(filename):
    return any(filename.endswith(extension) for extension in [".pkl"])

def load_pkl(filename_):
    with open(filename_, 'rb') as f:
        ret_dict = pickle.load(f)
    return ret_dict    

def save_dict(dict_, filename_):
    with open(filename_, 'wb') as f:
        pickle.dump(dict_, f)    

def load_npy(filepath):
    img = np.load(filepath)
    return img

def load_img(filepath):
    img = cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32)
    img = img/255.
    return img

def save_img(filepath, img):
    cv2.imwrite(filepath,cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

def myPSNR(tar_img, prd_img):
    imdff = torch.clamp(prd_img,0,1) - torch.clamp(tar_img,0,1)
    rmse = (imdff**2).mean().sqrt()
    ps = 20*torch.log10(1/rmse)
    return ps

def batch_PSNR(img1, img2, average=True):
    PSNR = []
    for im1, im2 in zip(img1, img2):
        psnr = myPSNR(im1, im2)
        PSNR.append(psnr)
    return sum(PSNR)/len(PSNR) if average else sum(PSNR)

def get_image_paths(imgs_dir, extensions=IMAGE_EXTENSIONS):
    img_paths = []
    for extension in extensions:
        img_paths.extend(glob(
            os.path.join(imgs_dir, "*.{}".format(extension)),
        ))

    return sorted(img_paths)

class ImagesLoader(object):

    def __len__(self):
        return self._get_len()

    def __iter__(self):
        for img_idx in range(len(self)):
            yield self._read(img_idx)

    def __getitem__(self, key):
        assert type(key) == int

        return self._read(img_idx)

    @abstractmethod
    def _get_len(self):
        pass

    @abstractmethod
    def _read(self, img_idx):
        pass

class ImagesFolderLoader(ImagesLoader):

    def __init__(self, imgs_dir):
        self._imgs_dir = imgs_dir

        self._img_paths = get_image_paths(self._imgs_dir)

    def _get_len(self):
        return len(self._img_paths)

    def _read(self, img_idx):
        img_path = self._img_paths[img_idx]
        img = cv2.imread(img_path)

        return img_path, img
