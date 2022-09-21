import numpy as np
import os
import math
import pickle
from torch.utils.data import Dataset
import numpy as np
import torch
from utils import is_png_file, load_img, Augment_RGB_torch
import torch.nn.functional as F
import random
import cv2
from PIL import Image
import torchvision.transforms.functional as TF
from natsort import natsorted
from glob import glob
augment   = Augment_RGB_torch()
transforms_aug = [method for method in dir(augment) if callable(getattr(augment, method)) if not method.startswith('_')] 


def load_pickle(file_path):
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])
    
##################################################################################################
class DefectsDataset(Dataset):

    def __init__(
        self,
        rgb_dir,
        tar_size,
        weight_min = 1,
        weight_max = 20000,
        img_options=None,
        target_transform=None,
    ):
        super(DefectsDataset, self).__init__()

        self._data_dir = rgb_dir
        self._weight_min = weight_min
        self._weight_max = weight_max
        self.tar_size = tar_size
        self.target_transform = target_transform
        self.img_options=img_options

        self._dir_noisy = os.path.join(self._data_dir, 'before')
        self._dir_clean = os.path.join(self._data_dir, 'after')
        self._dir_diff = os.path.join(self._data_dir, 'diff')

        # Load mapping for metadata.
        self._metadata_mapping = load_pickle(
            os.path.join(self._data_dir, 'metadata.pickle'),
        )

        # Initilization for sampling.
        self._renew()

    def _redistribute_weights(self, diff_values):
        return np.clip(
            math.e ** np.array(diff_values),
            self._weight_min,
            self._weight_max,
        )

    def _choose_seq(self):
        # Get sequence filenames.
        filenames = list(self._metadata_mapping.keys())
        # Get average sequence diff values.
        seq_diff_values = [m[0] for m in self._metadata_mapping.values()]
        # Define the weights for sequence sampling.
        seq_weights = self._redistribute_weights(seq_diff_values)

        # Choose a filename by weighted random sampling.
        filename = random.choices(filenames, weights=seq_weights, k=1)[0]

        # Load files for noisy and clean patches (images), and their diff
        # values.
        patches_noisy = np.load(os.path.join(self._dir_noisy, filename))
        patches_clean = np.load(os.path.join(self._dir_clean, filename))
        diff_values = np.load(os.path.join(self._dir_diff, filename))

        # Define the weights for patches sampling.
        weights = self._redistribute_weights(diff_values)

        return patches_noisy, patches_clean, weights

    def _sample(self):
        # Weighted random sampling the patches indexes.
        idxes = random.choices(
            range(self._patches_noisy.shape[0]),
            weights=self._weights,
            k=1,
        )

        # Get the noisy and clean patches data by the indexes.
        data_noisy = self._patches_noisy[idxes]
        data_clean = self._patches_clean[idxes]

        # Re-initilize if needed.
        self._num_repeats += 1
        if self._num_repeats >= self._max_repeats:
            self._renew()

        return data_noisy, data_clean

    def _renew(self):
        # Choose a sequence that contains noisy and clean patches (images) and
        # their diff values.
        self._patches_noisy, self._patches_clean, self._weights = \
            self._choose_seq()

        # Define the upper bound of repeats for the sequence.
        self._max_repeats = self._patches_noisy.shape[0]
        self._num_repeats = 0

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        noisy, clean = self._sample()
        noisy = torch.from_numpy(
            cv2.cvtColor(noisy[0], cv2.COLOR_BGR2RGB).astype(np.float32),
        )
        clean = torch.from_numpy(
            cv2.cvtColor(clean[0], cv2.COLOR_BGR2RGB).astype(np.float32),
        )
        noisy = noisy / 255.
        clean = clean / 255.

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        clean_filename = ""
        noisy_filename = ""

        #Crop Input and Target
        ps = self.img_options['patch_size']
        H = clean.shape[1]
        W = clean.shape[2]
        # r = np.random.randint(0, H - ps) if not H-ps else 0
        # c = np.random.randint(0, W - ps) if not H-ps else 0
        if H-ps==0:
            r=0
            c=0
        else:
            r = np.random.randint(0, H - ps)
            c = np.random.randint(0, W - ps)
        clean = clean[:, r:r + ps, c:c + ps]
        noisy = noisy[:, r:r + ps, c:c + ps]

        apply_trans = transforms_aug[random.getrandbits(3)]

        clean = getattr(augment, apply_trans)(clean)
        noisy = getattr(augment, apply_trans)(noisy)

        return clean, noisy, clean_filename, noisy_filename


##################################################################################################
class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, target_transform=None):
        super(DataLoaderVal, self).__init__()

        self.target_transform = target_transform

        gt_dir = 'groundtruth'
        input_dir = 'input'
        
        clean_files = sorted(os.listdir(os.path.join(rgb_dir, gt_dir)))
        noisy_files = sorted(os.listdir(os.path.join(rgb_dir, input_dir)))


        self.clean_filenames = [os.path.join(rgb_dir, gt_dir, x) for x in clean_files if is_png_file(x)]
        self.noisy_filenames = [os.path.join(rgb_dir, input_dir, x) for x in noisy_files if is_png_file(x)]
        

        self.tar_size = len(self.clean_filenames)  

    def __len__(self):
        return self.tar_size

    def __getitem__(self, index):
        tar_index   = index % self.tar_size
        

        clean = torch.from_numpy(np.float32(load_img(self.clean_filenames[tar_index])))
        noisy = torch.from_numpy(np.float32(load_img(self.noisy_filenames[tar_index])))
                
        clean_filename = os.path.split(self.clean_filenames[tar_index])[-1]
        noisy_filename = os.path.split(self.noisy_filenames[tar_index])[-1]

        clean = clean.permute(2,0,1)
        noisy = noisy.permute(2,0,1)

        return clean, noisy, clean_filename, noisy_filename

##################################################################################################

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(inp_dir))
        self.inp_filenames = [os.path.join(inp_dir, x) for x in inp_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = self.inp_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)

        inp = TF.to_tensor(inp)
        return inp, filename


def get_training_data(rgb_dir, img_options):
    assert os.path.exists(rgb_dir)
    return DataLoaderTrain(rgb_dir, img_options, None)


def get_validation_data(rgb_dir):
    assert os.path.exists(rgb_dir)
    return DataLoaderVal(rgb_dir, None)

def get_test_data(rgb_dir, img_options=None):
    assert os.path.exists(rgb_dir)
    return DataLoaderTest(rgb_dir, img_options)
