import numpy as np
import os,sys
import argparse
from tqdm import tqdm
from einops import rearrange, repeat

import cv2
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from ptflops import get_model_complexity_info

dir_name = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(dir_name,'../dataset/'))
sys.path.append(os.path.join(dir_name,'..'))

import scipy.io as sio
from dataset.dataset_denoise import *
import utils
import math
from model import UNet,Uformer

from skimage import img_as_float32, img_as_ubyte
from skimage.metrics import peak_signal_noise_ratio as psnr_loss
from skimage.metrics import structural_similarity as ssim_loss

parser = argparse.ArgumentParser(description='Image denoising evaluation on defect images')
parser.add_argument('input_path', type=str, help='Path to defects images directory or video file')
parser.add_argument('weights', type=str, help='Path to weights')
parser.add_argument('--result_dir', default='./denoised', type=str, help='Directory for results')
parser.add_argument('--gpus', default='0', type=str, help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--arch', default='Uformer_B', type=str, help='arch')
parser.add_argument('--batch_size', default=1, type=int, help='Batch size for dataloader')
parser.add_argument('--split', action='store_true', help='Split the image into two parts for inference (helpful for out-of-memory cases)')
parser.add_argument('--diff_thrd', type=int, default=0, help='Different threshold for post-processing')
parser.add_argument('--preserve_large', action='store_true', help='Preserve large noisy regoin as original')
parser.add_argument('--extension', type=str, default="png", help='Output image file extension format')
parser.add_argument('--save_original', action='store_true', help='Save original (noisy) images in result directory')
parser.add_argument('--save_mask', action='store_true', help='Save mask images in result directory')
parser.add_argument('--embed_dim', type=int, default=32, help='number of data loading workers')
parser.add_argument('--win_size', type=int, default=8, help='number of data loading workers')
parser.add_argument('--token_projection', type=str,default='linear', help='linear/conv token projection')
parser.add_argument('--token_mlp', type=str,default='leff', help='ffn/leff token mlp')
parser.add_argument('--dd_in', type=int, default=3, help='dd_in')

# args for vit
parser.add_argument('--vit_dim', type=int, default=256, help='vit hidden_dim')
parser.add_argument('--vit_depth', type=int, default=12, help='vit depth')
parser.add_argument('--vit_nheads', type=int, default=8, help='vit hidden_dim')
parser.add_argument('--vit_mlp_dim', type=int, default=512, help='vit mlp_dim')
parser.add_argument('--vit_patch_size', type=int, default=16, help='vit patch_size')
parser.add_argument('--global_skip', action='store_true', default=False, help='global skip connection')
parser.add_argument('--local_skip', action='store_true', default=False, help='local skip connection')
parser.add_argument('--vit_share', action='store_true', default=False, help='share vit module')

parser.add_argument('--train_ps', type=int, default=128, help='patch size of training sample')

args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus

restored_dir = os.path.join(args.result_dir, "denoised")
utils.mkdir(restored_dir)

if args.save_original:
    original_dir = os.path.join(args.result_dir, "noisy")
    utils.mkdir(original_dir)

if args.save_mask:
    mask_dir = os.path.join(args.result_dir, "mask")
    utils.mkdir(mask_dir)

model_restoration= utils.get_arch(args)

utils.load_checkpoint(model_restoration,args.weights)

model_restoration.cuda()
model_restoration.eval()

def expand2square(timg,factor=16.0):
    _, _, h, w = timg.size()

    X = int(math.ceil(max(h,w)/float(factor))*factor)

    img = torch.zeros(1,3,X,X).type_as(timg) # 3, h,w
    mask = torch.zeros(1,1,X,X).type_as(timg)

    # print(img.size(),mask.size())
    # print((X - h)//2, (X - h)//2+h, (X - w)//2, (X - w)//2+w)
    img[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)] = timg
    mask[:,:, ((X - h)//2):((X - h)//2 + h),((X - w)//2):((X - w)//2 + w)].fill_(1)

    return img, mask

def split_image(image):
    patches = []
    half_images = torch.split(
        image,
        math.ceil(image.shape[2] / 2),
        dim=2,
    )

    for half_image in half_images:
        patches += torch.split(
            half_image,
            math.ceil(image.shape[3] / 2),
            dim=3,
        )

    return patches

def merge_patches(patches):
    return torch.concat([
        torch.concat(patches[:2], dim=3),
        torch.concat(patches[2:], dim=3),
    ], dim=2)

def restore_image(input_image):
    noisy_image = input_image.astype(np.float32)
    noisy_image /= 255.
    noisy_image = torch.from_numpy(noisy_image).unsqueeze(0).permute(0,3,1,2).cuda()

    _, _, h, w = noisy_image.shape
    noisy_image, mask = expand2square(noisy_image, factor=128)
    noisy_patches = \
        split_image(noisy_image) \
        if args.split \
        else [noisy_image]
    restored_patches = []
    for noisy_patch in noisy_patches:
        restored_patch = model_restoration(noisy_patch)
        restored_patches.append(restored_patch)
    restored_image = \
        merge_patches(restored_patches) \
        if args.split \
        else restored_patches[0]
    restored_image = torch.masked_select(restored_image,mask.bool()).reshape(1,3,h,w)
    restored_image = torch.clamp(restored_image,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0)
    restored_image = img_as_ubyte(restored_image)

    return restored_image

def apply_diff_thrd(noisy_image, restored_image, diff_thrd):
    image_diff = cv2.absdiff(noisy_image, restored_image)
    restored_image = np.where(
        image_diff > args.diff_thrd,
        restored_image,
        noisy_image,
    )

    return restored_image

def generate_mask(
    noisy_image,
    restored_image,
    binary_threshold=10,
    kernel_size=5,
    area_threshold=1000,
):
    image_diff = cv2.absdiff(noisy_image, restored_image)
    image_diff_gray = cv2.cvtColor(image_diff, cv2.COLOR_RGB2GRAY)
    image_diff_blur = cv2.blur(image_diff_gray, (kernel_size, kernel_size))
    _, image_diff_binary = cv2.threshold(
        image_diff_blur,
        binary_threshold,
        255,
        cv2.THRESH_BINARY,
    )
    contours, hierarchy = cv2.findContours(
        image_diff_binary,
        cv2.RETR_TREE,
        cv2.CHAIN_APPROX_SIMPLE,
    )[-2:]
    contours = [c for c in contours if cv2.contourArea(c) >= area_threshold]
    image_mask = cv2.drawContours(
        np.zeros_like(restored_image),
        contours,
        -1,
        (255, 255, 255),
        -1,
    )

    return image_mask

is_video = not os.path.isdir(args.input_path)
loader_class = utils.VideoLoader if is_video else utils.ImagesFolderLoader
imgs_loader = loader_class(args.input_path)

with torch.no_grad():
    for img_id, noisy_image in imgs_loader:
        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
        restored_image = restore_image(noisy_image)
        restored_image = apply_diff_thrd(
            noisy_image,
            restored_image,
            args.diff_thrd,
        )

        image_mask = generate_mask(noisy_image, restored_image)

        if args.preserve_large:
            mask = image_mask.astype(bool)
            restored_image[mask] = noisy_image[mask]

        img_name = \
            str(img_id + 1).zfill(8) \
            if is_video \
            else os.path.splitext(os.path.basename(img_id))[0]
        img_filename = "{}.{}".format(img_name, args.extension)

        restored_path = os.path.join(restored_dir, img_filename)
        utils.save_img(restored_path, restored_image)

        if args.save_original:
            original_path = os.path.join(original_dir, img_filename)
            utils.save_img(original_path, noisy_image)

        if args.save_mask:
            mask_path = os.path.join(mask_dir, img_filename)
            utils.save_img(mask_path, image_mask)

        print("Done: {}".format(img_id))
