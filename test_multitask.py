import os
import h5py
import torch
import nibabel as nib
from networks.vnet_multitask import VNetMultiTask
from monai.inferers import SlidingWindowInferer
from skimage import morphology

from glob import glob
from tqdm import tqdm
import h5py
import nibabel as nib
import pandas as pd
import cv2
import pdb
from scipy.ndimage import binary_dilation, binary_erosion, generate_binary_structure, label

import os
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk

import numpy as np
from scipy import ndimage
import scipy
import argparse
from skimage.morphology import dilation, ball, erosion
from skimage.transform import resize
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg
 
# model_path
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="None")

args = parser.parse_args()

assert args.model_path != "None", "Please input the model path: --model_path xxx/xxx.pth"


import numpy as np
from scipy import ndimage

yinshe = {1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18, 9: 21, 10: 22, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28, 17: 31, 18: 32, 19: 33, 20: 34, 21: 35, 22: 36, 23: 37, 24: 38, 25: 41, 26: 42, 27: 43, 28: 44, 29: 45, 30: 46, 31: 47, 32: 48, 0:0}
import time



net = VNetMultiTask(
    n_channels=1, n_classes_1=2, n_classes_2=33, normalization="batchnorm",n_filters=16, SCANet=False
)


net = net.cuda()
path_weight = args.model_path
print(f'Using model: {path_weight}')
check_point = torch.load(path_weight, weights_only=True)
check_point = {k.replace("_orig_mod.", ""): v for k, v in check_point.items()}
net = net.cuda()
net.load_state_dict(check_point)
net.eval()

val_path = 'data/MICCAI2024/alls_data_no/val/'
save_path = 'data/MICCAI2024/TEST_final/'


# 定义采样_here

with torch.no_grad():
    sliding_window_inferer = SlidingWindowInferer(roi_size=(112, 112, 80), sw_batch_size=2, overlap=0.5)
    def net_fake(x):
        return net(x, ret_feats = False, drop=False, out_multiclass=True)[1]
    for path in tqdm(os.listdir(val_path)):
        img = val_path + path
        id = img[-12:-8]
        # print(id)
        
        h5f = h5py.File(img, "r")
        image = h5f["image"][:]
        image_roi = h5f["ROI_image"][:]
        shape = h5f['shape'][:]
        posi = h5f['ROI_posi'][:]

        image_roi = image_roi.reshape(1, 1, image_roi.shape[0], image_roi.shape[1], image_roi.shape[2]).astype(np.float32)
        now_device = torch.device("cuda")
        image_roi = torch.from_numpy(image_roi)
        image_roi = image_roi.to(device=now_device)
        
        outputs = torch.argmax(sliding_window_inferer(image_roi, net_fake).squeeze(0).softmax(dim=0), dim=0).cpu().numpy()
        
        for cls in np.unique(outputs)[::-1]:                      
            outputs[outputs == cls] = yinshe[cls]
        outputs = outputs.astype(np.uint8)
        
        #roi
        mask = np.zeros_like(image)
        x_min, x_max, y_min, y_max, z_min, z_max = posi
        mask[x_min:x_max+1,y_min:y_max+1,z_min:z_max+1] = outputs
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        mask = np.transpose(mask,axes=(2, 1, 0))
        sitk_prediction = sitk.GetImageFromArray(mask)
        # outputs =  ImageResample(mask, shape.tolist(), save_path + "/Validation_" + id + "_Mask.nii.gz")

        sitk.WriteImage(sitk_prediction,  save_path + "/Validation_" + id + "_Mask.nii.gz")