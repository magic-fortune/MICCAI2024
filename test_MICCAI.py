import os
import h5py
import torch
import nibabel as nib
from networks.vnet import VNet
from monai.inferers import SlidingWindowInferer

from glob import glob
from tqdm import tqdm
import h5py
import nibabel as nib
import pandas as pd
import pdb

import os
import pydicom
import matplotlib.pyplot as plt
import SimpleITK as sitk

import numpy as np
from scipy import ndimage



def ImageResample(numpy_image, new_size, save_path):

    sitk_image = sitk.GetImageFromArray(numpy_image)  

    # resample = sitk.ResampleImageFilter()
    # resample.SetOutputDirection(sitk_image.GetDirection())
    # resample.SetOutputOrigin(sitk_image.GetOrigin())
    # resample.SetSize(new_size)
    # print("infromation is as follows:***********************")
    # print(sitk_image.GetDirection())
    # print(sitk_image.GetOrigin())
    # print(new_size)
    # print("************************************************************")
    
    # resample.SetInterpolator(sitk.sitkNearestNeighbor)
    # newimage = resample.Execute(sitk_image)

    sitk_image = sitk.GetArrayFromImage(sitk_image)
    # print(newimage.shape)
    sitk_image = np.transpose(sitk_image, axes=(2, 1, 0)) 
    resize_factor = (new_size[0]/ sitk_image.shape[0], new_size[1] / sitk_image.shape[1], new_size[2] / sitk_image.shape[2])
    sitk_image = ndimage.zoom(sitk_image, zoom=resize_factor,order=0)
    
    sitk_image = sitk_image.astype(np.int16)
    out = sitk.GetImageFromArray(sitk_image)
    sitk.WriteImage(out, save_path)
    return sitk_image

net = VNet(
    n_channels=1, n_classes=33, normalization="batchnorm", 
)
net = net.cuda()
path_weight = "model/lab30/AdamW/15k/e0.5/t0.75/no_cutmix/s1_to_s2/best_model.pth"
check_point = torch.load(path_weight, weights_only=True)
net.load_state_dict(check_point)
net.eval()
yinshe = {1: 11, 2: 12, 3: 13, 4: 14, 5: 15, 6: 16, 7: 17, 8: 18, 9: 21, 10: 22, 11: 23, 12: 24, 13: 25, 14: 26, 15: 27, 16: 28, 17: 31, 18: 32, 19: 33, 20: 34, 21: 35, 22: 36, 23: 37, 24: 38, 25: 41, 26: 42, 27: 43, 28: 44, 29: 45, 30: 46, 31: 47, 32: 48, 0:0}
val_path = 'data/MICCAI2024/alls_data_(-500-2000)/val/'
save_path = 'data/MICCAI2024/TEST/'


# 定义采样_here

with torch.no_grad():
    sliding_window_inferer = SlidingWindowInferer(roi_size=(112, 112, 80), sw_batch_size=4, overlap=0.5)
    for path in os.listdir(val_path):
        img = val_path + path
        id = img[-12:-8]
        # print(id)
        
        h5f = h5py.File(img, "r")
        image = h5f["image"][:]
        spacing = h5f['spacing'][()]
        shape = h5f['shape'][:]
        # print(shape)
        # print(shape.tolist())
     
        # print(image.shape)
        image = image.reshape(1, 1, image.shape[0], image.shape[1], image.shape[2]).astype(np.float32)
        now_device = torch.device("cuda")
        image = torch.from_numpy(image)
        image = image.to(device=now_device)
        
        outputs = torch.argmax(sliding_window_inferer(image, net).squeeze(0).softmax(dim=0), dim=0).cpu().numpy()
        # print(outputs.shape)
        
        for cls in np.unique(outputs)[::-1]:                      
            outputs[outputs == cls] = yinshe[cls]
        # save
        outputs = outputs.astype(np.uint8)
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        # outputs = np.transpose(outputs, (2, 1, 0))
        # print(outputs.shape)
        outputs =  ImageResample(outputs, shape.tolist(), save_path + "/Validation_" + id + "_Mask.nii.gz")
        # break