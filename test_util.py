import h5py
import math
import nibabel as nib
import numpy as np
from medpy import metric
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os
import pandas as pd
from collections import OrderedDict
from monai.inferers import SlidingWindowInferer
from torchvision import transforms as T
from dataloaders.la_heart import ToTensor


def test(net, image_list, roi_size=(112, 112, 80), stage=2):
    sliding_window_inferer = SlidingWindowInferer(
    roi_size=roi_size, sw_batch_size=4, overlap=0.75)
    total_metric = 0.0
    now_device = torch.device("cuda")
    for img in image_list:
        h5f = h5py.File(img, "r")
        image = h5f["ROI_image"][:]
        label = h5f["ROI_label"][:]
        image = image.reshape(1, 1, image.shape[0], image.shape[1], image.shape[2]).astype(
            np.float32
        )
        image = torch.from_numpy(image)
        image = image.to(device=now_device)
        # print(image.shape)
        outputs = torch.argmax(sliding_window_inferer(image, net).squeeze(0).softmax(dim=0), dim=0).cpu().numpy()
        # print(outputs.shape)
        if np.sum(outputs) == 0:
            final = 0
        else:
            final = compute_metrics(outputs, label[:], stage)
        total_metric += final
    total_metric /= len(image_list)
    print("average metric is {}".format(total_metric))
    return total_metric


def test_all_case(
    net,
    image_list,
    num_classes,
    patch_size=(112, 112, 80),
    stride_xy=18,
    stride_z=4,
    save_result=True,
    test_save_path=None,
    preproc_fn=None,
):
    total_metric = 0.0

    
    for image_path in tqdm(image_list):
        case_name = image_path.split("/")[-2]
        id = image_path.split("/")[-1]
        
        h5f = h5py.File(image_path, "r")
        image = h5f["image"][:]
        label = h5f["label"][:]
        if preproc_fn is not None:
            image = preproc_fn(image)
        prediction, score_map = test_single_case(
            net, image, stride_xy, stride_z, patch_size, num_classes=num_classes
        )

        if np.sum(prediction) == 0:
            final = 0
        else:
            final = compute_metrics(prediction, label[:])

        total_metric += final

        if save_result:
            test_save_path_temp = os.path.join(test_save_path, case_name)
            if not os.path.exists(test_save_path_temp):
                os.makedirs(test_save_path_temp)
            nib.save(
                nib.Nifti1Image(prediction.astype(np.float32), np.eye(4)),
                test_save_path_temp + "/" + id + "_pred.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(image[:].astype(np.float32), np.eye(4)),
                test_save_path_temp + "/" + id + "_img.nii.gz",
            )
            nib.save(
                nib.Nifti1Image(label[:].astype(np.float32), np.eye(4)),
                test_save_path_temp + "/" + id + "_gt.nii.gz",
            )
    avg_metric = total_metric / len(image_list)
    if save_result:
        metric_csv = pd.DataFrame(metric_dict)
        metric_csv.to_csv(test_save_path + "/metric.csv", index=False)
    print("average metric is {}".format(avg_metric))

    return avg_metric


def test_single_case(net, image, stride_xy, stride_z, patch_size, num_classes=1):
    w, h, d = image.shape

    # if the size of image is less than patch_size, then padding it
    add_pad = False
    if w < patch_size[0]:
        w_pad = patch_size[0] - w
        add_pad = True
    else:
        w_pad = 0
    if h < patch_size[1]:
        h_pad = patch_size[1] - h
        add_pad = True
    else:
        h_pad = 0
    if d < patch_size[2]:
        d_pad = patch_size[2] - d
        add_pad = True
    else:
        d_pad = 0
    wl_pad, wr_pad = w_pad // 2, w_pad - w_pad // 2
    hl_pad, hr_pad = h_pad // 2, h_pad - h_pad // 2
    dl_pad, dr_pad = d_pad // 2, d_pad - d_pad // 2
    if add_pad:
        image = np.pad(
            image,
            [(wl_pad, wr_pad), (hl_pad, hr_pad), (dl_pad, dr_pad)],
            mode="constant",
            constant_values=0,
        )
        
    ww, hh, dd = image.shape

    sx = math.ceil((ww - patch_size[0]) / stride_xy) + 1
    sy = math.ceil((hh - patch_size[1]) / stride_xy) + 1
    sz = math.ceil((dd - patch_size[2]) / stride_z) + 1
    # print("{}, {}, {}".format(sx, sy, sz))
    score_map = np.zeros((num_classes,) + image.shape).astype(np.float32)
    cnt = np.zeros(image.shape).astype(np.float32)

    for x in range(0, sx):
        xs = min(stride_xy * x, ww - patch_size[0])
        for y in range(0, sy):
            ys = min(stride_xy * y, hh - patch_size[1])
            for z in range(0, sz):
                zs = min(stride_z * z, dd - patch_size[2])
                test_patch = image[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ]
                test_patch = np.expand_dims(
                    np.expand_dims(test_patch, axis=0), axis=0
                ).astype(np.float32)
                print(test_patch.shape)
                test_patch = torch.from_numpy(test_patch).cuda()
                y1 = net(test_patch) 
                y = F.softmax(y1, dim=1)
                y = y.cpu().data.numpy()
                y = y[0, :, :, :, :]
                score_map[
                    :,
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ] = (
                    score_map[
                        :,
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ]
                    + y
                )
                cnt[
                    xs : xs + patch_size[0],
                    ys : ys + patch_size[1],
                    zs : zs + patch_size[2],
                ] = (
                    cnt[
                        xs : xs + patch_size[0],
                        ys : ys + patch_size[1],
                        zs : zs + patch_size[2],
                    ]
                    + 1
                )
    score_map = score_map / np.expand_dims(cnt, axis=0)
    label_map = np.argmax(score_map, axis=0)
    if add_pad:
        label_map = label_map[
            wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d
        ]
        score_map = score_map[
            :, wl_pad : wl_pad + w, hl_pad : hl_pad + h, dl_pad : dl_pad + d
        ]
    return label_map, score_map


def cal_dice(prediction, label, num=2):
    total_dice = np.zeros(num - 1)
    for i in range(1, num):
        prediction_tmp = prediction == i
        label_tmp = label == i
        prediction_tmp = prediction_tmp.astype(np.float)
        label_tmp = label_tmp.astype(np.float)

        dice = (
            2
            * np.sum(prediction_tmp * label_tmp)
            / (np.sum(prediction_tmp) + np.sum(label_tmp))
        )
        total_dice[i - 1] += dice

    return total_dice


import numpy as np

def cacl_iou(y, y_pred):
    y = y.flatten()
    y_pred = y_pred.flatten()
    
    intersection = np.sum(y * y_pred)
    union = np.sum(y) + np.sum(y_pred) - intersection
    
    return (intersection + 1e-7) / (union + 1e-7)



#############################################for MICCAI 2024 Test########################################################

import os
import argparse
import numpy as np
import pandas as pd
from tqdm import tqdm
import SimpleITK as sitk
import multiprocessing as mp
from collections import OrderedDict
from SurfaceDice import (compute_surface_distances, compute_surface_dice_at_tolerance, compute_dice_coefficient,
                         compute_iou_score)


def compute_multi_class_iou(gt, seg):
    iou = []
    for i in np.unique(gt):
        if i == 0:
            continue
        gt_i = gt == i
        seg_i = seg == i
        iou.append(compute_iou_score(gt_i, seg_i))
    return np.mean(iou)


def compute_multi_class_dsc(gt, seg):
    dsc = []
    for i in np.unique(gt):
        if i == 0:
            continue
        gt_i = gt == i
        seg_i = seg == i
        dsc.append(compute_dice_coefficient(gt_i, seg_i))
    return np.mean(dsc)


def compute_multi_class_nsd(gt, seg, spacing, tolerance=2.0):
    nsd = []
    for i in np.unique(gt):
        if i == 0:
            continue
        gt_i = gt == i
        seg_i = seg == i
        surface_distance = compute_surface_distances(gt_i, seg_i, spacing_mm=spacing)
        nsd.append(compute_surface_dice_at_tolerance(surface_distance, tolerance))
    return np.mean(nsd)



def compute_metrics(seg, gt, stage):
    if stage == 1:
        image_dsc = compute_dice_coefficient(gt != 0, seg != 0)
        image_iou = compute_iou_score(gt != 0, seg != 0)
        metic = (image_dsc + image_iou) / 2.0
        print("\n***** Evaluation ***** >>>> image_dsc: {:.4f}, image_iou: {:.4f} \n".format( image_dsc , image_iou ))
        return metic
        
    # seg:array, gt:array for 3 channel(h, w, d)

    # image-level metrics
    image_dsc = compute_dice_coefficient(gt != 0, seg != 0)
    image_iou = compute_iou_score(gt != 0, seg != 0)
    surface_distance = compute_surface_distances(gt != 0, seg != 0, spacing_mm=[0.25, 0.25, 0.25])
    image_nsd = compute_surface_dice_at_tolerance(surface_distance, tolerance_mm=2)

    # instance-level metrics
    instance_dsc = compute_multi_class_dsc(gt, seg)
    instance_nsd = compute_multi_class_nsd(gt, seg, spacing=[1, 1, 1])
    # TP means iou > 0.5 and class is equal
    TP = 0
    iou_list = []
    for i in np.unique(gt):
        if i == 0:
            continue
        iou = compute_iou_score(gt == i, seg == i)
        iou_list.append(iou)
        if iou > 0.5:
            TP += 1
    # instance-level iou
    instance_iou = np.mean(iou_list)
    # instance-level ia, -1 means the background 0. IA is a bit like class-level IoU
    ia = TP / (len(list(set(np.unique(gt)).union(set(np.unique(seg))))) - 1)
    metic = (image_dsc + image_iou + image_nsd + instance_dsc + instance_iou + instance_nsd + ia) / 7.0
    print(
        "\n***** Evaluation ***** >>>> image_dsc: {:.4f}, image_iou: {:.4f} , image_nsd: {:.4f}, instance_dsc: {:.4f}, instance_iou: {:.4f},instance_nsd: {:.4f},ia: {:.4f} \n".format(
            image_dsc , image_iou , image_nsd , instance_dsc ,  instance_iou ,instance_nsd , ia
        )
    )
    return  metic
