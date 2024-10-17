import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np
from networks.vnet import VNet
import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid
from test_util import test_all_case
from utils.losses_wcs import DiceLoss
from dataloaders.la_heart import (
    LAHeart,
    RandomCrop,
    RandomRotFlip,
    ToTensor,
    Tooth_WCS
)
from utils import ramps
import time
import h5py
from test_util import test
from torch import nn


parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="../data/2018LA_Seg_Training Set/",
    help="Name of Experiment",
)
parser.add_argument("--exp", type=str, default="distill_match", help="model_name")
parser.add_argument(
    "--max_iterations", type=int, default=12000, help="maximum epoch number to train"
)
parser.add_argument("--batch_size", type=int, default=2, help="batch_size per gpu")
parser.add_argument(
    "--base_lr", type=float, default=0.01, help="maximum epoch number to train"
)
parser.add_argument(
    "--deterministic", type=int, default=1, help="whether use deterministic training"
)
parser.add_argument("--seed", type=int, default=1337, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--label_num", type=int, default=30, help="label num")
parser.add_argument(
    "--eta", type=float, default=0.3, help="weight to balance loss"
)
parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer")
parser.add_argument("--conf_thresh", type=float, default=0.85, help="conf_thresh")
parser.add_argument('--pert_gap', type=float, default=0.5, help='the perturbation gap')
parser.add_argument('--pert_type', type=str, default='dropout', help='feature pertubation types')

parser.add_argument('--dataset_name', type=str, default='Tooth', help='dataset_name')
parser.add_argument('--s1_to_s2', action='store_true', help='s1 supervise s2')
# corr_match_type
parser.add_argument('--corr_match_type', type=str, default='kl', help='correlation match type')
# temperature
parser.add_argument('--temperature', type=float, default=1.5, help='temperature')
parser.add_argument('--consistency', type=float,
                    default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,
                    default=200.0, help='consistency_rampup')

args = parser.parse_args()

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

def kaiming_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model

def xavier_normal_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            torch.nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


def create_model(ema=False):
    # Network definition
    model =  VNet(
        n_channels=1, n_classes=33, normalization="batchnorm", 
    )
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

model1 = create_model()
model2 = create_model()
model1.cuda()
model2.cuda()

patch_size = (112, 112, 80)
num_classes = 33

dice_loss = DiceLoss(n_classes=num_classes)


if args.dataset_name == "Tooth":
    patch_size = (112, 112, 80)
    args.root_path = '../data/MICCAI2024/alls_data_no/'
    args.max_samples = 330
    DATASET_CLASS = Tooth_WCS
    TSFM = transforms.Compose(
        [
            RandomCrop(patch_size),
            ToTensor(),
        ]
    )

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(","))
max_iterations = args.max_iterations
base_lr = args.base_lr

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
else:
    cudnn.benchmark = True
    cudnn.deterministic = False
    cudnn.enabled = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

LABELED_ID_NUM = args.label_num  # 8 or 16
conf_thresh = args.conf_thresh
eta = args.eta
pert_gap = args.pert_gap

pervious_bset_dice1 = 0.0
pervious_bset_dice2 = 0.0


if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + "/code"):
        shutil.rmtree(snapshot_path + "/code")

    shutil.copytree(
        ".", snapshot_path + "/code", shutil.ignore_patterns([".git", "__pycache__"])
    )

    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainset_u = DATASET_CLASS(
        base_dir=train_data_path,
        mode="train_u",
        num=args.max_samples - LABELED_ID_NUM,
        transform=TSFM,
        id_path=f"{args.root_path}/train_{LABELED_ID_NUM}_unlabel.list",
    )
    trainsampler_u = torch.utils.data.sampler.RandomSampler(trainset_u)
    trainloader_u = DataLoader(
        trainset_u,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size * 2,
        drop_last=True,
        sampler=trainsampler_u,
        worker_init_fn=worker_init_fn
    )
    trainsampler_u_mix = torch.utils.data.sampler.RandomSampler(trainset_u, replacement=True)
    trainloader_u_mix = DataLoader(
        trainset_u,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size * 2,
        drop_last=True,
        sampler=trainsampler_u_mix,
        worker_init_fn=worker_init_fn
    )

    trainset_l = DATASET_CLASS(
        base_dir=train_data_path,
        mode="train_l",
        num=args.max_samples - LABELED_ID_NUM,
        transform=TSFM,
        id_path=f"{args.root_path}/train_{LABELED_ID_NUM}_label.list",
    )
    trainsampler_l = torch.utils.data.sampler.RandomSampler(trainset_l)
    trainloader_l = DataLoader(
        trainset_l,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size * 2,
        drop_last=True,
        sampler=trainsampler_l,
        worker_init_fn=worker_init_fn
    )

    model1.train()
    model2.train()
    if args.optimizer == "SGD":
        optimizer1 = optim.SGD(
            model1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
        )
        optimizer2 = optim.SGD(
            model2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
        )
    elif args.optimizer == "Adam":
        optimizer1 = optim.Adam(
            model1.parameters(), lr=base_lr, weight_decay=0.0001
        )
        optimizer2 = optim.Adam(
            model2.parameters(), lr=base_lr, weight_decay=0.0001
        )
    elif args.optimizer == "AdamW":
        optimizer1 = optim.AdamW(
            model1.parameters(), lr=base_lr, weight_decay=0.0001
        )
        optimizer2 = optim.AdamW(
            model2.parameters(), lr=base_lr, weight_decay=0.0001
        )
    else:
        raise NotImplementedError
    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} itertations per epoch".format(len(trainloader_l)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader_l) + 1
    print(f"All Epochs: {max_epoch}")
    lr_ = base_lr

    
    for epoch_num in tqdm(range(max_epoch), ncols=70):

        time1 = time.time()
        model1.train()
        model2.train()

        for i_batch, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, img_u_s2, _, _)
        ) in enumerate(zip(trainloader_l, trainloader_u)):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()            
            bef_time = time.time()

            pred_x_pred_u_w, pred_x_b_pred_u_w_b = model1(
                torch.cat((img_x, img_u_w)),ret_feats=True, drop=False
            )
            pred_x_pred_u_w2, pred_x_b_pred_u_w_b2 = model2(
                torch.cat((img_x, img_u_w)),ret_feats=True, drop=False
            )
            
            pred_x, pred_u_w = pred_x_pred_u_w.chunk(2)
            # bottleneck features
            pred_x_bf, pred_u_w_bf = pred_x_b_pred_u_w_b.chunk(2)
            
            pred_x2, pred_u_w2 = pred_x_pred_u_w2.chunk(2)
            # bottleneck features
            pred_x_bf2, pred_u_w_bf2 = pred_x_b_pred_u_w_b2.chunk(2)
            
            
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)
            
            pred_u_w2 = pred_u_w2.detach()
            conf_u_w2 = pred_u_w2.softmax(dim=1).max(dim=1)[0]
            mask_u_w2 = pred_u_w2.argmax(dim=1)
            
            pred_x_soft = torch.softmax(pred_x, dim=1)
            pred_x2_soft = torch.softmax(pred_x2, dim=1)
            pseudo_outputs2 = torch.argmax(pred_x_soft, dim=1, keepdim=False)
            pseudo_outputs1 = torch.argmax(pred_x2_soft, dim=1, keepdim=False)
            
            loss_x1 = (
                F.cross_entropy(pred_x, mask_x)
                + dice_loss(pred_x, mask_x)
            ) / 2.0
            
            loss_x2 = (
                F.cross_entropy(pred_x2, mask_x)
                + dice_loss(pred_x2, mask_x)
            ) / 2.0
            
            pseudo_supervision1 = F.cross_entropy(pred_x_soft, pseudo_outputs2)
            pseudo_supervision2 = F.cross_entropy(pred_x2_soft, pseudo_outputs1)
            
            consistency_weight = get_current_consistency_weight(iter_num // 150)
            model1_loss = loss_x1 + consistency_weight * pseudo_supervision1
            model2_loss = loss_x2 + consistency_weight * pseudo_supervision2
            
            loss = model1_loss + model2_loss
            
            conf_thresh = (
                args.conf_thresh + (1 - args.conf_thresh) * ramps.sigmoid_rampup(iter_num, max_iterations)
            ) * np.log(2)

            optimizer1.zero_grad()
            optimizer2.zero_grad()

            loss.backward()

            optimizer1.step()
            optimizer2.step()
        
            iter_num = iter_num + 1
            writer.add_scalar("lr", lr_, iter_num)
            writer.add_scalar("loss/model2_loss", model2_loss, iter_num)
            writer.add_scalar("loss/model1_loss", model1_loss, iter_num)
            writer.add_scalar("loss/loss", loss, iter_num)
            
            lr_ = base_lr * (1 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer1.param_groups:
                param_group["lr"] = lr_
            for param_group in optimizer2.param_groups:
                param_group["lr"] = lr_

            if iter_num % 50 == 0:
                image = (
                    img_x[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                )
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image("train/Image", grid_image, iter_num)

                outputs_soft = F.softmax(pred_x, 1)
                image = (
                    outputs_soft[0, 1:2, :, :, 20:61:10]
                    .permute(3, 0, 1, 2)
                    .repeat(1, 3, 1, 1)
                )
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image("train/Predicted_label", grid_image, iter_num)

                image = (
                    mask_x[0, :, :, 20:61:10]
                    .unsqueeze(0)
                    .permute(3, 0, 1, 2)
                    .repeat(1, 3, 1, 1)
                )
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image("train/Groundtruth_label", grid_image, iter_num)
                
            if iter_num % 1000 == 0:
                save_mode_path = os.path.join(
                    snapshot_path, "iter_" + str(iter_num) + ".pth"
                )
                save_mode_path1 = os.path.join(
                    snapshot_path, "model1_" + str(iter_num) + ".pth"
                )
                save_mode_path2 = os.path.join(
                    snapshot_path, "model2_" + str(iter_num) + ".pth"
                )
                torch.save(model1.state_dict(), save_mode_path1)
                torch.save(model2.state_dict(), save_mode_path2)

                logging.info("save model to {}".format(save_mode_path))

            if iter_num > max_iterations:
                print("finish training, iter_num > max_iterations")
                break
            time1 = time.time()
        if iter_num > max_iterations:
            print("finish training")
            break

        if (epoch_num + 1) % (4) == 0:
            from monai.inferers import SlidingWindowInferer
            import nibabel as nib
            # evals
            model1.eval()
            model2.eval()
            with torch.no_grad():
                with open(args.root_path + "./sample_test.list", "r") as f:
                    image_list = f.readlines()
                print(image_list)
                if args.dataset_name == "Tooth":
                    image_list = [
                        args.root_path + item.replace("\n", "") + ""
                        for item in image_list
                    ]
                    
                dice1 = test(model1, image_list)
                dice2 = test(model2, image_list)

                if dice1 > pervious_bset_dice1:
                    pervious_bset_dice1 = dice1
                    save_mode_path = os.path.join(snapshot_path, "best_model1.pth")
                    torch.save(model1.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))
                if dice2 > pervious_bset_dice2:
                    pervious_bset_dice2 = dice2
                    save_mode_path = os.path.join(snapshot_path, "best_model2.pth")
                    torch.save(model2.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

    save_mode_path = os.path.join(
        snapshot_path, "model1_iter_" + str(max_iterations + 1) + ".pth"
    )
    torch.save(model1.state_dict(), save_mode_path)
    save_mode_path = os.path.join(
        snapshot_path, "model2_iter_" + str(max_iterations + 1) + ".pth"
    )
    torch.save(model2.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
