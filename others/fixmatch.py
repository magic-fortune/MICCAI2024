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

args = parser.parse_args()

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

pervious_bset_dice = 0.0

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

    net = VNet(
        n_channels=1, n_classes=num_classes, normalization="batchnorm", 
    )
    net = net.cuda()
    
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

    net.train()
    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001
        )
    elif args.optimizer == "Adam":
        optimizer = optim.Adam(
            net.parameters(), lr=base_lr, weight_decay=0.0001
        )
    elif args.optimizer == "AdamW":
        optimizer = optim.AdamW(
            net.parameters(), lr=base_lr, weight_decay=0.0001
        )
    else:
        raise NotImplementedError
    writer = SummaryWriter(snapshot_path + "/log")
    logging.info("{} itertations per epoch".format(len(trainloader_l)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader_l) + 1
    print(f"All Epochs: {max_epoch}")
    lr_ = base_lr
    net.train()
    
    for epoch_num in tqdm(range(max_epoch), ncols=70):

        time1 = time.time()
        net.train()

        for i_batch, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, img_u_s2, _, _)
        ) in enumerate(zip(trainloader_l, trainloader_u)):
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            
            bef_time = time.time()

            pred_x_pred_u_w, pred_x_b_pred_u_w_b = net(
                torch.cat((img_x, img_u_w)),ret_feats=True, drop=False
            )
            pred_x, pred_u_w = pred_x_pred_u_w.chunk(2)
            # bottleneck features
            pred_x_bf, pred_u_w_bf = pred_x_b_pred_u_w_b.chunk(2)

            # pred_u_s1, pred_u_s2 = net(torch.cat((img_u_s1, img_u_s2)), ret_feats=True)
            pred_u_s1_pred_u_s2, pred_u_s1_b_pred_u_s2_b = net(
                torch.cat((img_u_s1, img_u_s2)), ret_feats=True, drop=True
            )
            pred_u_s1, pred_u_s2 = pred_u_s1_pred_u_s2.chunk(2)
            # bottleneck features
            pred_u_s1_bf, pred_u_s2_bf = pred_u_s1_b_pred_u_s2_b.chunk(2)
            
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)
            loss_x = (
                F.cross_entropy(pred_x, mask_x)
                + dice_loss(pred_x, mask_x)
            ) / 2.0

            loss_u_s1 = dice_loss(
                pred_u_s1,
                mask_u_w,
                ignore=(conf_u_w < conf_thresh).float(),
            )
            
            loss = (loss_x + loss_u_s1) / 2.0
            
            conf_thresh = (
                args.conf_thresh + (1 - args.conf_thresh) * ramps.sigmoid_rampup(iter_num, max_iterations)
            ) * np.log(2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
            iter_num = iter_num + 1
            writer.add_scalar("lr", lr_, iter_num)
            writer.add_scalar("loss/loss_u_s", loss_u_s1, iter_num)
            writer.add_scalar("loss/loss_x", loss_x, iter_num)
            writer.add_scalar("loss/loss", loss, iter_num)
            
            lr_ = base_lr * (1 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
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
                torch.save(net.state_dict(), save_mode_path)
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
            net.eval()
            with torch.no_grad():
                with open(args.root_path + "./sample_test.list", "r") as f:
                    image_list = f.readlines()
                
                if args.dataset_name == "Tooth":
                    image_list = [
                        args.root_path + item.replace("\n", "") + ""
                        for item in image_list
                    ]
                    
                dice = test(net, image_list)


                if dice > pervious_bset_dice:
                    pervious_bset_dice = dice
                    save_mode_path = os.path.join(snapshot_path, "best_model.pth")
                    torch.save(net.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

    save_mode_path = os.path.join(
        snapshot_path, "iter_" + str(max_iterations + 1) + ".pth"
    )
    torch.save(net.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()
