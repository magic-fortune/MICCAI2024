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

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from networks.unet_3D_dv_semi import unet_3D_dv_semi
from utils.losses import DiceLoss
from dataloaders.la_heart import (
    LAHeart,
    RandomCrop,
    RandomRotFlip,
    RandomNoise,
    ToTensor
)
from dataloaders.tooth import Tooth
from test_util import test_all_case, test
from utils import ramps
import time

import subprocess
import os
import time

torch.cuda.empty_cache()
def check_and_kill_tmux():
    try:
        # 执行 w 命令并获取输出
        w_output = subprocess.check_output(['w'], universal_newlines=True)

        # 检查输出中是否包含 "tangyutao"
        if 'tangyutao' in w_output:
            print("检测到 tangyutao，正在终止所有 tmux 进程...")

            # 获取所有 tmux 进程的 PID
            pgrep_output = subprocess.check_output(['pgrep', 'tmux'], universal_newlines=True)
            tmux_pids = pgrep_output.split()

            # 终止每个 tmux 进程
            for pid in tmux_pids:
                try:
                    os.kill(int(pid), 9)  # 9 是 SIGKILL 信号
                    print(f"已终止 tmux 进程，PID: {pid}")
                except ProcessLookupError:
                    print(f"进程 {pid} 不存在或已终止")
                except PermissionError:
                    print(f"没有权限终止进程 {pid}")

            print("所有 tmux 进程已终止")
        else:
            pass
    except subprocess.CalledProcessError:
        print("执行 w 命令时出错")
    except Exception as e:
        print(f"发生错误: {str(e)}")



parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    type=str,
    default="data/MICCAI2024/",
    help="Name of Experiment",
)
parser.add_argument("--exp", type=str, default="distill_match", help="model_name")
parser.add_argument(
    "--max_iterations", type=int, default=10000, help="maximum epoch number to train"
)
parser.add_argument("--batch_size", type=int, default=2, help="batch_size per gpu")
parser.add_argument(
    "--base_lr", type=float, default= 0.0005, help="maximum epoch number to train"
)
parser.add_argument(
    "--deterministic", type=int, default=0, help="whether use deterministic training"
) 
parser.add_argument("--seed", type=int, default=1237, help="random seed")
parser.add_argument("--gpu", type=str, default="0", help="GPU to use")
parser.add_argument("--label_num", type=int, default=27, help="label num")
parser.add_argument(
    "--eta", type=float, default=0.2, help="weight to balance loss"
)
parser.add_argument("--optimizer", type=str, default="AdamW", help="optimizer")
parser.add_argument("--conf_thresh", type=float, default=0.85, help="conf_thresh")
parser.add_argument("--temperature", type=float, default=1, help="temperature")

parser.add_argument('--pert_gap', type=float, default=0.00, help='the perturbation gap')
parser.add_argument('--pert_type', type=str, default='dropout', help='feature pertubation types')
parser.add_argument('--dataset_name', type=str, default='Tooth', help='dataset_name')
parser.add_argument('--use_dec_loss_weak', type=int, default=1, help='whether use dec loss weak')
parser.add_argument('--use_dec_loss_strong', type=int, default=1, help='whether use dec loss strong')
parser.add_argument('--dec_loss_type', type=str, default='dice', help='dec loss type')
parser.add_argument('--sup_pww', type=int, default=1)
parser.add_argument('--sup_psw', type=int, default=1)
parser.add_argument('--sup_pws', type=int, default=1)
parser.add_argument('--sup_pss', type=int, default=1)
parser.add_argument('--sup_strong1', type=int, default=1)
parser.add_argument('--sup_strong2', type=int, default=1)
parser.add_argument('--sup_loss_type', type=str, default='mix', help='sup loss type')

args = parser.parse_args()

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

patch_size = (112, 112, 160)
num_classes = 33
dice_loss = DiceLoss(n_classes=num_classes).cuda()

if args.dataset_name == "LA":
    patch_size = (112, 112, 80)
    args.root_path = '../data/2018LA_Seg_Training Set/'
    args.max_samples = 80
    DATASET_CLASS = LAHeart
    TSFM = transforms.Compose(
        [
            # RandomRotFlip(),
            RandomCrop(patch_size),
            ToTensor(),
        ]
    )
elif args.dataset_name == "Pancreas_CT":
    patch_size = (96, 96, 96)
    args.root_path = '../data/Pancreas/'
    args.max_samples = 62
    DATASET_CLASS = Pancreas
    TSFM = transforms.Compose(
        [
            RandomCrop(patch_size),
            ToTensor(),
        ]
    )
elif args.dataset_name == 'Tooth':
    patch_size = (128, 128, 128)
    args.root_path = 'data/MICCAI2024/'
    args.max_samples = 300
    DATASET_CLASS = Tooth
    TSFM = transforms.Compose(
        [
            # RandomNoise(),
            # RandomRotFlip(),
            RandomCrop(patch_size),
            ToTensor(),
        ]
    )

train_data_path = args.root_path
snapshot_path = "code_now/" + args.exp + "/"

LABELED_ID_NUM = args.label_num  # 27 
conf_thresh = args.conf_thresh
eta = args.eta
pert_gap = args.pert_gap

pervious_bset_dice = 0.0

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if not os.path.exists(snapshot_path + 'code'):
        os.makedirs(snapshot_path + 'code')
    if os.path.exists(snapshot_path + "/code"):
        shutil.rmtree(snapshot_path + "/code")
    
    shutil.copy('code_now/atrain_distill_match.py', snapshot_path + 'code.py')


    logging.basicConfig(
        filename=snapshot_path + "/log.txt",
        level=logging.INFO,
        format="[%(asctime)s.%(msecs)03d] %(message)s",
        datefmt="%H:%M:%S",
    )
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))


    net = VNet(
        n_channels=1, n_classes=num_classes, normalization="batchnorm", has_dropout=True, pert_gap=pert_gap,
        pert_type=args.pert_type
    )
    # net = unet_3D_dv_semi(feature_scale=4, n_classes=33, is_deconv=True, in_channels=1, is_batchnorm=True)
    net = net.cuda()
    # path_weight = "/code_now/distill_match_last/best_model.pth"
    # check_point = torch.load(path_weight, weights_only=True)
    # net.load_state_dict(check_point)
    
    
    
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    trainset_u = DATASET_CLASS(
        base_dir=train_data_path,
        mode="train_u",
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
        worker_init_fn=worker_init_fn,
    )
    trainsampler_u_mix = torch.utils.data.sampler.RandomSampler(trainset_u, replacement=True)
    trainloader_u_mix = DataLoader(
        trainset_u,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=batch_size * 2,
        drop_last=True,
        sampler=trainsampler_u_mix,
        worker_init_fn=worker_init_fn,
    )

    trainset_l = DATASET_CLASS(
        base_dir=train_data_path,
        mode="train_l",
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
        worker_init_fn=worker_init_fn,
    )

    net.train()
    if args.optimizer == "SGD":
        optimizer = optim.SGD(
            net.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0002 if LABELED_ID_NUM == 16 else 0.001
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
    bef = -1
    for epoch_num in tqdm(range(max_epoch), ncols=70):

        # time1 = time.time()
        net.train()
        # torch.autograd.set_detect_anomaly(True)

        for i_batch, (
            (img_x, mask_x),
            (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
            (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _),
        ) in enumerate(zip(trainloader_l, trainloader_u, trainloader_u_mix)):
            
            # check_and_kill_tmux()
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()
            
            # print(f"{img_x.min() = }, {img_x.max() = } \n {img_u_s1.max() = } {img_u_s2.max() = } \n {torch.unique(mask_x)}")
            # print(f"{img_x.shape = }, {img_u_s1.shape = }, {img_u_s2.shape = }, {mask_x.shape = }")
            
            # bef = time.time()
            with torch.no_grad():
                net.eval()
                pred_u_w_mix = net(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[
                cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1
            ] = img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[
                cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1
            ] = img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            net.train()
            # print(img_x.shape)
            pred_x, pred_u_w, pred_u_w_weak, pred_u_w_strong = net(
                torch.cat((img_x, img_u_w)),
                need_fp=True
            )
            

            pred_u_s1, pred_u_s2, pred_u_s2_weak, pred_u_s2_strong = net(torch.cat((img_u_s1, img_u_s2)), need_fp=True)
            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)
            
            # pred_u_w_weak = pred_u_w_weak.detach()
            conf_u_w_weak = pred_u_w_weak.softmax(dim=1).max(dim=1)[0]
            mask_u_w_weak = pred_u_w_weak.argmax(dim=1)
            pred_u_w_weak_cutmixed2 = pred_u_w_weak.clone()
            
            cutmix_box2_2ch = cutmix_box2.unsqueeze(1).expand(pred_u_w_weak.shape) == 1
            
            pred_u_w_weak_cutmixed2[cutmix_box2_2ch] = pred_u_w_mix[cutmix_box2_2ch]
            
            
            # pred_u_w_strong = pred_u_w_strong.detach()
            conf_u_w_strong = pred_u_w_strong.softmax(dim=1).max(dim=1)[0]
            mask_u_w_strong = pred_u_w_strong.argmax(dim=1)
            pred_u_w_strong_cutmixed2 = pred_u_w_strong.clone()
            pred_u_w_strong_cutmixed2[cutmix_box2_2ch] = pred_u_w_mix[cutmix_box2_2ch]
            
            
            # pred_u_s2_weak = pred_u_s2_weak.detach()
            conf_u_s2_weak = pred_u_s2_weak.softmax(dim=1).max(dim=1)[0]
            mask_u_s2_weak = pred_u_s2_weak.argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()
            mask_u_w_weak_cutmixed2, conf_u_w_weak_cutmixed2 = mask_u_w_weak.clone(), conf_u_w_weak.clone()
            mask_u_w_strong_cutmixed2, conf_u_w_strong_cutmixed2 = mask_u_w_strong.clone(), conf_u_w_strong.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]
            
            mask_u_w_weak_cutmixed2[cutmix_box2 == 1] = mask_u_w_weak[cutmix_box2 == 1]
            conf_u_w_weak_cutmixed2[cutmix_box2 == 1] = conf_u_w_weak[cutmix_box2 == 1]
            
            mask_u_w_strong_cutmixed2[cutmix_box2 == 1] = mask_u_w_strong[cutmix_box2 == 1]
            conf_u_w_strong_cutmixed2[cutmix_box2 == 1] = conf_u_w_strong[cutmix_box2 == 1]
            
            
            if args.sup_loss_type == 'mix':
                loss_x = (
                    F.cross_entropy(pred_x, mask_x)
                    + dice_loss(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())
                ) / 2.0
            elif args.sup_loss_type == 'ce':
                loss_x = F.cross_entropy(pred_x, mask_x) 
            elif args.sup_loss_type == 'dice':
                loss_x = dice_loss(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())
                
            
            if args.sup_strong1 == 1:
                loss_u_s1 = dice_loss(
                    pred_u_s1.softmax(dim=1),
                    mask_u_w_cutmixed1.unsqueeze(1).float(),
                    ignore=(conf_u_w_cutmixed1 < conf_thresh).float(),
                )
            else:
                loss_u_s1 = 0.0
                
                
            if args.sup_strong2 == 1:
                loss_u_s2 = dice_loss(
                    pred_u_s2.softmax(dim=1),
                    mask_u_w_cutmixed2.unsqueeze(1).float(),
                    ignore=(conf_u_w_cutmixed2 < conf_thresh).float(),
                )
            else:
                loss_u_s2 = 0.0
            
            # if args.use_dec_loss_weak:
            #     # pred_u_w_weak to supervise pred_u_s2_weak
            #     if args.dec_loss_type == 'dice':
            #         loss_weak_dec_w_s2 = dice_loss(
            #             pred_u_s2_weak.softmax(dim=1),
            #             mask_u_w_weak_cutmixed2.unsqueeze(1).float(),
            #             ignore=(conf_u_w_weak_cutmixed2 < conf_thresh).float(),
            #         )
            #     elif args.dec_loss_type == 'ce':
            #         loss_weak_dec_w_s2 = F.cross_entropy(
            #             pred_u_s2_weak, mask_u_w_weak_cutmixed2, ignore_index=0
            #         )
            #     elif args.dec_loss_type == 'kl':
            #         loss_weak_dec_w_s2 = F.kl_div(
            #             F.log_softmax(pred_u_s2_weak / args.temperature, dim=1),
            #             F.softmax(pred_u_w_weak_cutmixed2 / args.temperature, dim=1)
            #         )
            #         #  0.5 + F.kl_div(
            #         #     F.log_softmax(pred_u_w_weak_cutmixed2 / args.temperature, dim=1),
            #         #     F.softmax(pred_u_s2_weak / args.temperature, dim=1)
            #         # ) * 0.5

            # else:
            #     loss_weak_dec_w_s2 = 0.0
                
            # if args.use_dec_loss_strong:
            #     # pred_u_w_strong to supervise pred_u_s2_strong
            #     if args.dec_loss_type == 'dice':
            #         loss_strong_dec_w_s2 = dice_loss(
            #             pred_u_s2_strong.softmax(dim=1),
            #             mask_u_w_strong_cutmixed2.unsqueeze(1).float(),
            #             ignore=(conf_u_w_strong_cutmixed2 < conf_thresh).float(),
            #         )
            #     elif args.dec_loss_type == 'ce':
            #         loss_strong_dec_w_s2 = F.cross_entropy(
            #             pred_u_s2_strong, mask_u_w_strong_cutmixed2, ignore_index=0
            #         )
            #     elif args.dec_loss_type == 'kl':
            #         loss_strong_dec_w_s2 = F.kl_div(
            #             F.log_softmax(pred_u_s2_strong / args.temperature, dim=1),
            #             F.softmax(pred_u_w_strong_cutmixed2 / args.temperature, dim=1)
            #         )
                
            #         # * 0.5 + F.kl_div(
            #         #     F.log_softmax(pred_u_w_strong_cutmixed2 / args.temperature, dim=1),
            #         #     F.softmax(pred_u_s2_strong / args.temperature, dim=1)
            #         # ) * 0.5
                    
                
            # else:
            #     loss_strong_dec_w_s2 = 0.0
            
            # if loss_strong_dec_w_s2 == 0.0 and loss_weak_dec_w_s2 == 0.0:
            #     eta = 0.0
            
            # ignores = (conf_u_w < conf_thresh).float()
            # # pred_u_w to supervise pred_u_w_weak
            # if args.sup_pww == 1:
            #     loss_weak_dec_w_t = dice_loss(
            #         pred_u_w_weak.softmax(dim=1),
            #         mask_u_w.unsqueeze(1).float(),
            #         ignore=ignores,
            #     )
            # else: loss_weak_dec_w_t = 0.0
            
            # pred_u_w to supervise pred_u_w_strong
            # if args.sup_pws == 1:
            #     loss_strong_dec_w_t = dice_loss(
            #         pred_u_w_strong.softmax(dim=1),
            #         mask_u_w.unsqueeze(1).float(),
            #         ignore=ignores,
            #     )
            # else: loss_strong_dec_w_t = 0.0
            
            # # pred_u_w to supervise pred_u_s2_weak
            # if args.sup_psw == 1:
            #     loss_weak_dec_w_s2_t = dice_loss(
            #         pred_u_s2_weak.softmax(dim=1),
            #         mask_u_w_cutmixed2.unsqueeze(1).float(),
            #         ignore=(conf_u_w_cutmixed2 < conf_thresh).float(),
            #     )
            # else: loss_weak_dec_w_s2_t = 0.0
            
            # # pred_u_w to supervise pred_u_s2_strong
            # if args.sup_pss == 1:
            #     loss_strong_dec_w_s2_t = dice_loss(
            #         pred_u_s2_strong.softmax(dim=1),
            #         mask_u_w_cutmixed2.unsqueeze(1).float(),
            #         ignore=(conf_u_w_cutmixed2 < conf_thresh).float(),
            #     )
            # else: loss_strong_dec_w_s2_t = 0.0
            
            # weig_sup = args.sup_pww + args.sup_psw + args.sup_pws + args.sup_pss
            
            # loss_u_w_kd = (1-eta) * (
            #     loss_weak_dec_w_t + loss_strong_dec_w_t + loss_weak_dec_w_s2_t + loss_strong_dec_w_s2_t
            # ) / weig_sup + eta * (loss_weak_dec_w_s2 + loss_strong_dec_w_s2) / 2.0
            loss_u_w_kd = 0.00
            
            loss = (loss_x * 1.5 + loss_u_s1 * 0.25 + loss_u_s2 * 0.25) / 2.0
            
            # conf_thresh = (
            #     args.conf_thresh + (1 - args.conf_thresh) * ramps.sigmoid_rampup(iter_num, 17000)
            # ) * np.log(2)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # print(time.time() - bef)

            iter_num = iter_num + 1
            writer.add_scalar("lr", lr_, iter_num)
            writer.add_scalar("loss/loss_u_kd", loss_u_w_kd, iter_num)
            writer.add_scalar("loss/loss_x", loss_x, iter_num)
            writer.add_scalar("loss/loss", loss, iter_num)
            
            # lr_ = base_lr * (1 - iter_num / 17000) ** 0.9
            lr_ = base_lr * (1 - iter_num / max_iterations) ** 0.9
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_

            if iter_num % 1 == 0:
                image = (
                    img_x[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                )
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image("train/Image", grid_image, iter_num)
                # print(pred_x.shape)
                outputs_soft = torch.argmax(pred_x, 1)
                image = (
                    outputs_soft[0, :, :, 20:61:10]
                    .unsqueeze(0)
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
            # time1 = time.time()
        if iter_num > max_iterations:
            print("finish training")
            break

        if (epoch_num + 1) % 10 == 0:
            # evals
            net.eval()
            with torch.no_grad():
                with open("data/MICCAI2024/sample_test.list", "r") as f:
                    image_list = f.readlines()
                
                if args.dataset_name == "LA":
                    image_list = [
                        args.root_path + item.replace("\n", "") + "/mri_norm2.h5"
                        for item in image_list
                    ]

                    dice, jc, hd, asd = test_all_case(
                        net,
                        image_list,
                        num_classes=num_classes,
                        patch_size=patch_size,
                        stride_xy=18,
                        stride_z=4,
                        save_result=False,
                        test_save_path=None,
                    )
                elif args.dataset_name == "Pancreas_CT":
                    image_list = [args.root_path + "/Pancreas_h5/" + item.replace('\n', '') + "_norm.h5" for item in image_list]

                    dice, jc, hd, asd = test_all_case(
                        net,
                        image_list,
                        num_classes=num_classes,
                        patch_size=patch_size,
                        stride_xy=16,
                        stride_z=16,
                        save_result=False,
                        test_save_path=None,
                    )
                    
                elif args.dataset_name == "Tooth":
                    image_list = [args.root_path + "/alls_data/lab/" + item.replace('\n', '')  for item in image_list]

                    metric = test(net, image_list)
                    # metric = test_all_case(
                    #     net,
                    #     image_list,
                    #     num_classes=num_classes,
                    #     patch_size=patch_size,
                    #     stride_xy=36,
                    #     stride_z=36,
                    #     save_result=False,
                    #     test_save_path=None,
                    # )

                if metric > pervious_bset_dice:
                    pervious_bset_dice = metric
                    logging.info(f"********************* Best Eval: {metric}************************")
                    save_mode_path = os.path.join(snapshot_path, "best_model.pth")
                    torch.save(net.state_dict(), save_mode_path)
                    logging.info("save model to {}".format(save_mode_path))

        save_mode_path = os.path.join(
            snapshot_path,  "last.pth"
        )
        torch.save(net.state_dict(), save_mode_path)
        logging.info("save model to {}".format(save_mode_path))
