import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import math
import itertools
from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch.nn import LayerNorm
from typing_extensions import Final
#import MultiHeadAttention module
from .transformer import AttentionBlock
from .scanet import SplitCoordAtt

def sparse_init_weight(model):
    for m in model.modules():
        if isinstance(m, nn.Conv3d):
            torch.nn.init.kaiming_normal_(
                m.weight,
            )
        elif isinstance(m, nn.BatchNorm3d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
    return model


class ConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization="none", kernel_size=3):
        super(ConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, kernel_size=kernel_size, padding=kernel_size//2))
            if normalization == "batchnorm":
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == "groupnorm":
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == "instancenorm":
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != "none":
                assert False
            ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class ResidualConvBlock(nn.Module):
    def __init__(self, n_stages, n_filters_in, n_filters_out, normalization="none"):
        super(ResidualConvBlock, self).__init__()

        ops = []
        for i in range(n_stages):
            if i == 0:
                input_channel = n_filters_in
            else:
                input_channel = n_filters_out

            ops.append(nn.Conv3d(input_channel, n_filters_out, 3, padding=1))
            if normalization == "batchnorm":
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == "groupnorm":
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == "instancenorm":
                ops.append(nn.InstanceNorm3d(n_filters_out))
            elif normalization != "none":
                assert False

            if i != n_stages - 1:
                ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x) + x
        x = self.relu(x)
        return x


class DownsamplingConvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization="none"):
        super(DownsamplingConvBlock, self).__init__()

        ops = []
        if normalization != "none":
            ops.append(
                nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride)
            )
            if normalization == "batchnorm":
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == "groupnorm":
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == "instancenorm":
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(
                nn.Conv3d(n_filters_in, n_filters_out, stride, padding=0, stride=stride)
            )

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class UpsamplingDeconvBlock(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization="none"):
        super(UpsamplingDeconvBlock, self).__init__()

        ops = []
        if normalization != "none":
            ops.append(
                nn.ConvTranspose3d(
                    n_filters_in, n_filters_out, stride, padding=0, stride=stride
                )
            )
            if normalization == "batchnorm":
                ops.append(nn.BatchNorm3d(n_filters_out))
            elif normalization == "groupnorm":
                ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
            elif normalization == "instancenorm":
                ops.append(nn.InstanceNorm3d(n_filters_out))
            else:
                assert False
        else:
            ops.append(
                nn.ConvTranspose3d(
                    n_filters_in, n_filters_out, stride, padding=0, stride=stride
                )
            )

        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x


class Upsampling(nn.Module):
    def __init__(self, n_filters_in, n_filters_out, stride=2, normalization="none"):
        super(Upsampling, self).__init__()

        ops = []
        ops.append(
            nn.Upsample(scale_factor=stride, mode="trilinear", align_corners=False)
        )
        ops.append(nn.Conv3d(n_filters_in, n_filters_out, kernel_size=3, padding=1))
        if normalization == "batchnorm":
            ops.append(nn.BatchNorm3d(n_filters_out))
        elif normalization == "groupnorm":
            ops.append(nn.GroupNorm(num_groups=16, num_channels=n_filters_out))
        elif normalization == "instancenorm":
            ops.append(nn.InstanceNorm3d(n_filters_out))
        elif normalization != "none":
            assert False
        ops.append(nn.ReLU(inplace=True))

        self.conv = nn.Sequential(*ops)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class VNetEncoder(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_classes=2,
        n_filters=16,
        normalization="none",
        kernel_size=7,
        SCANet=False
    ):
        super(VNetEncoder, self).__init__()
        
        self.block_one = ConvBlock(
            1, n_channels, n_filters, normalization=normalization, kernel_size=kernel_size
        )
        self.block_one_dw = DownsamplingConvBlock(
            n_filters, 2 * n_filters, normalization=normalization
        )

        self.block_two = ConvBlock(
            2, n_filters * 2, n_filters * 2, normalization=normalization, kernel_size=kernel_size
        ) if not SCANet else SplitCoordAtt(n_filters * 2, n_filters * 2, 3)
        self.block_two_dw = DownsamplingConvBlock(
            n_filters * 2, n_filters * 4, normalization=normalization
        )

        self.block_three = ConvBlock(
            3, n_filters * 4, n_filters * 4, normalization=normalization, kernel_size=kernel_size
        ) if not SCANet else SplitCoordAtt(n_filters * 4, n_filters * 4, 3)
        self.block_three_dw = DownsamplingConvBlock(
            n_filters * 4, n_filters * 8, normalization=normalization
        )

        self.block_four = ConvBlock(
            3, n_filters * 8, n_filters * 8, normalization=normalization, kernel_size=kernel_size
        ) if not SCANet else SplitCoordAtt(n_filters * 8, n_filters * 8, 3)
        self.block_four_dw = DownsamplingConvBlock(
            n_filters * 8, n_filters * 16, normalization=normalization
        )

        self.block_five = ConvBlock(
            3, n_filters * 16, n_filters * 16, normalization=normalization, kernel_size=kernel_size
        ) if not SCANet else SplitCoordAtt(n_filters * 16, n_filters * 16, 3)
        self.dropout = nn.Dropout3d(p=0.5, inplace=False)

    def forward(self, input):
        x1 = self.block_one(input)
        x1_dw = self.block_one_dw(x1)

        x2 = self.block_two(x1_dw)
        x2_dw = self.block_two_dw(x2)

        x3 = self.block_three(x2_dw)
        x3_dw = self.block_three_dw(x3)

        x4 = self.block_four(x3_dw)
        x4_dw = self.block_four_dw(x4)

        x5 = self.block_five(x4_dw)

        x5 = self.dropout(x5)

        res = [x1, x2, x3, x4, x5]

        return res
    
    
class MyMLP(nn.Module):
    def __init__(self, dim, hidden_dim, out_dim, num_layers, dropout=0.):
        super(MyMLP, self).__init__()
        self.layers = nn.ModuleList([])
        self.layers.append(nn.Linear(dim, hidden_dim))
        self.layers.append(nn.ReLU())
        for i in range(num_layers-1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(hidden_dim, out_dim))
        self.layers = nn.Sequential(*self.layers)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        x = self.dropout(x)
        x = self.layers(x)
        return x
    
class VNetDecoder(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_classes=2,
        n_filters=16,
        normalization="none",
        inject_features_dim=0,
        kernel_size=7,
        transformer_attention=False,
        direct_mlp=False,
        SCANet=False
    ):
        super(VNetDecoder, self).__init__()
        
        self.block_five_up = UpsamplingDeconvBlock(
            n_filters * 16, n_filters * 8, normalization=normalization
        )

        self.block_six = ConvBlock(
            3, n_filters * 8, n_filters * 8, normalization=normalization, kernel_size=kernel_size
        ) if not SCANet else SplitCoordAtt(n_filters * 8, n_filters * 8, 3)
        self.block_six_up = UpsamplingDeconvBlock(
            n_filters * 8, n_filters * 4, normalization=normalization
        )

        self.block_seven = ConvBlock(
            3, n_filters * 4, n_filters * 4, normalization=normalization, kernel_size=kernel_size
        ) if not SCANet else SplitCoordAtt(n_filters * 4, n_filters * 4, 3)
        self.block_seven_up = UpsamplingDeconvBlock(
            n_filters * 4, n_filters * 2, normalization=normalization
        )

        self.block_eight = ConvBlock(
            2, n_filters * 2, n_filters * 2, normalization=normalization, kernel_size=kernel_size
        ) if not SCANet else SplitCoordAtt(n_filters * 2, n_filters * 2, 3)
        self.block_eight_up = UpsamplingDeconvBlock(
            n_filters * 2, n_filters, normalization=normalization
        )

        self.block_nine = ConvBlock(
            1, n_filters + inject_features_dim, n_filters, normalization=normalization, kernel_size=kernel_size
        ) if not SCANet else SplitCoordAtt(n_filters + inject_features_dim, n_filters, 3)
        if transformer_attention:
            self.attention = AttentionBlock(dim=n_filters)
            self.block_nine_2 = ConvBlock(
                1, n_filters, n_filters, normalization=normalization, kernel_size=kernel_size
            )
            
        self.out_conv = nn.Conv3d(n_filters, n_classes, 1, padding=0)
        
        if direct_mlp:
            hdw = (112//2)*(112//2)*(80//2)
            self.block_dw = DownsamplingConvBlock(
                n_filters, n_filters * 2, normalization=normalization
            )
            self.mlp = MyMLP(dim=hdw * 2, hidden_dim=hdw, out_dim=hdw*2, num_layers=2, dropout=0.5)
            self.block_up = UpsamplingDeconvBlock(
                2, n_filters, normalization=normalization
            )
            self.out_conv = nn.Conv3d(n_filters * 2, n_classes, 1, padding=0)
            
        
        self.inject_features_dim = inject_features_dim
        self.transformer_attention = transformer_attention
        self.direct_mlp = direct_mlp

    def forward(self, features, inject_feature=None):
        if inject_feature is not None:
            assert self.inject_features_dim != 0, "inject_features_dim is not set"
            
        x1 = features[0]
        x2 = features[1]
        x3 = features[2]
        x4 = features[3]
        x5 = features[4]

        x5_up = self.block_five_up(x5)
        x5_up = x5_up + x4

        x6 = self.block_six(x5_up)
        x6_up = self.block_six_up(x6)
        x6_up = x6_up + x3

        x7 = self.block_seven(x6_up)
        x7_up = self.block_seven_up(x7)
        x7_up = x7_up + x2

        x8 = self.block_eight(x7_up)
        x8_up = self.block_eight_up(x8)
        x8_up = x8_up + x1
        if inject_feature is not None:
            x8_up = torch.cat([x8_up, inject_feature], dim=1)
        x9 = self.block_nine(x8_up)
        if self.transformer_attention:
            x9_attn = self.attention(x9)
            x9 = self.block_nine_2(x9) + x9_attn
        if self.direct_mlp:
            x9_half = self.block_dw(x9)
            B,C,H,W,D = x9_half.shape
            # print(x9.shape)
            x9_half = x9_half.view(B, -1)
            x9_half = self.mlp(x9_half)
            x9_half = x9_half.view(B, C, H, W, D)
            x9_half = self.block_up(x9_half)
            x9 = torch.cat([x9, x9_half], dim=1)
            
        
        out = self.out_conv(x9)
        return out

class VNetMultiTask(nn.Module):
    def __init__(
        self,
        n_channels=3,
        n_classes_1=2,
        n_classes_2=33,
        n_filters=16,
        normalization="none",
        transformer_attention_in_dec2=False,
        direct_mlp=False,
        SCANet=False
    ):
        super(VNetMultiTask, self).__init__()
        
        self.encoder = VNetEncoder(
            n_channels=n_channels, n_classes=n_classes_1, n_filters=n_filters, normalization=normalization,kernel_size=7, SCANet=SCANet
        )
        # self.decoder = VNetDecoder(
        #     n_channels=n_channels, n_classes=n_classes, n_filters=n_filters, normalization=normalization
        # )
        self.binary_decoder = VNetDecoder(
            n_channels=n_channels, n_classes=n_classes_1, n_filters=n_filters, normalization=normalization,kernel_size=7, SCANet=SCANet
        )
        self.multiclass_decoder = VNetDecoder(
            n_channels=n_channels, n_classes=n_classes_2, n_filters=n_filters, normalization=normalization, inject_features_dim=2, kernel_size=7, transformer_attention=transformer_attention_in_dec2, direct_mlp=direct_mlp,
            SCANet=SCANet
        )
        
    def forward(self, input, ret_feats = False, drop=False, out_multiclass=False):
        if ret_feats:
            features = self.encoder(input)
            bottleneck = features[-1]
            if drop:
                features_1, features_2 = [],[]
                for feat in features:
                    feat_1, feat_2 = feat.chunk(2, dim=0)
                    features_1.append(feat_1)
                    features_2.append(feat_2)
            
                # do drop for features_2
                features_2 = [nn.Dropout3d(p=0.5)(feat) for feat in features_2]
                features = [torch.cat([f1, f2], dim=0) for f1, f2 in zip(features_1, features_2)]
            
            out_binary = self.binary_decoder(features)
            if out_multiclass:
                out_multiclass = self.multiclass_decoder(features, out_binary)
                return out_binary, out_multiclass, bottleneck
            else:
                return out_binary, bottleneck
        else:
            features = self.encoder(input)
            if drop:
                features_1, features_2 = [],[]
                for feat in features:
                    feat_1, feat_2 = feat.chunk(2, dim=0)
                    features_1.append(feat_1)
                    features_2.append(feat_2)
            
                # do drop for features_2
                features_2 = [nn.Dropout3d(p=0.5)(feat) for feat in features_2]
                features = [torch.cat([f1, f2], dim=0) for f1, f2 in zip(features_1, features_2)]
            out_binary = self.binary_decoder(features)
            if out_multiclass:
                out_multiclass = self.multiclass_decoder(features, out_binary)
                return out_binary, out_multiclass
            else:
                return out_binary


class Corr3D(nn.Module):
    def __init__(self, nclass=2):
        super(Corr3D, self).__init__()
        self.nclass = nclass

    def forward(self, f1, f2):
        f1 = rearrange(f1, 'n c h w d -> n c (h w d)')
        f2 = rearrange(f2, 'n c h w d -> n c (h w d)')
        # f1 = F.normalize(f1, dim=-1)
        # f2 = F.normalize(f2, dim=-1)
        corr_map = torch.matmul(f1.transpose(1, 2), f2) / torch.sqrt(torch.tensor(f1.shape[1]).float())
        corr_map = F.softmax(corr_map, dim=-1)
        return corr_map
    
if __name__ == "__main__":
    net = VNetMultiTask(n_channels=1, n_classes_1=2, n_classes_2=33, n_filters=16, normalization="batchnorm", transformer_attention_in_dec2=False, direct_mlp=True)
    a = torch.randn(1, 1, 112,112,80)
    b = torch.randn(1, 1, 64,64,64)
    out = net(a)
    print(out.shape)
    