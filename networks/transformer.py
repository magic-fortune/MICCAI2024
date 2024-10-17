import types
from typing import Optional
import torch.nn as nn
from timm.layers import DropPath, trunc_normal_, create_conv2d, ConvNormAct, SqueezeExcite, use_fused_attn
import torch


class Attention(nn.Module):
    """Multi-headed Self Attention module.

    Source modified from:
    https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
    """
    fused_attn: torch.jit.Final[bool]

    def __init__(
            self,
            dim: int,
            head_dim: int = 32,
            qkv_bias: bool = False,
            attn_drop: float = 0.0,
            proj_drop: float = 0.0,
    ) -> None:
        """Build MHSA module that can handle 3D or 4D input tensors.

        Args:
            dim: Number of embedding dimensions.
            head_dim: Number of hidden dimensions per head. Default: ``32``
            qkv_bias: Use bias or not. Default: ``False``
            attn_drop: Dropout rate for attention tensor.
            proj_drop: Dropout rate for projection tensor.
        """
        super().__init__()
        # print(dim, head_dim)
        assert dim % head_dim == 0, "dim should be divisible by head_dim"
        self.head_dim = head_dim
        self.num_heads = dim // head_dim
        self.scale = head_dim ** -0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W, D = x.shape
        N = H * W * D
        x = x.flatten(2).transpose(-2, -1)  # (B, N, C)
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        if self.fused_attn:
            x = torch.nn.functional.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.transpose(-2, -1).reshape(B, C, H, W, D)

        return x


    
class LayerScale3d(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma



class ConvNormAct3d(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 1,
            stride: int = 1,
            padding = '',
            dilation: int = 1,
            groups: int = 1,
            bias: bool = False,
            apply_norm: bool = True,
            apply_act: bool = True,
            norm_layer = nn.BatchNorm3d,
            act_layer = nn.ReLU,
            aa_layer = None,
            drop_layer = None,
            conv_kwargs = None,
            norm_kwargs = None,
            act_kwargs = None,
    ):
        super(ConvNormAct3d, self).__init__()
        conv_kwargs = conv_kwargs or {}
        norm_kwargs = norm_kwargs or {}
        act_kwargs = act_kwargs or {}
        use_aa = aa_layer is not None and stride > 1

        # self.conv = create_conv2d(
        #     in_channels,
        #     out_channels,
        #     kernel_size,
        #     stride=1 if use_aa else stride,
        #     padding=padding,
        #     dilation=dilation,
        #     groups=groups,
        #     bias=bias,
        #     **conv_kwargs,
        # )
        self.conv = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=1 if use_aa else stride,
            padding=kernel_size // 2,
            dilation=dilation,
            groups=groups,
            bias=bias,
            **conv_kwargs,
        )

        if apply_norm:
            # # NOTE for backwards compatibility with models that use separate norm and act layer definitions
            # norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
            # # NOTE for backwards (weight) compatibility, norm layer name remains `.bn`
            # if drop_layer:
            #     norm_kwargs['drop_layer'] = drop_layer
            # self.bn = norm_act_layer(
            #     out_channels,
            #     apply_act=apply_act,
            #     act_kwargs=act_kwargs,
            #     **norm_kwargs,
            # )
            self.bn = nn.BatchNorm3d(out_channels)
        else:
            self.bn = nn.Sequential()
            if drop_layer:
                norm_kwargs['drop_layer'] = drop_layer
                self.bn.add_module('drop', drop_layer())


    @property
    def in_channels(self):
        return self.conv.in_channels

    @property
    def out_channels(self):
        return self.conv.out_channels

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class ConvMlp(nn.Module):
    """Convolutional FFN Module."""

    def __init__(
            self,
            in_chs: int,
            hidden_channels: Optional[int] = None,
            out_chs: Optional[int] = None,
            act_layer: nn.Module = nn.GELU,
            drop: float = 0.0,
    ) -> None:
        """Build convolutional FFN module.

        Args:
            in_chs: Number of input channels.
            hidden_channels: Number of channels after expansion. Default: None
            out_chs: Number of output channels. Default: None
            act_layer: Activation layer. Default: ``GELU``
            drop: Dropout rate. Default: ``0.0``.
        """
        super().__init__()
        out_chs = out_chs or in_chs
        hidden_channels = hidden_channels or in_chs
        self.conv = ConvNormAct3d(
            in_chs,
            out_chs,
            kernel_size=7,
            groups=in_chs,
            apply_act=False,
        )
        self.fc1 = nn.Conv3d(in_chs, hidden_channels, kernel_size=1)
        self.act = act_layer()
        self.fc2 = nn.Conv3d(hidden_channels, out_chs, kernel_size=1)
        self.drop = nn.Dropout(drop)
        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Conv2d):
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionBlock(nn.Module):
    """Implementation of metaformer block with MHSA as token mixer.

    For more details on Metaformer structure, please refer to:
    `MetaFormer Is Actually What You Need for Vision <https://arxiv.org/pdf/2111.11418.pdf>`_
    """

    def __init__(
            self,
            dim: int,
            mlp_ratio: float = 4.0,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = nn.BatchNorm3d,
            proj_drop: float = 0.0,
            drop_path: float = 0.0,
            layer_scale_init_value: float = 1e-5,
    ):
        """Build Attention Block.

        Args:
            dim: Number of embedding dimensions.
            mlp_ratio: MLP expansion ratio. Default: 4.0
            act_layer: Activation layer. Default: ``nn.GELU``
            norm_layer: Normalization layer. Default: ``nn.BatchNorm2d``
            proj_drop: Dropout rate. Default: 0.0
            drop_path: Drop path rate. Default: 0.0
            layer_scale_init_value: Layer scale value at initialization. Default: 1e-5
        """

        super().__init__()

        self.norm = norm_layer(dim)
        self.token_mixer = Attention(dim=dim, head_dim=16)
        if layer_scale_init_value is not None:
            self.layer_scale_1 = LayerScale3d(dim, layer_scale_init_value)
        else:
            self.layer_scale_1 = nn.Identity()
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.mlp = ConvMlp(
            in_chs=dim,
            hidden_channels=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        if layer_scale_init_value is not None:
            self.layer_scale_2 = LayerScale3d(dim, layer_scale_init_value)
        else:
            self.layer_scale_2 = nn.Identity()
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path1(self.layer_scale_1(self.token_mixer(self.norm(x))))
        x = x + self.drop_path2(self.layer_scale_2(self.mlp(x)))
        return x


if __name__ == '__main__':
    # Test the AttentionBlock
    block = AttentionBlock(dim=32)
    x = torch.randn(1, 32, 16, 16, 16)
    out = block(x)
    print(out.shape)
    print("AttentionBlock test passed")