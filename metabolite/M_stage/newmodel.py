from functools import partial
from collections import OrderedDict
import torch
import torch.nn as nn
import copy
# from .customized_linear import CustomizedLinear
# from einops import rearrange
import random
import numpy as np
import pandas as pd
import torch.nn.functional as F
from random import randint

def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


def normalize(ar):
    min_v = np.min(ar)
    max_v = np.max(ar)
    for i in ar:
        i = (i - min_v) / (max_v - min_v)

    return ar


def gumbel_softmax(x, dim, tau):
    gumbels = torch.rand_like(x)
    while bool((gumbels == 0).sum() > 0):
        gumbels = torch.rand_like(x)

    gumbels = -(-gumbels.log()).log()
    gumbels = (x + gumbels) / tau
    x = gumbels.softmax(dim)

    return x


class BaseLoss(nn.Module):
    def __init__(self, device):
        super(BaseLoss, self).__init__()
        self.base_function = torch.nn.CrossEntropyLoss()
        self.device = device

    def forward(self, output, target):
        # print('output', output.shape)
        # print('target', target.shape)
        base_loss = self.base_function(output, target.to(self.device).long())

        return base_loss


class MaskLoss(nn.Module):
    def __init__(self, lambda_mask=5.0, constraint_lambda=1.0):
        super(MaskLoss, self).__init__()
        self.lambda_mask = lambda_mask
        self.constraint_lambda = constraint_lambda

    def forward(self, important_loss, unimportant_loss, mask, mask_permute, mask_):
        mask_mean = torch.mean(mask)
        mask_constraint_loss = torch.relu(mask_mean)
        loss_contrastive = torch.mean(important_loss - unimportant_loss)
        # loss_contrastive = 1
        permute_loss = torch.mean((mask_ - mask_permute) ** 2)

        regularization_term = torch.mean(torch.abs(mask - 1))

        total_loss = self.lambda_mask * loss_contrastive + self.constraint_lambda * mask_constraint_loss + permute_loss + regularization_term
        return total_loss, loss_contrastive, mask_constraint_loss, permute_loss


class MaskNet(nn.Module):
    def __init__(self, in_features, out_feature):
        super().__init__()
        self.in_features = in_features
        self.out_feature = out_feature
        self.fc1 = nn.Linear(self.in_features, self.in_features * 4)
        self.fc2 = nn.Linear(self.in_features * 4, self.in_features * 4)
        self.fc3 = nn.Linear(self.in_features * 4, self.out_feature * 2)
        self.act = nn.GELU()
        # self.drop = nn.Dropout(0.2)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        # mask 网络不加dropout
        # x = self.drop(x)
        x = self.fc3(x)
        # x = torch.sigmoid(x)
        return x


class MaskNetStatic(nn.Module):
    def __init__(self, in_features, out_feature):
        super().__init__()
        self.mask = nn.Parameter(torch.rand(1, in_features))
        self.in_features = in_features
        self.out_feature = out_feature
        self.fc1 = nn.Linear(self.in_features, self.in_features * 4)
        self.fc2 = nn.Linear(self.in_features * 4, self.in_features * 4)
        self.fc3 = nn.Linear(self.in_features * 4, self.out_feature * 2)
        self.act = nn.GELU()
        # self.drop = nn.Dropout(0.2)

    def forward(self):
        x = self.fc1(self.mask)
        x = self.act(x)
        # x = self.drop(x)
        x = self.fc2(x)
        x = self.act(x)
        # mask 网络不加dropout
        # x = self.drop(x)
        x = self.fc3(x)
        # print('x', x.shape)
        # x = torch.sigmoid(x)
        return x


class Embedding(nn.Module):
    def __init__(self, num_mz, embed_dim, ismask, is_all_transformer, is_all_mlp, bias=True, norm_layer=None):
        super().__init__()
        self.num_mz = num_mz
        self.embed_dim = embed_dim
        self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()
        # self.mask = nn.Parameter(torch.randint(0,1e6,(num_mz,2)))
        self.ismask = ismask
        self.is_all_transformer = is_all_transformer
        self.is_all_mlp = is_all_mlp
        self.identity = torch.eye(self.num_mz)

        # self.MaskNet = MaskNet(in_features=num_mz, out_feature=num_mz)
        self.MaskNet = MaskNetStatic(in_features=num_mz, out_feature=num_mz)

        self.input_drop = nn.Dropout(0)

    def forward(self, x, important=True, tau=0.001, epoch=300):
        # print('self.ismask', self.ismask)
        x = self.input_drop(x)
        if x.dim() == 2:
            bs, c = x.shape
        else:
            bs, n, c = x.shape
        if self.ismask == True:
            # mask = self.MaskNet(x)
            mask = self.MaskNet()
            mask = mask.repeat(bs, 1)
            mask = mask.reshape(bs, c, 2)
            # print('mask_1', mask.shape)
            # print('tau', tau)
            if self.training == True:
                # print('here0')
                index_num = randint(0, 1)
                if index_num == 0:
                    mask = gumbel_softmax(mask, 2, tau)[:, :, 0]
                else:
                    mask = gumbel_softmax(mask, 2, tau)[:, :, 1]
                    mask = 1 - mask
            else:
                mask = gumbel_softmax(mask, 2, tau)
                # print('here1')
                if important == True:
                    mask = (mask[:, :, 0] > mask[:, :, 1]).float()
                else:
                    mask = (mask[:, :, 1] > mask[:, :, 0]).float()
                # print('mask', torch.mean(mask))
            # print('mask_2', mask.shape)
            # if important:
            # print('x', x.shape)
            # print(epoch)
            # if epoch > 30:
            x = torch.mul(x, mask)

            # else:
            #     x = torch.mul(x, 1 - mask)
            # x = torch.mul(x, mask)
            # x = x + 1e-6
            x = x.repeat_interleave(self.embed_dim, dim=1).reshape(-1, self.embed_dim, self.num_mz).permute(0, 2, 1)
            # x = torch.unsqueeze(x, dim=2)
            # x = x.repeat_interleave(self.embed_dim, dim=2)
            x = self.norm(x)
            # print(x.shape)
            return x, mask
        else:
            # print('x', x.shape)
            if self.is_all_transformer == True or self.is_all_mlp == True:
                pass
            else:
                x = x.repeat_interleave(self.embed_dim, dim=1).reshape(-1, self.embed_dim, self.num_mz).permute(0, 2, 1)

            # x = torch.unsqueeze(x, dim=2)
            # x = x.repeat_interleave(self.embed_dim, dim=2)
            x = self.norm(x)
            # print(x.shape)
            return x, None


# class Embedding(nn.Module):
#     def __init__(self, num_mz, embed_dim, ismask, bias=True, norm_layer=None):
#         super().__init__()
#         self.num_mz = num_mz
#         self.embed_dim = embed_dim
#         self.norm = norm_layer(self.embed_dim) if norm_layer else nn.Identity()
#         # self.mask = nn.Parameter(torch.randint(0,1e6,(num_mz,2)))
#         self.ismask = ismask
#         self.identity = torch.eye(self.num_mz)
#
#         # self.MaskNet = MaskNet(in_features=num_mz,out_feature=num_mz)
#         self.MaskNet = MaskNetStatic(in_features=num_mz, out_feature=num_mz)
#
#     def forward(self, x, important=True, tau=0.001, epoch=300):
#         # print('self.ismask', self.ismask)
#         bs, c = x.shape
#         if self.ismask == True:
#             # print('x', x.shape)
#             # dynamic
#             # mask = self.MaskNet(x)
#             # print('mask', mask.shape)
#             # static
#             mask = self.MaskNet()
#             mask = mask.repeat(bs, 1)
#
#             mask = mask.reshape(bs, c, 2)
#             # print('mask_1', mask.shape)
#             if self.training == True:
#                 mask = gumbel_softmax(mask, 2, tau)[:, :, 0]
#             else:
#                 mask = gumbel_softmax(mask, 2, tau)
#                 if important == True:
#                     mask = (mask[:, :, 0] > mask[:, :, 1]).float()
#                 else:
#                     mask = (mask[:, :, 1] > mask[:, :, 0]).float()
#             # print('mask_2', mask.shape)
#             # if important:
#             # print('x', x.shape)
#             # print(epoch)
#             # if epoch > 30:
#             # x = torch.mul(x, mask)
#
#             # else:
#             #     x = torch.mul(x, 1 - mask)
#             x = torch.mul(x, mask)
#             # x = x + 1e-6
#             x = x.repeat_interleave(self.embed_dim, dim=1).reshape(-1, self.embed_dim, self.num_mz).permute(0, 2, 1)
#             # x = torch.unsqueeze(x, dim=2)
#             # x = x.repeat_interleave(self.embed_dim, dim=2)
#             x = self.norm(x)
#             # print(x.shape)
#             return x, mask
#         else:
#             x = x.repeat_interleave(self.embed_dim, dim=1).reshape(-1, self.embed_dim, self.num_mz).permute(0, 2, 1)
#             # x = torch.unsqueeze(x, dim=2)
#             # x = x.repeat_interleave(self.embed_dim, dim=2)
#             x = self.norm(x)
#             # print(x.shape)
#             return x, None

class Attention(nn.Module):
    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop_ratio=0.2,
                 proj_drop_ratio=0.2):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
    def forward(self, x):
        B, N, C = x.shape
        # print(x.shape)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        # print(qkv.shape)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        weights = attn
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, weights


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(nn.Module):
    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop_ratio=0.,
                 attn_drop_ratio=0.,
                 drop_path_ratio=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)
        # self.norm1 = nn.Identity()
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                              attn_drop_ratio=attn_drop_ratio, proj_drop_ratio=drop_ratio)
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        # self.norm2 = nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x.to(torch.float32)

        x = self.norm1(x)
        hhh, weights = self.attn(x)
        # hhh, weights = self.attn(self.norm1(x))

        x = x + self.drop_path(hhh)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, weights


class LinearBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LinearBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class LinearUNet(nn.Module):
    def __init__(self,
                 dim,
                 # num_heads,
                 # mlp_ratio=4.,
                 # qkv_bias=False,
                 # qk_scale=None,
                 drop_ratio=0.,
                 # attn_drop_ratio=0.,
                 # drop_path_ratio=0.,
                 # act_layer=nn.GELU,
                 # norm_layer=nn.LayerNorm
                 ):
        super(LinearUNet, self).__init__()
        self.l1_d = nn.Linear(58752, 512)
        self.l2_d = nn.Linear(512, 256)
        self.l3_d = nn.Linear(256, 128)
        self.l4_d = nn.Linear(128, 64)

        self.l4_u = nn.Linear(64 + 128, 128)
        self.l3_u = nn.Linear(128 + 256, 256)
        self.l2_u = nn.Linear(256 + 512, 512)
        self.l1_u = nn.Linear(512, 2)
        self.drop = nn.Dropout(drop_ratio)

        self.lrelu = nn.LeakyReLU()

    def forward(self, x):
        # x = x + self.drop_path(self.attn(self.norm1(x)))
        # print('x', x.shape)
        # exit(0)
        bs, n, c = x.shape
        x = x.to(torch.float32)
        x = x.reshape(bs, n * c)

        # print(x, x.shape)
        # exit(0)
        x1 = self.drop(self.lrelu(self.l1_d(x)))
        x2 = self.drop(self.lrelu(self.l2_d(x1)))
        x3 = self.drop(self.lrelu(self.l3_d(x2)))
        x4 = self.drop(self.lrelu(self.l4_d(x3)))

        x3 = self.drop(self.lrelu(self.l4_u(torch.cat([x4, x3], dim=1))))
        x2 = self.drop(self.lrelu(self.l3_u(torch.cat([x3, x2], dim=1))))
        x1 = self.drop(self.lrelu(self.l2_u(torch.cat([x2, x1], dim=1))))
        x = self.l1_u(x1)

        return x


# class Transformer(nn.Module):
#     def __init__(self, num_classes, num_genes, tau=0.2, ismask=False, fe_bias=True,
#                  embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
#                  qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
#                  attn_drop_ratio=0., drop_path_ratio=0., embed_layer=Embedding, norm_layer=None,
#                  act_layer=None, need_cal_weight=True):
#         """
#         Args:
#             num_classes (int): number of classes for classification head
#             num_genes (int): number of feature of input(expData)
#             embed_dim (int): embedding dimension
#             depth (int): depth of transformer
#             num_heads (int): number of attention heads
#             mlp_ratio (int): ratio of mlp hidden dim to embedding dim
#             qkv_bias (bool): enable bias for qkv if True
#             qk_scale (float): override default qk scale of head_dim ** -0.5 if set
#             representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
#             distilled (bool): model includes a distillation token and head as in DeiT models
#             drop_ratio (float): dropout rate
#             attn_drop_ratio (float): attention dropout rate
#             drop_path_ratio (float): stochastic depth rate
#             embed_layer (nn.Module): feature embed layer
#             norm_layer: (nn.Module): normalization layer
#         """
#         super(Transformer, self).__init__()
#         self.m = LinearUNet(num_genes, drop_ratio=drop_ratio)
#         self.num_classes = num_classes
#         self.num_features = self.embed_dim = embed_dim
#         self.num_tokens = 2 if distilled else 1
#         norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
#         act_layer = act_layer or nn.GELU
#         self.feature_embed = embed_layer(num_genes, embed_dim=embed_dim, ismask=ismask)
#
#     def forward_features(self, x, important, tau=0.001, epoch=300):
#
#         x, mask = self.feature_embed(x, tau=tau, important=important, epoch=epoch)
#         # print(x.shape)
#
#         return self.m(x), mask
#
#     def forward(self, x, important=True, tau=0.001, epoch=300):
#         # if self.training==True:
#         #     noise = torch.randn_like(x)
#         #     x = x + noise
#         latent, mask = self.forward_features(x, important, tau, epoch)
#
#         pre = latent
#         # print('head', self.head)
#         # print('pre', pre.shape)
#         return latent, pre, mask

class Transformer(nn.Module):
    def __init__(self, num_classes, num_genes, tau=0.2, ismask=False, is_all_transformer=False, is_all_mlp=False,
                 fe_bias=True,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bias=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer=Embedding, norm_layer=None,
                 act_layer=None, need_cal_weight=True):
        """
        Args:
            num_classes (int): number of classes for classification head
            num_genes (int): number of feature of input(expData)
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            qk_scale (float): override default qk scale of head_dim ** -0.5 if set
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_ratio (float): dropout rate
            attn_drop_ratio (float): attention dropout rate
            drop_path_ratio (float): stochastic depth rate
            embed_layer (nn.Module): feature embed layer
            norm_layer: (nn.Module): normalization layer
        """
        super(Transformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU
        self.is_all_transformer = is_all_transformer
        self.is_all_mlp = is_all_mlp
        self.feature_embed = embed_layer(num_genes, embed_dim=embed_dim, ismask=ismask,
                                         is_all_transformer=is_all_transformer, is_all_mlp=is_all_mlp)
        self.tau = tau
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        self.need_cal_weight = need_cal_weight
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        # self.blocks = nn.Sequential(*[
        #    Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
        #          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
        #          norm_layer=norm_layer, act_layer=act_layer)
        #    for i in range(depth)
        # ])
        self.blocks = nn.ModuleList()
        for i in range(depth):
            layer = Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                          drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                          norm_layer=norm_layer, act_layer=act_layer)
            self.blocks.append(copy.deepcopy(layer))
        self.norm = norm_layer(embed_dim)

        if self.is_all_mlp == True:
            self.m = LinearUNet(num_genes, drop_ratio=drop_ratio)
        else:
            if representation_size and not distilled:
                self.has_logits = True
                self.num_features = representation_size
                self.pre_logits = nn.Sequential(OrderedDict([
                    ("fc", nn.Linear(embed_dim, representation_size)),
                    ("act", nn.Tanh())
                ]))
            else:
                self.has_logits = False
                self.pre_logits = nn.Identity()
        # Classifier head(s)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes > 0 else nn.Identity()
        # Weight init
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x, important, tau=0.001, epoch=300):
        x, mask = self.feature_embed(x, tau=tau, important=important, epoch=epoch)
        # print('x', x.shape)
        if self.is_all_mlp == True:
            return self.m(x), mask
        else:
            # print(x.shape)
            cls_token = self.cls_token.expand(x.shape[0], -1, -1)
            # print(cls_token.shape)
            if self.dist_token is None:  # ViT中就是None
                x = torch.cat((cls_token, x), dim=1)
            else:
                x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
            tem = x
            for i, layer_block in enumerate(self.blocks):
                if i < len(self.blocks) - 1:
                    tem, _ = layer_block(tem)
                else:
                    tem, weight = layer_block(tem)
            x = self.norm(tem)
            # print('x', x.shape)

            return self.pre_logits(x[:, 0]), mask

    def forward(self, x, important=True, tau=0.001, epoch=300):
        #
        latent, mask = self.forward_features(x, important, tau, epoch)

        if self.is_all_mlp == True:
            pre = latent
        else:
            pre = self.head(latent)
        # print('head', self.head)
        # print('pre', pre.shape)
        return latent, pre, mask


def _init_vit_weights(m):
    """
    ViT weight initialization
    :param m: module
    """
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.LayerNorm):
        nn.init.zeros_(m.bias)
        nn.init.ones_(m.weight)


def create_model(num_classes, num_genes, tau=0.2, ismask=False, is_all_transformer=False, is_all_mlp=False,
                 embed_dim=48, depth=2, num_heads=4, has_logits: bool = True, need_cal_weight=True):
    model = Transformer(num_classes=num_classes,
                        num_genes=num_genes,
                        embed_dim=embed_dim,
                        depth=depth,
                        num_heads=num_heads,
                        drop_ratio=0.5, attn_drop_ratio=0.5, drop_path_ratio=0.5,
                        # drop_ratio=0.0, attn_drop_ratio=0.0, drop_path_ratio=0.0,
                        representation_size=embed_dim if has_logits else None,
                        need_cal_weight=need_cal_weight,
                        tau=tau,
                        ismask=ismask,
                        is_all_transformer=is_all_transformer,
                        is_all_mlp=is_all_mlp,
                        )

    return model

