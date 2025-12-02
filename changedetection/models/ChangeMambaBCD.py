import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict

# 引入必要的模块
from changedetection.models.ChangeDecoder import ChangeDecoder
from classification.models.vmamba import LayerNorm2d

# 引入你的新骨干
from changedetection.models.DINO_backbone import Backbone_DINOv2

class ChangeMambaBCD(nn.Module):
    def __init__(self, pretrained, **kwargs):
        super(ChangeMambaBCD, self).__init__()
        
        # -----------------------------------------------------------
        # [修改点 1] 实例化 DINOv2 多尺度骨干
        # -----------------------------------------------------------
        self.encoder = Backbone_DINOv2(
            version='vit_base_patch14_dinov2', # 使用 Base 版本
            use_lora=True,
            r=32,                              # [策略升级] 改为 32 以提升 Recall
            pretrained_path='./pretrained_weight/dinov2_vitb14_pretrain.pth' 
            # 如果本地没文件，把上面改成 pretrained_path=None，它会自动下载
        )
        
        # [修改点 2] 强制指定维度，对齐 VMamba-Tiny 的配置
        # 确保 Decoder 知道输入的 4 层特征分别是多少通道
        self.encoder.dims = [96, 192, 384, 768] 

        # -----------------------------------------------------------
        # 下面是 ChangeMamba 原有的解码器配置逻辑 (保持不变)
        # -----------------------------------------------------------
        _NORMLAYERS = dict(
            ln=nn.LayerNorm,
            ln2d=LayerNorm2d,
            bn=nn.BatchNorm2d,
        )
        
        _ACTLAYERS = dict(
            silu=nn.SiLU, 
            gelu=nn.GELU, 
            relu=nn.ReLU, 
            sigmoid=nn.Sigmoid,
        )
 
        # 根据 Encoder 的 channel_first 属性决定用什么归一化层
        # DINO_backbone.py 里我们已经设置了 self.channel_first = True
        norm_layer = LayerNorm2d if getattr(self.encoder, 'channel_first', False) else _NORMLAYERS.get(kwargs['norm_layer'].lower(), None)        
        ssm_act_layer: nn.Module = _ACTLAYERS.get(kwargs['ssm_act_layer'].lower(), None)
        mlp_act_layer: nn.Module = _ACTLAYERS.get(kwargs['mlp_act_layer'].lower(), None)

        # 移除显式传递的参数，防止冲突，剩下的 kwargs (如 ssm_ratio 等) 传给 Decoder
        clean_kwargs = {k: v for k, v in kwargs.items() if k not in ['norm_layer', 'ssm_act_layer', 'mlp_act_layer']}
        
        # 初始化 Mamba 解码器
        self.decoder = ChangeDecoder(
            encoder_dims=self.encoder.dims, # 使用我们刚才强制指定的维度
            channel_first=getattr(self.encoder, 'channel_first', False),
            norm_layer=norm_layer,
            ssm_act_layer=ssm_act_layer,
            mlp_act_layer=mlp_act_layer,
            **clean_kwargs
        )

        self.main_clf = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=1)

    def _upsample_add(self, x, y):
        _, _, H, W = y.size()
        return F.interpolate(x, size=(H, W), mode='bilinear') + y

    def forward(self, pre_data, post_data):
        # 1. 编码器提取特征 (现在是 DINOv2 输出 4 层特征)
        pre_features = self.encoder(pre_data)
        post_features = self.encoder(post_data)

        # 2. 解码器融合特征 (Mamba 机制)
        output = self.decoder(pre_features, post_features)

        # 3. 分类头输出
        output = self.main_clf(output)
        
        # 4. 上采样回原图尺寸
        output = F.interpolate(output, size=pre_data.size()[-2:], mode='bilinear')
        return output