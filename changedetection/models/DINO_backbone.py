import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from peft import LoraConfig, get_peft_model

class Backbone_DINOv2(nn.Module):
    """
    针对 timm 1.0.22 优化的 DINOv2 Backbone
    使用 reshape=True 自动处理 Patch/CLS Token，杜绝尺寸报错
    """
    def __init__(
        self,
        use_lora: bool = True,
        r: int = 32, # 默认 32
        pretrained_path: str = "./pretrained_weight/dinov2_vitb14_pretrain.pth",
        version: str = "vit_base_patch14_dinov2",
    ):
        super().__init__()
        self.embed_dim = 768
        self.channel_first = True
        # 对应 VMamba-Tiny 的 4 层维度
        self.dims = [96, 192, 384, 768]

        print(f"[Backbone] init DINOv2 ({version}) - Timm 1.0.22 Compatible Mode")
        
        # 1. 创建模型 (启用动态分辨率)
        self.dino = timm.create_model(
            version,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True, 
        )

        # 2. 加载本地权重 (如果存在)
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"[Backbone] load local ckpt: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            # 清理可能的权重前缀
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = self.dino.load_state_dict(state_dict, strict=False)
            print(f"[Backbone] load_state_dict: {msg}")
        else:
            print(f"[Backbone] Using random init or auto-downloading if available.")

        # 3. 冻结参数
        for p in self.dino.parameters():
            p.requires_grad = False

        # 4. 注入 LoRA
        if use_lora:
            print(f"[Backbone] inject LoRA (r={r}) on qkv")
            cfg = LoraConfig(
                r=r,
                lora_alpha=r * 2,
                target_modules=["qkv"],
                lora_dropout=0.1,
                bias="none",
            )
            self.dino = get_peft_model(self.dino, cfg)
            # 重新标记属性，防止被 PeftModel 覆盖
            self.channel_first = True

        # 5. 定义适配器 (Adapters)
        # 将 DINO 的 768 维 映射到 [96, 192, 384, 768]
        self.adapter1 = nn.Sequential(nn.Conv2d(768, 96, 1), nn.BatchNorm2d(96), nn.ReLU())
        self.adapter2 = nn.Sequential(nn.Conv2d(768, 192, 1), nn.BatchNorm2d(192), nn.ReLU())
        self.adapter3 = nn.Sequential(nn.Conv2d(768, 384, 1), nn.BatchNorm2d(384), nn.ReLU())
        self.adapter4 = nn.Sequential(nn.Conv2d(768, 768, 1), nn.BatchNorm2d(768), nn.ReLU())

    def forward(self, x):
        B, C, H, W = x.shape

        # --- 关键步骤 1: Padding ---
        # 即使 timm 支持动态尺寸，手动 pad 到 14 的倍数能避免边缘特征对齐问题
        patch = 14
        pad_h = (patch - H % patch) % patch
        pad_w = (patch - W % patch) % patch
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # --- 关键步骤 2: 提取多层特征 (Timm 1.x 原生写法) ---
        # n=12: 获取所有 12 层的输出
        # reshape=True: 让 timm 自动把 [B, N, C] 变成 [B, C, H, W]，自动剔除 CLS Token
        # 这一步绝对不会报错，因为它利用了 timm 内部对 grid_size 的计算
        all_feats = self.dino.get_intermediate_layers(x, n=12, reshape=True)

        # --- 关键步骤 3: 挑选 4 层 (浅->深) ---
        # 索引 1, 4, 7, 11 对应物理层数 2, 5, 8, 12
        f0 = all_feats[1]  
        f1 = all_feats[4]
        f2 = all_feats[7]
        f3 = all_feats[11]

        # --- 关键步骤 4: 构建金字塔 (Feature Pyramid) ---
        # 目标: 将不同尺度的 DINO 特征调整为 VMamba 解码器需要的 H/4, H/8, H/16, H/32
        
        # Stage 1: H/4
        out1 = self.adapter1(f0)
        out1 = F.interpolate(out1, size=(H // 4, W // 4), mode="bilinear", align_corners=False)

        # Stage 2: H/8
        out2 = self.adapter2(f1)
        out2 = F.interpolate(out2, size=(H // 8, W // 8), mode="bilinear", align_corners=False)

        # Stage 3: H/16
        out3 = self.adapter3(f2)
        out3 = F.interpolate(out3, size=(H // 16, W // 16), mode="bilinear", align_corners=False)

        # Stage 4: H/32
        out4 = self.adapter4(f3)
        out4 = F.interpolate(out4, size=(H // 32, W // 32), mode="bilinear", align_corners=False)

        return (out1, out2, out3, out4)