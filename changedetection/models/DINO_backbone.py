import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from peft import LoraConfig, get_peft_model

# ==========================================
# [新增模块] 空间适配器 (Run 3 核心修改)
# 作用：解决 DINOv2 直接上采样产生的锯齿和棋盘伪影
# ==========================================
class SpatialAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        # 1. 降维 (Channel Mapping)
        self.project = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        # 2. 空间重构 (Spatial Reconstruction)
        # 使用 3x3 卷积 + Padding=1，利用邻域信息平滑上采样后的特征
        self.smooth = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target_size):
        # Step 1: 降维 (B, 768, h, w) -> (B, 96, h, w)
        x = self.project(x)
        
        # Step 2: 上采样/下采样 (B, 96, h, w) -> (B, 96, H_target, W_target)
        # 这里的 interpolate 只是粗略的拉伸
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        
        # Step 3: 平滑去噪 (Run 3 关键点)
        # 通过 3x3 卷积消除上采样带来的锯齿，填补空间细节
        x = self.smooth(x)
        return x

# ==========================================
# 主 Backbone 类
# ==========================================
class Backbone_DINOv2(nn.Module):
    def __init__(
        self,
        use_lora: bool = True,
        r: int = 32,
        pretrained_path: str = "./pretrained_weight/dinov2_vitb14_pretrain.pth",
        version: str = "vit_base_patch14_dinov2",
    ):
        super().__init__()
        self.embed_dim = 768
        self.channel_first = True
        self.dims = [96, 192, 384, 768]

        print(f"[Backbone] init DINOv2 ({version}) - Run 3: Spatial Adapter Mode")
        
        # 1. 创建模型
        self.dino = timm.create_model(
            version,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True, 
        )

        # 2. 加载权重 (保持不变)
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"[Backbone] load local ckpt: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            self.dino.load_state_dict(state_dict, strict=False)
        else:
            print(f"[Backbone] Using random init or auto-downloading.")

        # 3. 冻结参数 (保持不变)
        for p in self.dino.parameters():
            p.requires_grad = False

        # 4. 注入 LoRA (保持不变)
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
            self.channel_first = True

        # ==========================================
        # [修改点 1] 初始化适配器
        # 不再使用简单的 nn.Sequential，而是使用自定义的 SpatialAdapter
        # ==========================================
        self.adapter1 = SpatialAdapter(768, 96)  # Output: H/4
        self.adapter2 = SpatialAdapter(768, 192) # Output: H/8
        self.adapter3 = SpatialAdapter(768, 384) # Output: H/16
        self.adapter4 = SpatialAdapter(768, 768) # Output: H/32

    def forward(self, x):
        B, C, H, W = x.shape

        # 关键步骤 1: Padding (保持不变)
        patch = 14
        pad_h = (patch - H % patch) % patch
        pad_w = (patch - W % patch) % patch
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        # 关键步骤 2: 提取特征 (保持不变)
        all_feats = self.dino.get_intermediate_layers(x, n=12, reshape=True)

        # 关键步骤 3: 挑选 4 层 (保持不变)
        selected_indices = [1, 4, 7, 11]
        raw_feats = [all_feats[i] for i in selected_indices]
        # BHWC -> BCHW
        # 4. 维度自动纠正 
        # 不管 timm 返回的是 BHWC 还是 BCHW，这里都会统一成 BCHW
        processed_feats = []
        for feat in raw_feats:
            if feat.shape[-1] == self.embed_dim and feat.ndim == 4:
                # 如果最后一位是768，说明是 BHWC，需要转
                feat = feat.permute(0, 3, 1, 2).contiguous()
            processed_feats.append(feat)
        
        f0, f1, f2, f3 = processed_feats
        # ==========================================
        # [修改点 2] 前向传播
        # 将 target_size 传给适配器，让适配器内部处理 "插值+卷积"
        # ==========================================
        
        # Stage 1: H/4
        # 此时 f0 尺寸约为 H/14，需要上采样到 H/4，并用 3x3 卷积修复细节
        out1 = self.adapter1(f0, target_size=(H // 4, W // 4))

        # Stage 2: H/8
        out2 = self.adapter2(f1, target_size=(H // 8, W // 8))

        # Stage 3: H/16
        out3 = self.adapter3(f2, target_size=(H // 16, W // 16))

        # Stage 4: H/32
        out4 = self.adapter4(f3, target_size=(H // 32, W // 32))

        return (out1, out2, out3, out4)