import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from peft import LoraConfig, get_peft_model

# ==========================================
# 1. 定义轻量级 CNN 旁路 (保持不变)
# ==========================================
class CNN_SideBranch(nn.Module):
    def __init__(self):
        super().__init__()
        # Stage 1: -> H/4 (96 channels)
        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),   # H/2
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 96, 3, stride=2, padding=1),  # H/4
            nn.BatchNorm2d(96), nn.ReLU()
        )
        # Stage 2: -> H/8 (192 channels)
        self.stage2 = nn.Sequential(
            nn.Conv2d(96, 192, 3, stride=2, padding=1), # H/8
            nn.BatchNorm2d(192), nn.ReLU()
        )
        # Stage 3: -> H/16 (384 channels)
        self.stage3 = nn.Sequential(
            nn.Conv2d(192, 384, 3, stride=2, padding=1), # H/16
            nn.BatchNorm2d(384), nn.ReLU()
        )

    def forward(self, x):
        s1 = self.stage1(x) # [B, 96, H/4, W/4]
        s2 = self.stage2(s1) # [B, 192, H/8, W/8]
        s3 = self.stage3(s2) # [B, 384, H/16, W/16]
        return s1, s2, s3

# ==========================================
# 2. 空间适配器 (Run 3 的改进，保持不变)
# ==========================================
class SpatialAdapter(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.project = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )
        self.smooth = nn.Sequential(
            nn.Conv2d(out_dim, out_dim, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x, target_size):
        x = self.project(x)
        x = F.interpolate(x, size=target_size, mode="bilinear", align_corners=False)
        x = self.smooth(x)
        return x

# ==========================================
# 3. 主 Backbone (Run 4: Fusion Mode)
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
        self.dims = [96, 192, 384, 768] # 对应 Decoder 需要的维度

        print(f"[Backbone] init DINOv2 ({version}) - Run 4: Side-Tuning Fusion Mode")
        
        # --- DINO 初始化 ---
        self.dino = timm.create_model(
            version,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True, 
        )

        # 加载权重
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"[Backbone] load local ckpt: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            self.dino.load_state_dict(state_dict, strict=False)
        else:
            print(f"[Backbone] Using random init or auto-downloading.")

        # 冻结 DINO 参数
        for p in self.dino.parameters():
            p.requires_grad = False

        # 注入 LoRA
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

        # --- Adapter 初始化 ---
        self.adapter1 = SpatialAdapter(768, 96)  # Output: H/4
        self.adapter2 = SpatialAdapter(768, 192) # Output: H/8
        self.adapter3 = SpatialAdapter(768, 384) # Output: H/16
        self.adapter4 = SpatialAdapter(768, 768) # Output: H/32

        # --- [关键修改] Side Branch 初始化 ---
        # 这是一个全新的可训练模块，不冻结
        print("[Backbone] Initializing CNN Side-Branch...")
        self.side_branch = CNN_SideBranch()

    def forward(self, x):
        B, C, H, W = x.shape

        # ===========================
        # Part 1: DINO 路径 (语义流)
        # ===========================
        
        # 1. Padding (为了适配 ViT patch)
        patch = 14
        pad_h = (patch - H % patch) % patch
        pad_w = (patch - W % patch) % patch
        # 记录 padding 后的 input，用于喂给 DINO
        x_pad = x
        if pad_h > 0 or pad_w > 0:
            x_pad = F.pad(x, (0, pad_w, 0, pad_h))

        # 2. DINO 特征提取
        all_feats = self.dino.get_intermediate_layers(x_pad, n=12, reshape=True)

        # 3. 筛选层级 & 维度校正
        selected_indices = [1, 4, 7, 11]
        raw_feats = [all_feats[i] for i in selected_indices]
        
        processed_feats = []
        for feat in raw_feats:
            # 兼容 timm 可能返回的 BHWC
            if feat.shape[-1] == self.embed_dim and feat.ndim == 4:
                feat = feat.permute(0, 3, 1, 2).contiguous()
            processed_feats.append(feat)
        
        f0, f1, f2, f3 = processed_feats

        # 4. 通过 Adapter 调整到目标尺寸 (H/4, H/8, etc.)
        # 注意：这里传入 target_size 使用原始 H, W，确保去除 padding 的影响
        d1 = self.adapter1(f0, target_size=(H // 4, W // 4))
        d2 = self.adapter2(f1, target_size=(H // 8, W // 8))
        d3 = self.adapter3(f2, target_size=(H // 16, W // 16))
        d4 = self.adapter4(f3, target_size=(H // 32, W // 32))

        # ===========================
        # Part 2: Side Branch 路径 (细节流)
        # ===========================
        
        # 直接喂原始图片 x (不需要 padding，卷积能处理任意尺寸)
        # s1: H/4, s2: H/8, s3: H/16
        s1, s2, s3 = self.side_branch(x)

        # ===========================
        # Part 3: 特征融合 (Fusion)
        # ===========================
        
        # [核心逻辑] DINO(语义) + SideBranch(纹理)
        # 只要输入尺寸是 32 的倍数 (如 256, 512)，这里尺寸会完美对齐
        out1 = d1 + s1
        out2 = d2 + s2
        out3 = d3 + s3
        out4 = d4  # 最深层不需要高频细节，只保留 DINO 特征

        return (out1, out2, out3, out4)