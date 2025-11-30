import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from peft import LoraConfig, get_peft_model


class Backbone_DINOv2(nn.Module):
    """
    DINOv2 ViT-B/14 backbone with LoRA，输出 BCHW（channel_first=True）
    """
    def __init__(
        self,
        use_lora: bool = True,
        r: int = 16,
        pretrained_path: str = "./pretrained_weight/dinov2_vitb14_pretrain.pth",
        version: str = "vit_base_patch14_dinov2",
    ):
        super().__init__()
        self.embed_dim = 768
        self.channel_first = True          # 告诉解码器用 BCHW
        self.dims = [96, 192, 384, 768]

        print(f"[Backbone] init DINOv2 ({version})")
        self.dino = timm.create_model(
            version,
            pretrained=False,
            num_classes=0,
            dynamic_img_size=True,
        )

        if os.path.exists(pretrained_path):
            print(f"[Backbone] load local ckpt: {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}
            msg = self.dino.load_state_dict(state_dict, strict=False)
            print(f"[Backbone] load_state_dict: {msg}")
        else:
            print(f"[Backbone] WARNING: ckpt {pretrained_path} not found, using random init.")

        for p in self.dino.parameters():
            p.requires_grad = False

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
            # 让属性不被 LoRA 包裹覆盖
            self.channel_first = True

        self.adapter1 = nn.Conv2d(self.embed_dim, 96, 1)
        self.adapter2 = nn.Conv2d(self.embed_dim, 192, 1)
        self.adapter3 = nn.Conv2d(self.embed_dim, 384, 1)
        self.adapter4 = nn.Conv2d(self.embed_dim, 768, 1)

    def forward(self, x):
        B, C, H, W = x.shape

        # pad 到 14 的倍数
        patch = 14
        pad_h = (patch - H % patch) % patch
        pad_w = (patch - W % patch) % patch
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))

        outputs = self.dino.forward_features(x)
        x_tokens = outputs.get("x_norm_patchtokens", outputs.get("x")) if isinstance(outputs, dict) else outputs
        if hasattr(self.dino, "num_prefix_tokens") and self.dino.num_prefix_tokens > 0:
            x_tokens = x_tokens[:, self.dino.num_prefix_tokens :]

        h_dino = (H + pad_h) // patch
        w_dino = (W + pad_w) // patch
        feat = x_tokens.transpose(1, 2).reshape(B, self.embed_dim, h_dino, w_dino)

        out1 = self.adapter1(F.interpolate(feat, size=(H // 4, W // 4), mode="bilinear", align_corners=False))
        out2 = self.adapter2(F.interpolate(feat, size=(H // 8, W // 8), mode="bilinear", align_corners=False))
        out3 = self.adapter3(F.interpolate(feat, size=(H // 16, W // 16), mode="bilinear", align_corners=False))
        out4 = self.adapter4(F.interpolate(feat, size=(H // 32, W // 32), mode="bilinear", align_corners=False))

        # 保持 BCHW，解码器 concat 预期通道 2*768=1536
        return (out1, out2, out3, out4)
