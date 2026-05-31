# DINOMamba / FMB-CD

Foundation Model Bridged Change Detection via Semantic-Guided Detail Injection and Cross-Stream Feature Aggregation.

This repository contains the binary remote-sensing change detection implementation used for our FMB-CD paper draft. The code is adapted from ChangeMamba and replaces the fully fine-tuned VMamba encoder with a parameter-efficient DINOv2-based dual-stream encoder.

## Overview

Remote-sensing change detection needs both high-level semantic robustness and fine spatial details. Full fine-tuning of a large backbone can be accurate, but it is expensive. Freezing a foundation model is efficient, but patch-based ViT features may blur boundaries. A CNN detail branch can recover high-frequency cues, but naive fusion may amplify background noise.

FMB-CD addresses this trade-off with:

- **Frozen DINOv2 semantic stream** with LoRA adapters for lightweight domain adaptation.
- **Trainable CNN detail stream** for local boundaries and high-frequency spatial cues.
- **Semantic-guided detail injection (SGDI)** to suppress task-irrelevant detail responses before fusion.
- **Cross-stream feature aggregation (CSFA)** to align and recalibrate heterogeneous ViT/CNN features.
- **Mamba-based decoder** inherited from ChangeMamba for multi-scale feature aggregation.

In the current code, the binary change detection path is implemented mainly in:

- `changedetection/models/ChangeMambaBCD.py`
- `changedetection/models/DINO_backbone.py`
- `changedetection/models/ChangeDecoder.py`

The SCD/BDA scripts and VMamba-related modules are retained from the original ChangeMamba codebase for reference, but the README below focuses on the FMB-CD binary change detection setup.

## Paper Results

The following numbers are from the current FMB-CD manuscript.

| Dataset | OA | F1 | IoU | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| LEVIR-CD+ | 98.90 | 85.90 | 75.29 | 83.63 | 88.30 |
| WHU-CD | 99.49 | 92.76 | 86.50 | 92.14 | 93.38 |
| SYSU-CD | 92.85 | 84.32 | 72.89 | 81.55 | 87.29 |

Efficiency comparison:

| Method | Trainable Params | Training Iters |
|---|---:|---:|
| ChangeMamba-Base | 84.7M | 240k |
| FMB-CD | 17.55M | 40k |

Note: FMB-CD contains a frozen DINOv2 backbone. Trainable parameters are reported because they better reflect the optimization cost.

## Repository Layout

```text
DINOMamba/
+-- changedetection/
|   +-- configs/vssm1/              # Mamba decoder configs
|   +-- datasets/                   # BCD/SCD/BDA dataloaders
|   +-- models/
|   |   +-- ChangeMambaBCD.py       # FMB-CD BCD model entry
|   |   +-- DINO_backbone.py        # DINOv2 + LoRA + detail branch
|   |   +-- ChangeDecoder.py        # Mamba decoder
|   +-- script/
|       +-- train_MambaBCD.py       # BCD training
|       +-- infer_MambaBCD.py       # BCD inference
+-- classification/                 # VMamba backbone code inherited from ChangeMamba
+-- kernels/selective_scan/         # CUDA selective scan kernels
+-- pretrained_weight/              # DINOv2/checkpoints
+-- data/                           # optional local dataset lists/data
+-- requirements.txt
```

## Installation

The selective scan kernel is designed for Linux + CUDA. Windows can be used for editing and lightweight inspection, but training is expected to run on a CUDA-enabled Linux environment.

```bash
conda create -n dinomamba python=3.8 -y
conda activate dinomamba

# Install PyTorch according to your CUDA version first.
# Example only:
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install peft "timm>=0.9.0"

cd kernels/selective_scan
pip install .
cd ../..
```

Important dependency note:

- The original ChangeMamba dependency list pins `timm==0.4.12`.
- FMB-CD uses `timm.create_model("vit_base_patch14_dinov2")`, so a DINOv2-capable `timm` version is required.
- `peft` is required for LoRA injection.

## Pretrained Weights

Place the DINOv2 ViT-B/14 checkpoint at:

```text
pretrained_weight/dinov2_vitb14_pretrain.pth
```

The current implementation loads this path in `changedetection/models/DINO_backbone.py`.

The old ChangeMamba VMamba checkpoints are not used by the FMB-CD BCD encoder. Use `--resume` only for checkpoints trained with this DINOMamba/FMB-CD model.

## Data Preparation

For binary change detection, organize SYSU-CD, LEVIR-CD+, or WHU-CD as:

```text
DATASET_ROOT/
+-- train/
|   +-- T1/
|   |   +-- 00001.png
|   |   +-- ...
|   +-- T2/
|   |   +-- 00001.png
|   |   +-- ...
|   +-- GT/
|       +-- 00001.png
|       +-- ...
+-- test/
|   +-- T1/
|   +-- T2/
|   +-- GT/
+-- train_list.txt
+-- test_list.txt
```

The GT mask should use `0` for unchanged pixels and `255` for changed pixels.

## Training

Run from the project root. This is important because the DINOv2 checkpoint path is currently relative to the project root.

Example: train FMB-CD on LEVIR-CD+ for 40k iterations.

```bash
python changedetection/script/train_MambaBCD.py \
  --dataset LEVIR-CD+ \
  --batch_size 16 \
  --crop_size 256 \
  --max_iters 40000 \
  --val_interval 1000 \
  --model_type FMB-CD \
  --model_param_path changedetection/saved_models \
  --train_dataset_path data/LEVIR-CD+/train \
  --train_data_list_path data/LEVIR-CD+/train_list.txt \
  --test_dataset_path data/LEVIR-CD+/test \
  --test_data_list_path data/LEVIR-CD+/test_list.txt \
  --cfg changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml
```

Example for SYSU-CD:

```bash
python changedetection/script/train_MambaBCD.py \
  --dataset SYSU \
  --batch_size 16 \
  --crop_size 256 \
  --max_iters 40000 \
  --val_interval 1000 \
  --model_type FMB-CD \
  --model_param_path changedetection/saved_models \
  --train_dataset_path /path/to/SYSU/train \
  --train_data_list_path /path/to/SYSU/train_list.txt \
  --test_dataset_path /path/to/SYSU/test \
  --test_data_list_path /path/to/SYSU/test_list.txt \
  --cfg changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml
```

The training script saves:

- `best_model.pth`
- `last_model.pth`
- optional visualization outputs under the model save directory

## Inference

Use `--resume` to load an FMB-CD checkpoint.

```bash
python changedetection/script/infer_MambaBCD.py \
  --dataset LEVIR-CD+ \
  --model_type FMB-CD \
  --test_dataset_path data/LEVIR-CD+/test \
  --test_data_list_path data/LEVIR-CD+/test_list.txt \
  --cfg changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml \
  --resume changedetection/saved_models/LEVIR-CD+/FMB-CD_xxx/best_model.pth \
  --result_saved_path results
```

Predicted binary change maps will be saved to:

```text
results/<dataset>/<model_type>/change_map/
```

## Ablation Protocol

The paper uses the following compact ablation path:

| Variant | Recall | Precision | OA | F1 | IoU |
|---|---:|---:|---:|---:|---:|
| Baseline (Frozen DINO) | 76.60 | 88.85 | 98.65 | 82.27 | 69.88 |
| + Detail Stream | 85.31 | 87.67 | 98.85 | 85.54 | 74.73 |
| + SGDI & CSFA (Full) | 83.63 | 88.30 | 98.90 | 85.90 | 75.29 |

The intended reading is:

- Frozen DINOv2 alone is semantically robust but misses fine details.
- Adding the detail branch recovers recall.
- The interaction design suppresses false alarms and improves precision.

## Common Issues

| Issue | Fix |
|---|---|
| `vit_base_patch14_dinov2` is unknown | Install a newer DINOv2-capable `timm` version. |
| `ModuleNotFoundError: peft` | Install `peft`. |
| `selective_scan_cuda_oflex` is not found | Rebuild `kernels/selective_scan` with the active CUDA/PyTorch environment. |
| Low inference accuracy | Use `--resume` with a trained FMB-CD checkpoint. Do not load old VMamba/ImageNet weights as a trained CD model. |
| CUDA out of memory | Reduce `--batch_size`, crop size, or disable extra visualization. |
| DINOv2 checkpoint not found | Put `dinov2_vitb14_pretrain.pth` under `pretrained_weight/` or update the path in `DINO_backbone.py`. |

## Citation

The manuscript BibTeX will be added after publication. For now, please cite the repository as:

```bibtex
@misc{yu2026fmbcd,
  title  = {FMB-CD: Foundation Model Bridged Change Detection via Semantic-Guided Detail Injection and Cross-Stream Feature Aggregation},
  author = {Yue Yu and Maoteng Zheng},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

This codebase is adapted from ChangeMamba:

```bibtex
@article{chen2024changemamba,
  author  = {Hongruixuan Chen and Jian Song and Chengxi Han and Junshi Xia and Naoto Yokoya},
  journal = {IEEE Transactions on Geoscience and Remote Sensing},
  title   = {ChangeMamba: Remote Sensing Change Detection with Spatiotemporal State Space Model},
  year    = {2024},
  volume  = {62},
  pages   = {1-20},
  doi     = {10.1109/TGRS.2024.3417253}
}
```

## Acknowledgments

This repository builds on the excellent open-source implementations of ChangeMamba, VMamba, timm, PEFT/LoRA, and DINOv2. Please also follow the licenses of the original projects and datasets.
