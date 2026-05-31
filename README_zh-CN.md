# DINOMamba / FMB-CD

基于基础模型桥接的遥感变化检测方法：通过语义引导细节注入和跨流特征聚合实现高效变化检测。

本仓库是 FMB-CD 小论文对应的二值遥感变化检测实现。代码基于 ChangeMamba 改造，将原本需要全量微调的 VMamba 编码器替换为参数高效的 DINOv2 双流编码器。

## 方法简介

遥感变化检测同时需要稳定的高层语义和清晰的空间细节。全量微调大模型效果较强，但训练成本高；冻结基础模型可以降低训练成本，但 ViT 的 patch 特征容易造成边界模糊；直接加入 CNN 细节分支又可能放大背景噪声。

FMB-CD 的核心设计包括：

- **冻结 DINOv2 语义流**：提供稳健的高层语义先验，并通过 LoRA 做轻量适配。
- **可训练 CNN 细节流**：补充建筑边界、小目标和高频纹理细节。
- **语义引导细节注入 SGDI**：在融合前抑制与变化任务无关的细节噪声。
- **跨流特征聚合 CSFA**：对齐并重标定 ViT/CNN 异构特征。
- **Mamba 解码器**：继承 ChangeMamba 的多尺度特征聚合能力。

当前二值变化检测代码主要位于：

- `changedetection/models/ChangeMambaBCD.py`
- `changedetection/models/DINO_backbone.py`
- `changedetection/models/ChangeDecoder.py`

SCD/BDA 相关脚本和 VMamba 模块仍保留自原 ChangeMamba 工程，主要用于参考；本 README 重点说明 FMB-CD 的二值变化检测流程。

## 论文实验结果

以下结果来自当前 FMB-CD 小论文稿件。

| 数据集 | OA | F1 | IoU | Recall | Precision |
|---|---:|---:|---:|---:|---:|
| LEVIR-CD+ | 98.90 | 85.90 | 75.29 | 83.63 | 88.30 |
| WHU-CD | 99.49 | 92.76 | 86.50 | 92.14 | 93.38 |
| SYSU-CD | 92.85 | 84.32 | 72.89 | 81.55 | 87.29 |

效率对比：

| 方法 | 可训练参数量 | 训练迭代数 |
|---|---:|---:|
| ChangeMamba-Base | 84.7M | 240k |
| FMB-CD | 17.55M | 40k |

说明：FMB-CD 包含冻结的 DINOv2 主干，因此这里报告可训练参数量，用于更准确反映优化成本。

## 项目结构

```text
DINOMamba/
├── changedetection/
│   ├── configs/vssm1/              # Mamba 解码器配置
│   ├── datasets/                   # BCD/SCD/BDA 数据加载
│   ├── models/
│   │   ├── ChangeMambaBCD.py       # FMB-CD 二值变化检测入口
│   │   ├── DINO_backbone.py        # DINOv2 + LoRA + 细节分支
│   │   └── ChangeDecoder.py        # Mamba 解码器
│   └── script/
│       ├── train_MambaBCD.py       # 二值变化检测训练
│       └── infer_MambaBCD.py       # 二值变化检测推理
├── classification/                 # 继承自 ChangeMamba 的 VMamba 代码
├── kernels/selective_scan/         # CUDA selective scan 算子
├── pretrained_weight/              # DINOv2 权重和模型 checkpoint
├── data/                           # 本地数据或数据列表
└── requirements.txt
```

## 环境安装

selective scan CUDA 算子主要面向 Linux + CUDA 环境。Windows 可以用于代码编辑和轻量查看，正式训练建议在 CUDA Linux 环境中运行。

```bash
conda create -n dinomamba python=3.8 -y
conda activate dinomamba

# 先按自己的 CUDA 版本安装 PyTorch。
# 示例：
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

pip install -r requirements.txt
pip install peft "timm>=0.9.0"

cd kernels/selective_scan
pip install .
cd ../..
```

注意：

- 原 ChangeMamba 的依赖中固定了 `timm==0.4.12`。
- FMB-CD 使用 `timm.create_model("vit_base_patch14_dinov2")`，因此需要支持 DINOv2 的新版 `timm`。
- LoRA 注入需要安装 `peft`。

## 预训练权重

DINOv2 ViT-B/14 权重请放置到：

```text
pretrained_weight/dinov2_vitb14_pretrain.pth
```

当前代码会在 `changedetection/models/DINO_backbone.py` 中加载该路径。

旧的 ChangeMamba/VMamba checkpoint 不适用于当前 FMB-CD 编码器。推理时请使用当前模型重新训练得到的 checkpoint，并通过 `--resume` 加载。

## 数据准备

SYSU-CD、LEVIR-CD+、WHU-CD 等二值变化检测数据集请组织为：

```text
DATASET_ROOT/
├── train/
│   ├── T1/
│   │   ├── 00001.png
│   │   └── ...
│   ├── T2/
│   │   ├── 00001.png
│   │   └── ...
│   └── GT/
│       ├── 00001.png
│       └── ...
├── test/
│   ├── T1/
│   ├── T2/
│   └── GT/
├── train_list.txt
└── test_list.txt
```

GT 标签中，未变化像素为 `0`，变化像素为 `255`。

## 训练

请在项目根目录运行训练命令。当前 DINOv2 权重路径是相对于项目根目录写的，因此不要先 `cd changedetection` 再运行。

LEVIR-CD+ 示例：

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

SYSU-CD 示例：

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

训练过程会保存：

- `best_model.pth`
- `last_model.pth`
- 若开启可视化，会在模型目录下保存误检/漏检和特征可视化结果。

## 推理

推理时请使用 `--resume` 加载 FMB-CD 训练好的 checkpoint。

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

预测结果会保存到：

```text
results/<dataset>/<model_type>/change_map/
```

## 消融实验设置

论文采用了简洁的三步消融：

| 变体 | Recall | Precision | OA | F1 | IoU |
|---|---:|---:|---:|---:|---:|
| Baseline (Frozen DINO) | 76.60 | 88.85 | 98.65 | 82.27 | 69.88 |
| + Detail Stream | 85.31 | 87.67 | 98.85 | 85.54 | 74.73 |
| + SGDI & CSFA (Full) | 83.63 | 88.30 | 98.90 | 85.90 | 75.29 |

对应分析逻辑：

- 仅使用冻结 DINOv2 时，语义较稳健，但空间细节不足，Recall 偏低。
- 加入细节分支后，Recall 明显提升，说明边界和小目标细节得到补充。
- 加入交互模块后，Precision 回升，说明语义门控和跨流聚合有助于抑制伪变化。

## 常见问题

| 问题 | 解决方式 |
|---|---|
| `vit_base_patch14_dinov2` 无法识别 | 安装支持 DINOv2 的新版 `timm`。 |
| `ModuleNotFoundError: peft` | 安装 `peft`。 |
| `selective_scan_cuda_oflex` 找不到 | 在当前 CUDA/PyTorch 环境下重新编译 `kernels/selective_scan`。 |
| 推理精度很低 | 使用 `--resume` 加载 FMB-CD 训练好的 checkpoint，不要把 ImageNet/VMamba 权重当作变化检测模型权重。 |
| CUDA 显存不足 | 降低 `--batch_size`、裁剪尺寸，或关闭额外可视化。 |
| 找不到 DINOv2 权重 | 将 `dinov2_vitb14_pretrain.pth` 放到 `pretrained_weight/` 下，或修改 `DINO_backbone.py` 中的路径。 |

## 引用

论文正式发表后可更新 BibTeX。当前可暂写为：

```bibtex
@misc{yu2026fmbcd,
  title  = {FMB-CD: Foundation Model Bridged Change Detection via Semantic-Guided Detail Injection and Cross-Stream Feature Aggregation},
  author = {Yue Yu and Maoteng Zheng},
  year   = {2026},
  note   = {Manuscript in preparation}
}
```

本项目基于 ChangeMamba 改造：

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

## 致谢

本仓库基于 ChangeMamba、VMamba、timm、PEFT/LoRA 和 DINOv2 等优秀开源项目构建。使用时请同时遵守原项目和数据集的许可证要求。
