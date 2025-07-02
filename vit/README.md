# Tutorial: ViT for Transformer-Based Mask R-CNN

This tutorial guides you through reproducing the experimental results of Transformer-based Mask R-CNN models using our ViT pipeline.

If you encounter any issues, please report them in our [GitHub Issues](https://github.com/DFQ-Dojo/dfq-toolkit/issues) section.

---

## Prerequisites

### Hardware Requirements

- At least **one NVIDIA GPU**
- Minimum **32 GiB RAM** (recommended)

### Software Requirements

We use [uv](https://docs.astral.sh/uv/) to manage the Python environment.

> 📥 Install `uv` by following [this guide](https://docs.astral.sh/uv/getting-started/installation/).

Once `uv` is installed, navigate to the `dfq-toolkit/vit` directory and run:

```bash
uv sync
```

This command installs all required dependencies and sets up a virtual environment. To activate it:

```bash
source .venv/bin/activate
```

> 🔧 Ensure your working directory is always `dfq-toolkit/vit` for the following steps.

---

### Known Issues

#### 🔄 Dynamic Linking Library Error

If you encounter:

```bash
Error: no module named mmcv._ext
```

Check if `_ext.cpython-39-x86_64-linux-gnu.so` exists in the `mmcv` installation directory. If it's missing, download a precompiled version from [this link](https://drive.google.com/file/d/1Dz0X6_Qe3whrAgmAAYn1auW57tY8PRXk/view?usp=drive_link).

This issue stems from compatibility problems in the `mmcv` library.

#### ⚠️ Virtual Environment Warning

You may see:

```bash
warning: `VIRTUAL_ENV=/foo/bar/.venv` does not match the project environment path `.venv`
```

This is harmless and can be safely ignored.

---

## Dataset and Model Setup

1. Download the [COCO2017](https://cocodataset.org/) dataset.
2. Place it under: `./data/coco/`

3. Then run: `python -m tools.download_models`

4. Your directory structure should look like:

```bash
    dfq-toolkit/
      ├── vit/
      │   ├── data/
      │   │   └── coco/
      │   │       ├── annotations/
      │   │       ├── labels/
      |   |       ├── val2017/
      │   │       └── train2017/
      |   |            └── 00000001.jpg...
      │   └── pretrained/
      │       └── *.pth
      └── yolov5/
```

> 💡 You can use symbolic links to avoid duplicating large datasets.

---

## Command-Line Interface via Pydantic

We use [Pydantic](https://docs.pydantic.dev/latest/) for CLI parsing.

### Formatting Guidelines

- Scalar values: `--option value`
- Nested attributes: `--hyps.momentum 0.937`

### Reference

- CLI schema: `GenOptions` in `generation.py` and `QatOptions` in `train_qat.py`

---

## Codebase Overview

| Folder          | Description                                          |
| --------------- | ---------------------------------------------------- |
| `baseline/`     | Legacy baseline methods (refer to their `README.md`) |
| `configs/`      | Model and dataset configuration files                |
| `mmcv_custom/`  | Extensions to `mmcv`                                 |
| `mmdet_custom/` | Extensions to `mmdetection`                          |
| `quant/`        | Neural network quantization logic                    |
| `tools/`        | Executable scripts for generation and training       |

---

## Stage I: Task-Specific Calibration Set Generation

To synthesize calibration data:

```bash
    GENERATION_CONFIG=./tools/config/generation.json \
    python -m tools.generation \
      --config ./configs/mask_rcnn/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
      --dataset_config ./configs/datasets/generation.py \
      --pretrained_weights ./pretrained/swin_t.pth \
      --work_dir ./outputs/generated_calib \
      --calibration_size 2048 \
      --batch_size 32 \
      --devices 0
```

### Argument Descriptions

- `GENERATION_CONFIG`: Default generation hyperparameters
- `--config`: Model config (e.g., Swin-T or Swin-S)
- `--dataset_config`: Use `./configs/datasets/generation.py`
- `--pretrained_weights`: Path to pretrained backbone
- `--work_dir`: Output directory for generated images
- `--calibration_size`: Total number of images to generate
- `--batch_size`: Number of images per batch. Reduce if you get CUDA OOM
- `--devices`: Single CUDA device ID (e.g., `0`)

📚 Full argument list: tools/generation.py

---

## Stage II: Quantization-Aware Training (QAT)

We evaluate the following QAT setups:

1. Full COCO real data (baseline)
2. 2k COCO real images (baseline)
3. 2k synthetic images (ours)

All commands below are run from the `dfq-toolkit/vit` directory.

---

### 1. Full Real Data QAT

```bash
    TRAIN_CONFIG=./tools/config/qat_full.json \
    python -m tools.train_qat \
      --work_dir ./outputs/qat_full \
      --config ./configs/mask_rcnn/mask_rcnn_swin-t-p4-w7_fpn_fp16_ms-crop-3x_coco.py \
      --pretrained_path ./pretrained/swin_t.pth \
      --dataset_path ./data/coco \
      --quant_mode.kind quantize_sym \
      --quant_mode.weight_bits 8 \
      --quant_mode.activation_bits 8
```

Use `quantize_sym` for LSQ, or `quantize_asym` for LSQ+.

---

### 2. 2k Real Data QAT

```bash
    TRAIN_CONFIG=./tools/config/2048_qat.json \
    python -m tools.train_qat \
      --work_dir ./outputs/qat_2k \
      --config ./configs/... \
      ...
```

Same arguments as above. Just change the config and dataset accordingly.

---

### 3. QAT with Generated Data + Knowledge Distillation (Ours)

```bash
    TRAIN_CONFIG=./tools/config/qat_syn.json \
    python -m tools.train_qat \
      --work_dir ./outputs/qat_synthetic \
      --pseudo_data ./outputs/generated_calib \
      --enable_kd \
      --kd_modules block \
      --original_loss_weight 1.0 \
      --kd_loss_weight 1.0 \
      --mse_loss_weight 1.0
```

#### Additional Arguments

- `--pseudo_data`: Path to generated calibration data
- `--enable_kd`: Enable knowledge distillation
- `--kd_modules`: Distill specific modules (`block`, `layer_norm` or `all`). By default `block` is preferred.
- `--*_loss_weight`: Set loss balancing weights

📚 Full argument list: tools/train_qat.py

---

## Useful Tips

- Use `tmux` or `screen` to keep processes alive after SSH disconnection
- Save logs by piping output:

```bash
python -m your_script | tee train.log
```

- Always prefer:

```bash
      python -m module_name
```

  over

```bash
      python script.py
```

---

## 🎉 You're Ready

You should now be able to reproduce the QAT results on Transformer-based Mask R-CNN architectures.

Need help? Drop an issue at [GitHub](https://github.com/DFQ-Dojo/dfq-toolkit/issues).

Happy quantizing!

---

## 📊 Evaluation Table

### Model: Mask R-CNN with Swin-T / Swin-S (MS-COCO val)

#### FP32 Baseline

| Method     | Real Data | #Images | Precision | Swin-T    | Swin-S    |
| ---------- | --------- | ------- | --------- | --------- | --------- |
| Pretrained | ✅         | 120k    | FP32      | 46.0/68.1 | 48.5/70.2 |

#### W8A8

| Method   | Real Data | #Images    | Precision | Swin-T        | Swin-S        |
| -------- | --------- | ---------- | --------- | ------------- | ------------- |
| LSQ      | ✅         | 120k       | W8A8      | 45.9/68.0     | 48.1/69.7     |
| LSQ      | ✅         | 2k         |           | 44.4/65.9     | 47.0/68.6     |
| **Ours** | ❌         | 2k (synth) |           | **45.1/66.7** | **47.1/68.8** |

#### W6A6

| Method   | Real Data | #Images    | Precision | Swin-T        | Swin-S        |
| -------- | --------- | ---------- | --------- | ------------- | ------------- |
| LSQ      | ✅         | 120k       | W6A6      | 44.7/66.8     | 47.1/68.8     |
| LSQ      | ✅         | 2k         |           | 41.2/62.9     | 44.4/65.9     |
| **Ours** | ❌         | 2k (synth) |           | **42.0/63.0** | **45.1/65.8** |

#### W4A8

| Method   | Real Data | #Images    | Precision | Swin-T        | Swin-S        |
| -------- | --------- | ---------- | --------- | ------------- | ------------- |
| LSQ      | ✅         | 120k       | W4A8      | 45.5/64.7     | 47.8/69.4     |
| LSQ      | ✅         | 2k         |           | 43.3/65.2     | 45.9/67.3     |
| **Ours** | ❌         | 2k (synth) |           | **43.0/64.2** | **46.2/67.1** |
