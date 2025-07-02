# Tutorial: YOLOv5

This tutorial provides step-by-step instructions for reproducing the experimental results for the **YOLOv5** series from our paper.

If you encounter any issues, please open a ticket in our [GitHub Issues](https://github.com/DFQ-Dojo/dfq-toolkit/issues) section.

---

## 🚀 Prerequisites

### Hardware Requirements

- At least **one NVIDIA GPU**
- Minimum **32 GiB of RAM**

### Software Requirements

We use [uv](https://docs.astral.sh/uv/) for Python environment and dependency management.  
To get started, follow the [official installation guide](https://docs.astral.sh/uv/getting-started/installation/).

Once installed, navigate to the `dfq-toolkit/yolov5` directory and run:

```bash
uv sync
```

This will install all necessary dependencies and set up a virtual environment.

To activate the environment:

```bash
source .venv/bin/activate
```

> ⚠️ In some regions, `uv sync` may fail due to network restrictions.  
> You can resolve this by configuring a custom [index](https://docs.astral.sh/uv/concepts/indexes/).

---

### Datasets and Pre-trained Models

Datasets and pre-trained models are automatically downloaded if not already present.  
The expected directory structure is:

```yaml
- datasets/
    - coco/
        - annotations/
        - images/
        - labels/
- dfq-toolkit/
    - yolov5/
        - .venv/
        - yolov5s.pt
        - yolov5m.pt
        ...
    - yolo11/
        ...
```

---

## 🧭 Code Structure

| File / Directory        | Description                                       |
| ----------------------- | ------------------------------------------------- |
| `models/`               | Model definitions using LSQ quantization          |
| `data/`                 | Dataset configuration files and hyperparameters   |
| `utils/`                | Helper functions for data-free quantization       |
| `distill_image.py`      | Script to generate calibration datasets           |
| `datafree_qat_ckpt.py`  | Script for QAT using generated calibration images |
| `generate_one_label.py` | Script for generating synthetic one-box labels    |

---

## 🧪 Real-label Based Calibration Set Synthesis

To generate task-specific calibration images using real labels, run:

```bash
python -m distill_image \
    --weights <path_to_weights> \
    --batch-size <batch_size> \
    --project <output_dir> \
    --data <data_yaml_path> \
    --r_feature 0.01 \
    --hyp data/hyps/hyp.distill.yaml \
    --lr 0.2 \
    --verifier \
    --verifier_weights <path_to_weights> \
    --imgsz 640 \
    --iterations 2500 \
    --tv_l1 0.0 \
    --tv_l2 0.0005 \
    --main_loss_multiplier 0.5 \
    --first_bn_coef 2.0 \
    --do_clip \
    --patience 500 \
    --calibration_size 2048 \
    --random_erase
```

### 🔧 Key Arguments

| Argument                          | Description                                                                                      |
| --------------------------------- | ------------------------------------------------------------------------------------------------ |
| `--weights`, `--verifier_weights` | Path to the model used for image generation. ⚠️ These must be the same in current implementation. |
| `--batch-size`                    | Number of images per generation batch. Reduce if CUDA out-of-memory occurs.                      |
| `--project`                       | Output directory for saving generated images.                                                    |
| `--data`                          | Dataset YAML path (`coco.yaml` for real labels, `cocoonebox.yaml` for adaptive sampling).        |
| `--r_feature`                     | Coefficient for $R_{feature}$ term in the loss.                                                  |
| `--hyp`                           | Path to hyperparameter configuration.                                                            |
| `--lr`                            | Learning rate for optimization.                                                                  |
| `--verifier`                      | Enables adaptive label verification.                                                             |
| `--imgsz`                         | Output image resolution. COCO standard is 640.                                                   |
| `--iterations`                    | Optimization steps per image. More steps yield better quality.                                   |
| `--tv_l1`, `--tv_l2`              | Total variation regularization loss coefficients.                                                |
| `--main_loss_multiplier`          | Weight for detection loss component.                                                             |
| `--first_bn_coef`                 | Loss weight for the first BatchNorm layer.                                                       |
| `--do_clip`                       | Clip pixel values to [0, 1] after each update.                                                   |
| `--patience`                      | Early stopping threshold (no improvement).                                                       |
| `--calibration_size`              | Total number of synthetic images to generate.                                                    |
| `--random_erase`                  | Enables random erasing during generation.                                                        |

Refer to the [full script](https://github.com/DFQ-Dojo/dfq-toolkit/blob/main/yolov5/distill_image.py) for all available arguments.

---

## 🔁 Adaptive Label Sampling

This section describes how to generate pseudo labels using one-box sampling.

### Step 1: Generate Synthetic Labels

```bash
python -m generate_one_label \
    --numImages 5120 \
    --outdir <output_directory>
```

| Argument      | Description                  |
| ------------- | ---------------------------- |
| `--numImages` | Number of labels to generate |
| `--outdir`    | Directory to save labels     |

Update the `labels` path in `./data/cocoonebox.yaml` to point to the generated directory.

---

### Step 2: Generate Pseudo Images with Adaptive Sampling

```bash
python -m distill_image \
    --weights yolov5s.pt \
    --batch-size 64 \
    --project runs/Distill/Images \
    --data data/cocoonebox.yaml \
    --r_feature 0.01 \
    --hyp data/hyps/hyp.distill.yaml \
    --lr 0.2 \
    --verifier \
    --verifier_weights yolov5s.pt \
    --imgsz 160 \
    --iterations 2500 \
    --tv_l1 0.0 \
    --tv_l2 0.0005 \
    --main_loss_multiplier 0.5 \
    --first_bn_coef 2.0 \
    --do_clip \
    --patience 500 \
    --calibration_size 50000 \
    --random_erase \
    --box-sampler \
    --box-sampler-warmup 800 \
    --box-sampler-conf 0.2 \
    --box-sampler-overlap-iou 0.35 \
    --box-sampler-minarea 0.01 \
    --box-sampler-maxarea 0.85 \
    --box-sampler-earlyexit 2800
```

Most arguments are identical to the real-label scenario. The `--box-sampler` flags introduce hyperparameters for adaptive sampling. Defaults are recommended for most use cases.

Output will be saved in: `runs/Distill/Images/`

---

## 🔧 Stage II: Quantization-Aware Training (QAT) with Task-Specific Distillation

We evaluate two QAT strategies:

1. **QAT using full real dataset** *(baseline)*
2. **QAT using synthetic calibration data** *(ours)*

Run all commands from the `./yolov5` directory.

---

### 1. QAT with Real Dataset (Baseline)

```bash
python -m qat \
    --data coco.yaml \
    --epochs 100 \
    --weights yolov5s.pt \
    --cfg 8bitlsq_yolov5s.yaml \
    --batch-size <batch_size> \
    --mode lsq \
    --optimizer Adam \
    --hyp data/hyps/hyps.qat.yaml \
    --project runs/train/8bitLSQ \
    --patience 5 \
    --check-ptq
```

| Argument       | Description                               |
| -------------- | ----------------------------------------- |
| `--data`       | Dataset config YAML (e.g., `coco.yaml`)   |
| `--epochs`     | Maximum number of training epochs         |
| `--weights`    | Pretrained model checkpoint               |
| `--cfg`        | Model architecture definition             |
| `--batch-size` | Training batch size (based on GPU memory) |
| `--mode`       | Quantization mode (`lsq` or `lsqplus`)    |
| `--optimizer`  | Optimizer for QAT (e.g., `Adam`)          |
| `--hyp`        | Hyperparameters for QAT                   |
| `--project`    | Output directory for logs and checkpoints |
| `--patience`   | Early stopping patience                   |
| `--check-ptq`  | Evaluate PTQ results before QAT           |

---

### 2. QAT with Generated Calibration Data (Ours)

```bash
python -m datafree_qat_ckpt \
    --data coco.yaml \
    --epochs 100 \
    --weights yolov5s.pt \
    --cfg 8bitlsq_yolov5s.yaml \
    --batch-size 32 \
    --mode quantize_sym \
    --optimizer Adam \
    --hyp data/hyps/hyps.qat.yaml \
    --project runs/train/ComparePseudo/8bitLSQ \
    --ckpt_path <generated_weights_dir> \
    --train_kind generate \
    --device 1 \
    --kd kdt4+mse \
    --teacher-weight yolov5s.pt \
    --teacher-cfg yolov5s.yaml \
    --module cnnbn \
    --ori-loss-weight 0.04 \
    --mse-loss-weight 1.0 \
    --kd-loss-weight 0.1
```

| Argument                                                     | Description                                                               |
| ------------------------------------------------------------ | ------------------------------------------------------------------------- |
| `--mode`                                                     | `quantize_sym` (LSQ) or `quantize_asym` (LSQ+)                            |
| `--ckpt_path`                                                | Directory containing generated image weights (should end with `/weights`) |
| `--train_kind`                                               | Type of training data (`generate` for synthetic)                          |
| `--device`                                                   | CUDA device ID                                                            |
| `--kd`                                                       | Knowledge distillation strategy (recommended default: `kdt4+mse`)         |
| `--teacher-weight`, `--teacher-cfg`                          | Same as `--weights` and original `cfg`                                    |
| `--module`                                                   | Distillation module (use default: `cnnbn`)                                |
| `--ori-loss-weight`, `--mse-loss-weight`, `--kd-loss-weight` | Loss weights for original, MSE, and distillation terms                    |

---