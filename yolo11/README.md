# Tutorial: YOLO11

This tutorial explains how to reproduce the experimental results in the **YOLO11** series.  
If you encounter any issues, feel free to report them in our [GitHub issues](https://github.com/DFQ-Dojo/dfq-toolkit/issues) section.

---

## Prerequisites

### Hardware Requirements

- At least **one NVIDIA GPU** is required.
- At least **32 GiB of RAM** is recommended.

### Software Requirements

We use [uv](https://docs.astral.sh/uv/) to manage the Python environment.  
Follow the [uv installation guide](https://docs.astral.sh/uv/getting-started/installation/) to get started.

Once `uv` is installed, navigate to the `dfq-toolkit/yolo11` directory and run:

```bash
uv sync
```

This will install all required dependencies and create a virtual environment. To activate it:

```bash
source .venv/bin/activate
```

Note: For network reasons, `uv sync` may not work. If it does not work, you maybe need to set uv [index](https://docs.astral.sh/uv/concepts/indexes/) to collect necessary dependencies.

### Dataset and Model

Datasets and pre-trained models will be automatically downloaded if not present.  
The expected directory structure is:

```yaml
- datasets
    - coco
        - annotations/
        - images/
        - labels/
        ...
- dfq-toolkit
    - yolo11/
        - .venv/
        - yolo11s.pt
        - yolo11m.pt
        ...
    - yolov5/
    ...
```

### The Pydantic CLI Interface

This codebase uses [Pydantic](https://docs.pydantic.dev/latest/) for command-line parsing.

Since the command-line parsing semantics can be complex, this section explains some formatting issues. When using undocumented options, refer to the code for guidance.

CLI schema definitions are located in `config.py` of each sub-module (e.g., `yolo11/external/config.py`). These files define `Config` classes.

- For scalar (simple) values, use: `--option xxx`
- For complex values (e.g., `momentum` of `THyperparameter`), use: `--hyps.momentum 0.937`

---

## Code Structure

This codebase consists of five main modules:

- `external`: Utilities and scaffolds for QAT.
- `generation`: Components and scaffold for pseudo data distillation.
- `label`: Programs for pseudo label generation.
- `quant`: Routines and classes for neural network quantization.
- `tools`: Scripts for visualization and data type conversion.

---

## Task-Specific Calibration Set Synthesis

To generate task-specific calibration images, run the following command in the `./yolo11` working directory:

```bash
GENERATION_CONFIG=./generation/config/true_label.json python -m generation.main \
    <args>
```

### Important Arguments

- `--calibration_size`: Number of images to generate.
- `--teacher_weights` / `--relabel_weights`: Path to the model used for generation.  
  ⚠️ **These two must be the same in the current implementation.**
- `--dataset_configs.batch_size`: Controls generation batch size.  
  If you encounter CUDA OOM errors, reduce this value.

For full argument explanation, refer to our [config code](https://github.com/DFQ-Dojo/dfq-toolkit/blob/main/yolo11/generation/config.py).

---

## Stage II: Quantization-Aware Training (QAT) with Task-Specific Distillation

We evaluate three training strategies:

1. **QAT with full real dataset** (baseline)  
2. **QAT with 2k real images** (baseline)  
3. **QAT with generated calibration data** (ours)

All the following commands should be run from the `./yolo11` directory.

---

### 1. QAT with Full Real Data

```bash
TRAIN_CONFIG=./external/config/qat_full.json python -m external.main \
    <args>
```

#### Arguments

- `--device`: CUDA device to use.
- `--batch_size`: Training batch size. Reduce this if encountering CUDA OOM errors.
- `--model`: Model file name (e.g., `yolo11s.pt`, `yolo11m.pt`, `yolo11l.pt`)
- `--model_quantize_mode`: Use `quantize_sym` for LSQ, `quantize_asym` for LSQ+.  
  ➤ To reproduce our method, **use LSQ** (`quantize_sym`).
- `--model_quantize_mode.weight_bits`: Bitwidth for weights.
- `--model_quantize_mode.activation_bits`: Bitwidth for activations.

For full argument explanation, refer to our [config code](https://github.com/DFQ-Dojo/dfq-toolkit/blob/main/yolo11/external/config.py).

---

### 2. QAT with 2k Real Data

```bash
TRAIN_CONFIG=./external/config/qat.json python -m external.main \
    <args>
```

Arguments are the same as in **(1)**.

---

### 3. QAT with Generated Calibration Data (Ours)

```bash
TRAIN_CONFIG=./external/config/kd.json python -m external.main \
    <args>
```

#### Additional Arguments

- `--generated_weights_path`: Path to generated calibration data.  
  Example: `runs/Distill/exp/weights`
- `--kd_method.teacher_weight`: Ensure this matches the `--model` argument.

---

## Trivia

To keep training running after closing your SSH session, use `tmux`.  
To save runtime logs, pipe output to `tee`:

```bash
python -m your_script | tee log.txt
```

Always prefer `python -m module` over `python script.py` when possible!

---

## 🚀 Bon Voyage

You’re now ready to reproduce the results in the YOLO11 series.  
If you run into any issues, reach out via our [GitHub issues](https://github.com/DFQ-Dojo/dfq-toolkit/issues).

Happy experimenting!

## Table: Comparison with real data QATs on YOLOv5/YOLO11 on MS-COCO validation set

| Method     | Real Data | Num Data     | Prec. | YOLOv5-s      | YOLOv5-m      | YOLOv5-l      | YOLO11-s     | YOLO11-m     | YOLO11-l     |
|------------|-----------|--------------|-------|---------------|---------------|---------------|---------------|---------------|---------------|
| Pre-trained | ✅         | 120k (full)  | FP    | 37.4/56.8     | 45.4/64.1     | 49.0/67.3     | 47.0/65.0     | 51.5/70.0     | 53.4/72.5     |

### W8A8

| Method     | Real Data | Num Data     | Prec. | YOLOv5-s      | YOLOv5-m      | YOLOv5-l      | YOLO11-s     | YOLO11-m     | YOLO11-l     |
|------------|-----------|--------------|-------|---------------|---------------|---------------|---------------|---------------|---------------|
| LSQ        | ✅         | 120k (full)  | W8A8  | 35.7/54.9     | 43.2/62.2     | 46.0/64.9     | 44.9/61.8     | 49.1/66.2     | 50.4/67.4     |
| LSQ+       | ✅         | 120k (full)  |       | 35.4/54.6     | 43.3/62.4     | 46.3/64.9     | 45.1/61.8     | 49.6/66.7     | 50.9/67.7     |
| LSQ        | ✅         | 2k           |       | 31.6/50.6     | 36.5/55.6     | 40.3/59.1     | 44.0/60.8     | 47.6/64.5     | 48.8/65.8     |
| LSQ+       | ✅         | 2k           |       | 31.5/50.3     | 36.6/55.8     | 40.1/58.6     | 43.8/60.7     | 47.8/64.7     | 48.5/65.3     |
| **Ours**   | ❌         | 2k           |       | **35.8/55.0** | **43.6/62.3** | **47.3/65.6** | **45.6/62.3** | **50.0/66.5** | **51.8/68.4** |

### W6A6

| Method     | Real Data | Num Data     | Prec. | YOLOv5-s      | YOLOv5-m      | YOLOv5-l      | YOLO11-s     | YOLO11-m     | YOLO11-l     |
|------------|-----------|--------------|-------|---------------|---------------|---------------|---------------|---------------|---------------|
| LSQ        | ✅         | 120k (full)  | W6A6  | 31.5/49.9     | 41.3/60.0     | 43.3/62.1     | 43.0/59.7     | 47.4/64.2     | 48.6/65.3     |
| LSQ+       | ✅         | 120k (full)  |       | 32.3/50.9     | **41.3/60.3** | 43.4/62.3     | **43.2/59.8** | **47.6/64.3** | **48.9/65.8** |
| LSQ        | ✅         | 2k           |       | 28.9/47.2     | 35.0/53.9     | 37.7/55.7     | 41.5/58.3     | 45.0/61.9     | 45.8/62.5     |
| LSQ+       | ✅         | 2k           |       | 28.6/46.7     | 34.2/52.6     | 37.5/55.8     | 41.6/58.2     | 44.8/61.7     | 45.9/62.8     |
| **Ours**   | ❌         | 2k           |       | **32.7/51.4** | 41.0/59.7     | **45.1/63.3** | 43.0/59.3     | 47.1/63.2     | 48.4/64.6     |

### W4A8

| Method     | Real Data | Num Data     | Prec. | YOLOv5-s      | YOLOv5-m      | YOLOv5-l      | YOLO11-s     | YOLO11-m     | YOLO11-l     |
|------------|-----------|--------------|-------|---------------|---------------|---------------|---------------|---------------|---------------|
| LSQ        | ✅         | 120k (full)  | W4A8  | 32.2/51.0     | 41.0/59.9     | 44.6/63.5     | 42.4/59.1     | 47.6/64.4     | 48.7/65.6     |
| LSQ+       | ✅         | 120k (full)  |       | 32.3/51.1     | 41.2/60.1     | 44.4/63.2     | **42.7/59.3** | **47.8/64.8** | 49.4/66.3     |
| LSQ        | ✅         | 2k           |       | 28.1/46.5     | 35.8/54.6     | 39.0/57.5     | 40.9/57.5     | 45.2/62.4     | 46.1/63.0     |
| LSQ+       | ✅         | 2k           |       | 29.3/47.8     | 37.8/56.9     | 40.6/59.7     | 40.7/57.3     | 45.2/62.3     | 46.4/63.4     |
| **Ours**   | ❌         | 2k           |       | **33.0/52.5** | **42.6/61.7** | **46.2/64.7** | 42.6/58.9     | 47.7/64.1     | **49.4/65.7** |
