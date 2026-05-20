# SGCB Ultra-Lite YOLOv8 Experiments 🚀


This repository contains the standalone reproducibility environment for the research on **SGCB (Selective Grouped Convolution Block) Ultra-Lite YOLO architectures**.

<<<<<<< HEAD
[DOI: https://doi.org/10.5281/zenodo.20157395](https://doi.org/10.5281/zenodo.20157395)
=======
[![DOI](https://doi.org/10.5281/zenodo.20157395)
>>>>>>> def22cd057537edd23f3f9ba20e0b14a35e39ab5

---

## 📖 Citation

If you use this work in your research, please cite our article published in **The Visual Computer**:

> **Abdellatif, A. O., Sakr, N., & Haikal, A. Y. (2026). Ultra-Lite YOLOv8: Enhancing Tiny Object Detection via Selective SGCB Modules and High-Resolution Feature Maps. *The Visual Computer*.**

```bibtex
@article{abdellatif2026ultralite,
  title={Ultra-Lite YOLOv8: Enhancing Tiny Object Detection via Selective SGCB Modules and High-Resolution Feature Maps},
  author={Abdellatif, Amr O. and Sakr, Noha and Haikal, Amira Y.},
  journal={The Visual Computer},
  year={2026},
  publisher={Springer Nature}
}
```

---

## ⚡ Quick Start (3 Steps)

### 1. Setup Environment
We recommend using `uv` for high-performance dependency management, but you can also use `pip` or `conda`.

```powershell
# Install dependencies
uv pip install -e .

# Setup your Roboflow API Key (Optional fallback)
cp .env.example .env
# Edit .env and add your ROBOFLOW_API_KEY
```

### 2. Download Data
The experiments use the **VisDrone2019** dataset. Run this script to automatically download and prepare the data:

```bash
# Windows (PowerShell)
python .\scripts\download_visdrone.py

# Linux / macOS / WSL
python scripts/download_visdrone.py
```

### 3. Run Experiments
The repository is organized into two primary experiments:

#### 🟢 Experiment 1: Nano Model Reproduction (Thesis Results)
Reproduces the core results for Nano-scale models at optimized inference speeds (384px).
```bash
# Windows
python .\experiments\exp1_nano_reproduction.py --epochs 300 --device 0 --batch 30 --imgsz 384

# Linux / macOS
python experiments/exp1_nano_reproduction.py --epochs 300 --device 0 --batch 30 --imgsz 384
```
- **Models**: YOLOv8n, YOLOv8n-UltraLite (Local Attn), YOLO11n.

#### 🔵 Experiment 2: S vs UltraLite S/X Comparison
Evaluates how the Ultra-Lite enhancements scale to larger architectures (S and X) at 640px resolution.
```bash
# Windows
python .\experiments\exp2_large_scaling.py --epochs 300 --device 0 --batch 30 --imgsz 640

# Linux / macOS
python experiments/exp2_large_scaling.py --epochs 300 --device 0 --batch 30 --imgsz 640
```
- **Models**: YOLOv8s, UltraLite-S (Ours), UltraLite-X (Ours).

#### 🧪 Experiment 3: Ablation Study
Quantifies the exact contribution of the P2 head and SGCB modules.
```bash
# Windows
python .\experiments\ablation_study.py --epochs 10 --device 0

# Linux / macOS
python experiments/ablation_study.py --epochs 10 --device 0
```
- **Output**: Results are saved in `results/ablation/`. See [Ablation README](file:///d:/projects/git-hub/Ultralytics%20-%20masters/ultralytics/sgcb-experiments/results/ablation/README.md) for details.

---

## 🚀 Model Deployment & ONNX

To support edge deployment, models can be converted to ONNX format. We provide a bulk conversion script that iterates through experiment folders and creates optimized weights.

### Export Comparison Models
To convert all models from the primary comparison experiment:
```bash
# Windows
python .\scripts\bulk_convert_onnx.py

# Linux / macOS
python scripts/bulk_convert_onnx.py
```
- **Location**: Converted models are stored in `<experiment_folder>/ONNX_weights/best.onnx`.
- **Note**: Requires `onnxscript` and `onnxruntime` to be installed.

---

## 🧠 Key Algorithm: SGCB Module

The **Spatially-Gated Context Block (SGCB)** is a novel architectural unit designed to preserve spatial details in aerial imagery. Unlike standard attention (SE) which uses Global Average Pooling, SGCB utilizes:
1. **Large-Kernel Depthwise Convolution (7x7)**: Captures local context without squashing spatial dimensions.
2. **Dense Feature Aggregation**: Concatenates raw and refined features to prevent vanishing information for small targets.
3. **Selective Replacement**: Applied specifically to P3 and P4 layers to maximize efficiency (Negative-Overhead).

---

## 📂 Repository Structure
- `experiments/`: Core training and comparison scripts for the paper.
- `third_party/ultralytics/`: Vendored YOLOv8 codebase with integrated SGCB modules.
- `configs/`: Dataset and model configuration files.
- `scripts/`: Utility scripts for data downloading, batch runs, and ONNX conversion.
- `results/`: (Auto-generated) Storage for ablation studies and visual metrics.
- `data/`: (Auto-generated) Location for training datasets.

---
📖 *Methodology: Technical Replication Report for Modified YOLO Architectures*
