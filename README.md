# SGCB Ultra-Lite YOLOv8 Experiments

Standalone reproducibility repository for **Ultra-Lite YOLOv8 with Spatially-Gated Context Blocks (SGCB)** for aerial tiny-object detection on VisDrone2019-DET, with additional AU-AIR validation.

Reproducibility package: [Zenodo record 21210376](https://zenodo.org/records/21210376).

## Citation

If you use this repository, please cite the archived reproducibility package.

```bibtex
@software{abdellatif2026ultralite,
  title={Ultra-Lite YOLOv8: Enhancing Tiny Object Detection via Selective SGCB Modules and High-Resolution Feature Maps},
  author={Abdellatif, Amr O. and Sakr, Noha and Haikal, Amira Y.},
  year={2026},
  doi={10.5281/zenodo.21210376},
  url={https://zenodo.org/records/21210376}
}
```

## Setup

We recommend using `uv`, but `pip` or `conda` can also be used.

Conda environment:

```bash
conda env create -f environment.yml
conda activate sgcb
```

Editable install with `uv`:

```powershell
uv pip install -e .
```

Optional Roboflow fallback:

```powershell
cp .env.example .env
```

Then edit `.env` and add `ROBOFLOW_API_KEY` if you plan to use the Roboflow download path.

## License

This repository is released under AGPL-3.0. The vendored Ultralytics code under
`third_party/ultralytics/` is also AGPL-3.0. See [LICENSE](LICENSE) and
`third_party/ultralytics/LICENSE` for details.

## Datasets

The primary controlled benchmark is **VisDrone2019-DET**. The canonical dataset config is:

```text
configs/VisDrone.yaml
```

Expected local layout:

```text
data/VisDrone/
  VisDrone2019-DET-train/images
  VisDrone2019-DET-val/images
  VisDrone2019-DET-test-dev/images
```

Download helper:

```bash
python scripts/download_visdrone.py
```

The additional validation benchmark is **AU-AIR**, converted to YOLO format with a deterministic seed-0 split. The canonical AU-AIR dataset config is:

```text
data/AU-AIR/yolo-auair8/dataset.yaml
```

AU-AIR preparation helper:

```bash
python scripts/download_prepare_auair.py
```

## Experiments

### Experiment 1: Nano Controlled Comparison

Nano-scale models at `384 x 384`.

```bash
python experiments/exp1_nano_reproduction.py --epochs 300 --device 0 --batch 30 --imgsz 384
```

Models:

- YOLOv8n baseline
- YOLO11n baseline
- UltraLite-Nano

### Experiment 2: Small/X Controlled Scaling

Small and extra-large models at `640 x 640`.

```bash
python experiments/exp2_large_scaling.py --epochs 300 --device 0 --batch 30 --imgsz 640
```

Models:

- YOLOv8s baseline
- YOLO11s baseline
- UltraLite-S
- UltraLite-X

### AU-AIR Additional Validation

Nano-scale AU-AIR validation at `384 x 384`.

```bash
python experiments/auair_exp1_nano_reproduction.py --epochs 300 --device 0 --batch 30 --imgsz 384
```

Small/X AU-AIR scaling validation at `640 x 640`.

```bash
python experiments/auair_exp2_large_scaling.py --epochs 300 --device 0 --batch 30 --imgsz 640
```

### Experiment 3: Full Ablation Study

The ablation uses the full controlled Nano protocol, not the earlier 10-epoch pilot.

```bash
python experiments/ablation_study.py --epochs 300 --device 0 --batch 30 --imgsz 384
```

Default output:

```text
results/ablation_full_ep300_bs30_img384_pat10/
```

### Optional SGCB Kernel-Size Ablation

The default SGCB spatial-gating kernel is `7x7`. To test sensitivity to nearby context sizes, run the controlled Nano kernel ablation for `5x5` and `9x9`.

```powershell
conda run --no-capture-output -n pytorch_env python experiments/sgcb_kernel_ablation.py `
  --data configs/VisDrone.yaml `
  --kernels 5 9 `
  --epochs 300 `
  --batch 30 `
  --imgsz 384 `
  --device 0
```

Default output:

```text
results/sgcb_kernel_ablation_ep300_bs30_img384_pat10/
```

The script generates model YAMLs under `configs/generated/`, trains each kernel variant, validates each resulting `weights/best.pt`, and writes `sgcb_kernel_ablation_summary.csv`.

Downloaded result summary:

```text
downloaded_results/sgcb_kernel_ablation_ep300_bs30_img384_pat10/sgcb_kernel_ablation_summary.csv
```

## Controlled Validation Metrics

Controlled VisDrone metrics are produced by revalidating each trained `weights/best.pt` checkpoint on the VisDrone2019-DET validation split.

```bash
python scripts/validate_controlled_checkpoints.py \
  --data configs/VisDrone.yaml \
  --device 0 \
  --batch 30 \
  --output results/validation/controlled_bestpt_val_summary.csv
```

For CPU-only validation:

```bash
python scripts/validate_controlled_checkpoints.py \
  --data configs/VisDrone.yaml \
  --device cpu \
  --batch 8 \
  --output results/validation/controlled_bestpt_val_summary.csv
```

## Tiny-Object Size-Stratified Recall

Reviewer-facing tiny-object recall is computed from existing `weights/best.pt` checkpoints and the VisDrone validation labels. It does not require retraining.

```powershell
conda run --no-capture-output -n pytorch_env python scripts/size_stratified_recall.py `
  --data configs/VisDrone.yaml `
  --models yolov8s_baseline ultralite_s_ours `
  --device 0 `
  --batch 1 `
  --half `
  --output results/validation/visdrone_size_stratified_recall_s_models.csv
```

The script uses equivalent side length `sqrt(w h)` in original-image pixels, class-aware IoU matching at 0.50, and confidence threshold 0.001.

### VisDrone Per-Class AP

Per-class AP is computed from the existing VisDrone `weights/best.pt` checkpoints; no retraining is required.

```powershell
python scripts/visdrone_ap_error_analysis.py `
  --skip-area `
  --device 0 `
  --half `
  --per-class-output results/validation/visdrone_per_class_ap.csv
```

### VisDrone COCO-Style Area AP And Error Analysis

COCO-style area AP and diagnostic error analysis are computed from existing VisDrone `weights/best.pt` checkpoints; no retraining is required.

```powershell
python scripts/visdrone_ap_error_analysis.py `
  --models yolov8s_baseline ultralite_s_ours `
  --skip-per-class `
  --area-output results/validation/visdrone_coco_area_ap_s_models.csv `
  --error-output results/validation/visdrone_error_summary_s_models.csv `
  --confusion-output results/validation/visdrone_confusion_pairs_s_models.csv
```

Occlusion-specific metrics remain future work because standard YOLO label files do not store the original VisDrone occlusion/truncation metadata.

## Model Checkpoints

Large checkpoint files are not tracked by git. The expected local `best.pt` paths for the controlled validation script are:

```text
Comparison_Exp1_ep300_bs30_img384_pat10/yolov8n_baseline/weights/best.pt
Comparison_Exp1_ep300_bs30_img384_pat10/yolo11n_baseline/weights/best.pt
Comparison_Exp1_ep300_bs30_img384_pat10/yolov8n_local_attn_ultra/weights/best.pt
Comparison_Exp2_ep300_bs30_img640_pat10/yolov8s_baseline/weights/best.pt
Comparison_Exp2_ep300_bs30_img640_pat10/yolo11s_baseline/weights/best.pt
Comparison_Exp2_ep300_bs30_img640_pat10/ultralite_s_ours/weights/best.pt
Comparison_Exp2_ep300_bs30_img640_pat10/ultralite_x_ours/weights/best.pt
```

The archived reproducibility package is available through the DOI badge above.

## ONNX Export And Deployment

Bulk conversion:

```bash
python scripts/bulk_convert_onnx.py
```

Single model export:

```bash
python scripts/export_onnx.py --model path/to/best.pt --imgsz 640
```

Converted models are stored under each experiment folder in:

```text
ONNX_weights/best.onnx
```

## SGCB Module

The **Spatially-Gated Context Block (SGCB)** preserves spatial detail for tiny aerial objects by avoiding global spatial pooling. It uses:

1. Large-kernel depthwise convolution to capture local context.
2. Dense feature aggregation to preserve fine detail.
3. Selective insertion at P3/P4 stages to balance accuracy and parameter count.

## Repository Structure

- `experiments/`: Training and controlled comparison scripts.
- `third_party/ultralytics/`: Vendored Ultralytics code with SGCB model definitions.
- `configs/`: Primary dataset and model configuration files.
- `scripts/`: Dataset, validation, GFLOPs, export, and deployment utilities.
- `results/`: Auto-generated experiment outputs.
- `downloaded_results/`: Locally downloaded result bundles; ignored by git.
- `data/`: Local datasets; ignored by git.
## Troubleshooting

- If CUDA runs out of memory, reduce `--batch` or use `--device cpu` for validation.
- Make sure your PyTorch build has CUDA support before using `--device 0`.
- On Windows, the scripts set `KMP_DUPLICATE_LIB_OK=TRUE` where needed to avoid OpenMP conflicts.
