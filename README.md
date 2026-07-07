# SGCB Ultra-Lite YOLOv8 Experiments

Standalone reproducibility repository for **Ultra-Lite YOLOv8 with Spatially-Gated Context Blocks (SGCB)** for aerial tiny-object detection on VisDrone2019-DET, with additional AU-AIR validation.

([https://doi.org/10.5281/zenodo.20157395](https://zenodo.org/records/21210376))
=======
Standalone reproducibility repository for **Ultra-Lite YOLOv8 with Spatially-Gated Context Blocks (SGCB)** for aerial tiny-object detection on VisDrone2019-DET.


## Citation

If you use this repository, please cite the manuscript and the archived reproducibility package. The manuscript is associated with *The Visual Computer* review workflow; final bibliographic details should be updated after publication.

```bibtex
@article{abdellatif2026ultralite,
  title={Ultra-Lite YOLOv8: Enhancing Tiny Object Detection via Selective SGCB Modules and High-Resolution Feature Maps},
  author={Abdellatif, Amr O. and Sakr, Noha and Haikal, Amira Y.},
  journal={The Visual Computer},
  year={2026},
  note={Manuscript under review},
  doi={10.5281/zenodo.20157395}
}
```

## Setup

We recommend using `uv`, but `pip` or `conda` can also be used.

<<<<<<< HEAD
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
=======
```powershell
uv pip install -e .
```

Optional Roboflow fallback:

```powershell
cp .env.example .env
```

Then edit `.env` and add `ROBOFLOW_API_KEY` if you plan to use the Roboflow download path.

## Dataset

The experiments use **VisDrone2019-DET**. The canonical dataset config is:

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

## Experiments

### Experiment 1: Nano Controlled Comparison

Nano-scale models at `384 x 384`.

```bash
>>>>>>> d9952b10fbe0e2796efca6d623c4de029ea870ce
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

<<<<<<< HEAD
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

The final downloaded lean ablation package used for manuscript updates is stored locally under:

```text
downloaded_results/ablation_full_ep300_bs30_img384_pat10_lean/
```

### Optional SGCB Kernel-Size Ablation

The manuscript uses the default `7x7` SGCB spatial-gating kernel. To test sensitivity to nearby context sizes, run the controlled Nano kernel ablation for `5x5` and `9x9`. These two runs are compared with the existing full UltraLite `7x7` row from the ablation table.

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

Current sensitivity results:

| SGCB kernel | Params (M) | GFLOPs | Precision (%) | Recall (%) | mAP50 (%) | mAP50-95 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 5x5 | 2.74 | 4.12 | 35.9 | 26.9 | 25.6 | 14.4 |
| 7x7 default | 2.75 | 4.13 | 37.5 | 26.9 | 25.6 | 14.4 |
| 9x9 | 2.76 | 4.15 | 37.4 | 26.6 | 25.7 | 14.5 |

The results show that the SGCB design is not highly sensitive to nearby odd kernel sizes. The `7x7` setting remains the practical default, not a claimed global optimum.

## Controlled Validation Metrics

The controlled VisDrone manuscript metrics are produced by revalidating each trained `weights/best.pt` checkpoint on the VisDrone2019-DET validation split.

```bash
python scripts/validate_controlled_checkpoints.py \
  --data configs/VisDrone.yaml \
  --device 0 \
  --batch 30 \
  --output journal_submission/controlled_bestpt_val_summary.csv
```

=======
### Experiment 3: Full Ablation Study

The ablation uses the full controlled Nano protocol, not the earlier 10-epoch pilot.

```bash
python experiments/ablation_study.py --epochs 300 --device 0 --batch 30 --imgsz 384
```

Default output:

```text
results/ablation_full_ep300_bs30_img384_pat10/
```

The final downloaded lean ablation package used for manuscript updates is stored locally under:

```text
downloaded_results/ablation_full_ep300_bs30_img384_pat10_lean/
```

## Controlled Validation Metrics

The controlled manuscript metrics are produced by revalidating each trained `weights/best.pt` checkpoint on the VisDrone2019-DET validation split.

```bash
python scripts/validate_controlled_checkpoints.py \
  --data configs/VisDrone.yaml \
  --device 0 \
  --batch 30 \
  --output journal_submission/controlled_bestpt_val_summary.csv
```

>>>>>>> d9952b10fbe0e2796efca6d623c4de029ea870ce
For CPU-only validation:

```bash
python scripts/validate_controlled_checkpoints.py \
  --data configs/VisDrone.yaml \
  --device cpu \
  --batch 8 \
  --output journal_submission/controlled_bestpt_val_summary.csv
```

Canonical validation summary:

```text
journal_submission/controlled_bestpt_val_summary.csv
```

Current controlled best-checkpoint validation results:

| Model | Input | Params (M) | GFLOPs | Precision (%) | Recall (%) | mAP50 (%) | mAP50-95 (%) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n baseline | 384 | 3.01 | 2.95 | 32.6 | 22.4 | 20.5 | 11.0 |
| YOLO11n baseline | 384 | 2.59 | 2.32 | 32.5 | 22.3 | 20.8 | 11.3 |
| UltraLite-Nano | 384 | 2.75 | 4.13 | 37.1 | 26.6 | 25.6 | 14.4 |
| YOLOv8s baseline | 640 | 11.14 | 28.67 | 50.5 | 37.8 | 38.9 | 23.1 |
| YOLO11s baseline | 640 | 9.43 | 21.57 | 51.0 | 37.8 | 38.6 | 23.1 |
| UltraLite-S | 640 | 9.89 | 33.20 | 55.5 | 42.3 | 44.4 | 27.0 |
<<<<<<< HEAD
| UltraLite-X | 640 | 41.95 | 199.46 | 59.4 | 49.4 | 51.0 | 31.7 |

## Tiny-Object Size-Stratified Recall

Reviewer-facing tiny-object recall is computed from existing `weights/best.pt` checkpoints and the VisDrone validation labels. It does not require retraining.

```powershell
conda run --no-capture-output -n pytorch_env python scripts/size_stratified_recall.py `
  --data configs/VisDrone.yaml `
  --models yolov8s_baseline ultralite_s_ours `
  --device 0 `
  --batch 1 `
  --half `
  --output journal_submission/visdrone_size_stratified_recall_s_models.csv
```

Canonical size-stratified recall summary:

```text
journal_submission/visdrone_size_stratified_recall_s_models.csv
```

Current VisDrone size-stratified recall results use equivalent side length `sqrt(w h)` in original-image pixels, class-aware IoU matching at 0.50, and confidence threshold 0.001.

| Model | Input | <10 px | 10-20 px | 20-32 px | >=32 px |
| --- | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s baseline | 640 | 17.1% | 47.1% | 69.4% | 85.1% |
| UltraLite-S | 640 | 37.2% | 63.7% | 76.7% | 87.2% |

### VisDrone Per-Class AP

Per-class AP is computed from the existing VisDrone `weights/best.pt` checkpoints; no retraining is required.

```powershell
python scripts/visdrone_ap_error_analysis.py `
  --skip-area `
  --device 0 `
  --half `
  --per-class-output journal_submission/visdrone_per_class_ap.csv
```

Canonical per-class AP summary:

```text
journal_submission/visdrone_per_class_ap.csv
```

Current UltraLite-S vs. YOLOv8s per-class AP50 gains:

| Class | YOLOv8s AP50 | UltraLite-S AP50 | Gain |
| --- | ---: | ---: | ---: |
| Pedestrian | 42.7% | 51.2% | +8.5 |
| People | 33.1% | 41.8% | +8.7 |
| Bicycle | 13.1% | 17.3% | +4.2 |
| Car | 79.6% | 84.0% | +4.4 |
| Van | 44.8% | 48.9% | +4.1 |
| Truck | 35.1% | 37.2% | +2.1 |
| Tricycle | 26.6% | 31.2% | +4.5 |
| Awning-tricycle | 14.9% | 17.1% | +2.1 |
| Bus | 54.7% | 62.9% | +8.2 |
| Motor | 44.5% | 51.9% | +7.4 |

### VisDrone COCO-Style Area AP And Error Analysis

COCO-style area AP and diagnostic error analysis are computed from existing VisDrone `weights/best.pt` checkpoints; no retraining is required.

```powershell
python scripts/visdrone_ap_error_analysis.py `
  --models yolov8s_baseline ultralite_s_ours `
  --skip-per-class `
  --area-output journal_submission/visdrone_coco_area_ap_s_models.csv `
  --error-output journal_submission/visdrone_error_summary_s_models.csv `
  --confusion-output journal_submission/visdrone_confusion_pairs_s_models.csv
```

Canonical area/error summaries:

```text
journal_submission/visdrone_coco_area_ap_s_models.csv
journal_submission/visdrone_error_summary_s_models.csv
journal_submission/visdrone_confusion_pairs_s_models.csv
```

Current COCO-style area AP results:

| Area group | YOLOv8s AP50 | UltraLite-S AP50 | Gain | YOLOv8s AP50-95 | UltraLite-S AP50-95 | Gain |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| All | 37.1% | 43.2% | +6.1 | 22.6% | 26.5% | +4.0 |
| Small | 19.8% | 26.9% | +7.1 | 9.2% | 13.3% | +4.2 |
| Medium | 43.5% | 44.9% | +1.3 | 27.5% | 29.1% | +1.6 |
| Large | 27.9% | 29.7% | +1.7 | 21.9% | 23.9% | +2.0 |

Diagnostic error analysis at class-aware IoU 0.50 shows that UltraLite-S increases true positives from 23,747 to 27,641 and reduces false negatives from 15,012 to 11,118. Localization errors decrease from 3,959 to 2,601, and class-confusion errors decrease from 4,189 to 3,870. Duplicate detections increase from 10,123 to 19,001, so duplicate suppression remains a limitation.

Occlusion-specific metrics remain future work because standard YOLO label files do not store the original VisDrone occlusion/truncation metadata.

## AU-AIR Validation Metrics

Canonical AU-AIR validation summaries:

```text
journal_submission/auair_exp1_bestpt_val_summary.csv
journal_submission/auair_exp2_combined_best_fitness_summary.csv
```

Current AU-AIR Nano validation results:

| Model | Input | Params (M) | GFLOPs | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8n baseline | 384 | 3.01 | 2.95 | 0.364 | 0.369 | 0.306 | 0.136 |
| YOLO11n baseline | 384 | 2.59 | 2.32 | 0.330 | 0.389 | 0.307 | 0.134 |
| UltraLite-Nano | 384 | 2.75 | 4.13 | 0.376 | 0.364 | 0.319 | 0.139 |

Current AU-AIR scaling validation results:

| Model | Input | Params (M) | GFLOPs | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| YOLOv8s baseline | 640 | 11.14 | 28.66 | 0.416 | 0.405 | 0.344 | 0.151 |
| YOLO11s baseline | 640 | 9.43 | 21.56 | 0.384 | 0.403 | 0.338 | 0.148 |
| UltraLite-S | 640 | 9.89 | 33.19 | 0.356 | 0.414 | 0.346 | 0.156 |
| UltraLite-X | 640 | 41.95 | 199.44 | 0.363 | 0.421 | 0.340 | 0.153 |
=======
| UltraLite-X | 640 | 41.95 | 199.46 | 59.4 | 49.4 | 51.0 | 31.9 |
>>>>>>> d9952b10fbe0e2796efca6d623c4de029ea870ce

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
<<<<<<< HEAD

The **Spatially-Gated Context Block (SGCB)** preserves spatial detail for tiny aerial objects by avoiding global spatial pooling. It uses:

1. Large-kernel depthwise convolution to capture local context.
2. Dense feature aggregation to preserve fine detail.
3. Selective insertion at P3/P4 stages to balance accuracy and parameter count.

## Repository Structure

- `experiments/`: Training and controlled comparison scripts.
- `third_party/ultralytics/`: Vendored Ultralytics code with SGCB model definitions.
- `configs/`: Primary dataset and model configuration files.
- `scripts/`: Dataset, validation, GFLOPs, export, and deployment utilities.
- `journal_submission/`: Manuscript files and canonical controlled validation summary.
- `results/`: Auto-generated experiment outputs.
- `downloaded_results/`: Locally downloaded result bundles; ignored by git.
- `data/`: Local datasets; ignored by git.

=======

The **Spatially-Gated Context Block (SGCB)** preserves spatial detail for tiny aerial objects by avoiding global spatial pooling. It uses:

1. Large-kernel depthwise convolution to capture local context.
2. Dense feature aggregation to preserve fine detail.
3. Selective insertion at P3/P4 stages to balance accuracy and parameter count.

## Repository Structure

- `experiments/`: Training and controlled comparison scripts.
- `third_party/ultralytics/`: Vendored Ultralytics code with SGCB model definitions.
- `configs/`: Dataset and model configuration files.
- `scripts/`: Dataset, validation, GFLOPs, export, and deployment utilities.
- `journal_submission/`: Manuscript files and canonical controlled validation summary.
- `results/`: Auto-generated experiment outputs.
- `downloaded_results/`: Locally downloaded result bundles; ignored by git.
- `data/`: Local datasets; ignored by git.

>>>>>>> d9952b10fbe0e2796efca6d623c4de029ea870ce
## Troubleshooting

- If CUDA runs out of memory, reduce `--batch` or use `--device cpu` for validation.
- Make sure your PyTorch build has CUDA support before using `--device 0`.
- On Windows, the scripts set `KMP_DUPLICATE_LIB_OK=TRUE` where needed to avoid OpenMP conflicts.
