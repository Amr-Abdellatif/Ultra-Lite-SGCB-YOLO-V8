# SGCB Ultra-Lite YOLOv8 Experiments

Standalone reproducibility repository for **Ultra-Lite YOLOv8 with Spatially-Gated Context Blocks (SGCB)** for aerial tiny-object detection on VisDrone2019-DET.

#DOI
(https://doi.org/10.5281/zenodo.20157395)

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
| UltraLite-X | 640 | 41.95 | 199.46 | 59.4 | 49.4 | 51.0 | 31.9 |

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
- `configs/`: Dataset and model configuration files.
- `scripts/`: Dataset, validation, GFLOPs, export, and deployment utilities.
- `journal_submission/`: Manuscript files and canonical controlled validation summary.
- `results/`: Auto-generated experiment outputs.
- `downloaded_results/`: Locally downloaded result bundles; ignored by git.
- `data/`: Local datasets; ignored by git.

## Troubleshooting

- If CUDA runs out of memory, reduce `--batch` or use `--device cpu` for validation.
- Make sure your PyTorch build has CUDA support before using `--device 0`.
- On Windows, the scripts set `KMP_DUPLICATE_LIB_OK=TRUE` where needed to avoid OpenMP conflicts.
