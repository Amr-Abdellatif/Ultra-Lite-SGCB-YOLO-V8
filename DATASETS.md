# 📊 Dataset Preparation: VisDrone2019-DET

The experiments in this research are primarily conducted using the **VisDrone2019-DET** dataset. This guide explains how to access and prepare the data.

## 1. Automated Download (Recommended)
We provide a Python script that automatically downloads the dataset from Roboflow (via the Universe API) and prepares it in the correct YOLO format.

```powershell
python .\scripts\download_visdrone.py
```

**Note:** This script requires a Roboflow API Key. You can set it in a `.env` file:
```text
ROBOFLOW_API_KEY=your_key_here
```

## 2. Manual Download
If you prefer to download the data manually, follow these steps:

1. Visit the [VisDrone Roboflow Page](https://universe.roboflow.com/uogolanrewaju/visdrone2019-det).
2. Download the dataset in **YOLOv8** format.
3. Extract the contents into the `data/` directory of this repository.
4. The structure should look like:
   ```text
   data/
   └── visdrone/
       ├── train/
       ├── val/
       └── data.yaml
   ```

## 3. Dataset Characteristics
- **Total Images:** 10,209 (6,471 train / 548 val / 3,190 test)
- **Object Categories:** 10 classes (pedestrian, person, car, van, bus, truck, motor, bicycle, awning-tricycle, and tricycle)
- **Primary Challenge:** Extreme scale variation and high object density.

## 4. Evaluation Metrics
We use the standard COCO evaluation metrics:
- **mAP50**: Mean Average Precision at IoU=0.50 (Primary metric for detection).
- **mAP50-95**: Mean Average Precision averaged over IoU thresholds from 0.50 to 0.95.
