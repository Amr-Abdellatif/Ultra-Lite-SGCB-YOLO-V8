import os
import argparse
from ultralytics import YOLO
from pathlib import Path

def run_ablation():
    parser = argparse.ArgumentParser(description='Ultra-Lite YOLO Ablation Study')
    parser.add_argument('--data', type=str, default='configs/VisDrone.yaml', help='Dataset config')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs for short test')
    parser.add_argument('--batch', type=int, default=16, help='Batch size')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    parser.add_argument('--device', type=str, default='0', help='Device')
    
    args = parser.parse_args()
    
    repo_root = Path(__file__).resolve().parents[1]
    cfg_dir = repo_root / "third_party" / "ultralytics" / "ultralytics" / "cfg" / "models" / "v8"
    
    # 4 Variants for Ablation Study
    variants = [
        {
            'name': 'A_Baseline',
            'cfg': str(cfg_dir / 'yolov8n.yaml'),
            'desc': 'Standard YOLOv8n'
        },
        {
            'name': 'B_Baseline_P2',
            'cfg': str(cfg_dir / 'yolov8-p2.yaml'),
            'desc': 'Standard YOLOv8 with P2 High-Res Head'
        },
        {
            'name': 'C_Baseline_SGCB',
            'cfg': str(repo_root / 'configs' / 'yolov8n-sgcb-only.yaml'),
            'desc': 'YOLOv8 with SGCB (Selective Replacement) - No P2'
        },
        {
            'name': 'D_UltraLite_Full',
            'cfg': str(cfg_dir / 'yolov8n-ultraliteattn-local.yaml'),
            'desc': 'Full Ultra-Lite (SGCB + P2 Head + Selective Strategy)'
        }
    ]
    
    # Create C_Baseline_SGCB config if it doesn't exist
    create_sgcb_only_config(repo_root)
    
    # Setup Output Directory
    output_dir = repo_root / "results" / "ablation"
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*80)
    print("🧪 STARTING ABLATION STUDY")
    print(f"📁 Results will be saved to: {output_dir}")
    print("="*80)
    
    results = []
    for var in variants:
        print(f"\n▶️ Running {var['name']} ({var['desc']})")
        model = YOLO(var['cfg'])
        
        # Train for short period to verify
        res = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch,
            imgsz=args.imgsz,
            device=args.device,
            project=str(output_dir),
            name=var['name'],
            exist_ok=True,
            plots=True
        )
        
        # Collect Metrics
        results.append({
            'variant': var['name'],
            'params': model.info()[1] / 1e6,
            'mAP50': res.results_dict.get('metrics/m_ap50', 0),
            'mAP50-95': res.results_dict.get('metrics/m_ap50_95', 0)
        })
        
    # Summary Table
    print("\n" + "="*80)
    print("📊 ABLATION SUMMARY")
    print("="*80)
    print(f"{'Variant':<20} | {'Params (M)':<10} | {'mAP50':<10}")
    print("-" * 45)
    for res in results:
        print(f"{res['variant']:<20} | {res['params']:<10.2f} | {res['mAP50']:<10.4f}")

def create_sgcb_only_config(repo_root):
    """Creates a config with SGCB but NO P2 head."""
    cfg_path = repo_root / 'configs' / 'yolov8n-sgcb-only.yaml'
    if cfg_path.exists():
        return
        
    content = """# YOLOv8 with SGCB Only (No P2 Head)
nc: 10
scales:
  n: [0.33, 0.25, 1024]

backbone:
  - [-1, 1, Conv, [64, 3, 2]]          # 0-P1/2
  - [-1, 1, Conv, [128, 3, 2]]         # 1-P2/4
  - [-1, 3, C2f, [128, True]]          # 2
  - [-1, 1, Conv, [256, 3, 2]]         # 3-P3/8
  - [-1, 6, SGCB, [256, True, 1, 0.5, 7]] # 4
  - [-1, 1, Conv, [512, 3, 2]]         # 5-P4/16
  - [-1, 6, SGCB, [512, True, 1, 0.5, 7]] # 6
  - [-1, 1, Conv, [1024, 3, 2]]        # 7-P5/32
  - [-1, 3, C2f, [1024, True]]         # 8
  - [-1, 1, SPPF, [1024, 5]]           # 9

head:
  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 6], 1, Concat, [1]]
  - [-1, 3, SGCB, [512]] # 12

  - [-1, 1, nn.Upsample, [None, 2, 'nearest']]
  - [[-1, 4], 1, Concat, [1]]
  - [-1, 3, SGCB, [256]] # 15

  - [-1, 1, Conv, [256, 3, 2]]
  - [[-1, 12], 1, Concat, [1]]
  - [-1, 3, SGCB, [512]] # 18

  - [-1, 1, Conv, [512, 3, 2]]
  - [[-1, 9], 1, Concat, [1]]
  - [-1, 3, C2f, [1024]] # 21

  - [[15, 18, 21], 1, Detect, [nc]]
"""
    os.makedirs(repo_root / 'configs', exist_ok=True)
    with open(cfg_path, 'w') as f:
        f.write(content)

if __name__ == '__main__':
    run_ablation()
