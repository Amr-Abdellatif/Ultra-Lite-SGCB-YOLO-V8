import os
import sys
import time
import csv
import argparse
import platform
from pathlib import Path
import torch
import numpy as np

# Ensure custom modules (SGCB) can be cleanly deserialized from checkpoints
repo_root = Path(__file__).resolve().parents[1]
local_ultralytics = str(repo_root / "third_party" / "ultralytics")
if local_ultralytics not in sys.path:
    sys.path.insert(0, local_ultralytics)

# Register custom module into the block namespace to satisfy PyTorch unpickling
try:
    import ultralytics.nn.modules.block as block
    from ultralytics.nn.modules.block import SGCB
    setattr(block, 'SGCB', SGCB)
except Exception:
    pass

from ultralytics import YOLO

def get_host_cpu_spec():
    """Extract full host CPU processor brand and core configuration."""
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        brand = info.get('brand_raw', platform.processor())
        return f"{brand} ({os.cpu_count()} Cores)"
    except Exception:
        proc = platform.processor()
        if not proc:
            proc = f"{platform.system()} {platform.machine()}"
        return f"{proc} ({os.cpu_count()} Cores)"

def evaluate_cpu_edge(repo_root, samples=50):
    cpu_spec = get_host_cpu_spec()
    print("\n" + "="*100)
    print("🖥️  EVALUATING EDGE DEPLOYMENT ON CPU: PyTorch (.pt) vs ONNX (.onnx)")
    print(f"    Host Hardware Spec: {cpu_spec}")
    print("="*100)
    
    # Locate all outcome directories
    exp_dirs = sorted([d for d in repo_root.iterdir() if d.is_dir() and d.name.startswith("Comparison_Exp")])
    
    if not exp_dirs:
        print("⚠️ No experiment folders found starting with 'Comparison_Exp'.")
        return

    results_dir = repo_root / "results"
    os.makedirs(results_dir, exist_ok=True)
    
    all_metrics = []
    
    for exp_dir in exp_dirs:
        # Determine inference input size based on folder context
        imgsz = 384 if "img384" in exp_dir.name else 640
        
        print(f"\n📁 Scanning Experiment Suite: {exp_dir.name} (Input Size: {imgsz}px)")
        
        # Target individual model checkpoints
        model_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir() and d.name != "ONNX_weights"])
        
        for mdir in model_dirs:
            weights_pt = mdir / "weights" / "best.pt"
            if not weights_pt.exists():
                weights_pt = mdir / "weights" / "last.pt"
                
            weights_onnx = mdir / "ONNX_weights" / "best.onnx"
            
            if not weights_pt.exists():
                print(f"   ⚠️ Skipping {mdir.name}: PyTorch weights not found.")
                continue
                
            print(f"\n   ⚙️ Benchmarking Model: {mdir.name}")
            
            # Use a dummy numpy image array for unified inference measurements across backends
            dummy_img = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)
            
            # --- 1. PyTorch (.pt) Inference Profiling ---
            pt_latency_ms, pt_fps = 0.0, 0.0
            params, flops = 0.0, 0.0
            try:
                print(f"      📥 Loading PyTorch model: {weights_pt.relative_to(repo_root)}")
                model_pt = YOLO(str(weights_pt))
                model_pt.to('cpu')
                
                # Extract complexity stats
                try:
                    info_res = model_pt.info(imgsz=imgsz, verbose=False)
                    params = info_res[1] / 1e6
                    flops = info_res[3]
                except Exception:
                    params = sum(p.numel() for p in model_pt.model.parameters()) / 1e6
                    flops = 0.0
                    
                # Warmup
                for _ in range(3):
                    model_pt(dummy_img, verbose=False)
                    
                # Timed inference
                start_time = time.time()
                for _ in range(samples):
                    model_pt(dummy_img, verbose=False)
                total_time = time.time() - start_time
                pt_latency_ms = (total_time / samples) * 1000.0
                pt_fps = 1000.0 / pt_latency_ms
                print(f"      ✅ PyTorch (.pt) Latency: {pt_latency_ms:.2f} ms | Throughput: {pt_fps:.2f} FPS")
            except Exception as e:
                print(f"      ❌ PyTorch benchmark failed: {e}")

            # --- 2. ONNX (.onnx) Inference Profiling ---
            onnx_latency_ms, onnx_fps = 0.0, 0.0
            if weights_onnx.exists():
                try:
                    print(f"      📥 Loading ONNX model: {weights_onnx.relative_to(repo_root)}")
                    model_onnx = YOLO(str(weights_onnx), task='detect')
                    
                    # Warmup
                    for _ in range(3):
                        model_onnx(dummy_img, verbose=False)
                        
                    # Timed inference
                    start_time = time.time()
                    for _ in range(samples):
                        model_onnx(dummy_img, verbose=False)
                    total_time = time.time() - start_time
                    onnx_latency_ms = (total_time / samples) * 1000.0
                    onnx_fps = 1000.0 / onnx_latency_ms
                    print(f"      ✅ ONNX (.onnx) Latency:  {onnx_latency_ms:.2f} ms | Throughput: {onnx_fps:.2f} FPS")
                except Exception as e:
                    print(f"      ❌ ONNX benchmark failed: {e}")
            else:
                print(f"      ⚠️ ONNX model not found at {weights_onnx.relative_to(repo_root)}")

            all_metrics.append({
                'Experiment': exp_dir.name.split('_')[0] + '_' + exp_dir.name.split('_')[1],
                'Model_Name': mdir.name,
                'Resolution': imgsz,
                'Params_M': round(params, 3),
                'GFLOPs': round(flops, 2),
                'PT_Latency_ms': round(pt_latency_ms, 2),
                'PT_FPS': round(pt_fps, 2),
                'ONNX_Latency_ms': round(onnx_latency_ms, 2),
                'ONNX_FPS': round(onnx_fps, 2),
                'Host_CPU_Spec': cpu_spec
            })

    # Save to consolidated summary CSV
    if all_metrics:
        csv_path = results_dir / "edge_deployment_pt_vs_onnx_summary.csv"
        fields = ['Experiment', 'Model_Name', 'Resolution', 'Params_M', 'GFLOPs', 'PT_Latency_ms', 'PT_FPS', 'ONNX_Latency_ms', 'ONNX_FPS', 'Host_CPU_Spec']
        
        with open(csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_metrics)
            
        print("\n" + "="*115)
        print("📊 CONSOLIDATED CPU DEPLOYMENT SUMMARY: PyTorch vs ONNX")
        print("="*115)
        header = f"{'Experiment':<15} | {'Model Name':<25} | {'Res':<5} | {'Params':<8} | {'PT Latency':<11} | {'PT FPS':<7} | {'ONNX Lat':<10} | {'ONNX FPS':<8}"
        print(header)
        print("-" * len(header))
        for row in all_metrics:
            pt_lat = f"{row['PT_Latency_ms']:.1f}ms" if row['PT_Latency_ms'] else "N/A"
            onnx_lat = f"{row['ONNX_Latency_ms']:.1f}ms" if row['ONNX_Latency_ms'] else "N/A"
            print(f"{row['Experiment']:<15} | {row['Model_Name']:<25} | {row['Resolution']:<5} | {row['Params_M']:<7.2f}M | {pt_lat:>10} | {row['PT_FPS']:>6.1f} | {onnx_lat:>9} | {row['ONNX_FPS']:>7.1f}")
        print("="*115)
        print(f"📁 Detailed side-by-side comparison CSV written to: {csv_path.relative_to(repo_root)}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate PyTorch vs ONNX Inference on CPU")
    parser.add_argument('--samples', type=int, default=20, help="Number of timed inference frames per model")
    args = parser.parse_args()
    
    evaluate_cpu_edge(repo_root, samples=args.samples)

if __name__ == '__main__':
    main()
