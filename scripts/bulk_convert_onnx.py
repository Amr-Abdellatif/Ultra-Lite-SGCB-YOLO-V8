import os
import sys
from pathlib import Path

# Add third_party/ultralytics to path to ensure SGCB and other modules are found
repo_root = Path(os.getcwd())
local_ultralytics = str(repo_root / "third_party" / "ultralytics")
sys.path.insert(0, local_ultralytics)

os.environ['ULTRALYTICS_OFFLINE'] = 'True'
os.environ['ULTRALYTICS_CHECK_REPOS'] = 'False'

# Manually register custom modules into the ultralytics namespace to avoid pickling errors
# We must import from the local path we just added
import ultralytics.nn.modules.block as block
try:
    from ultralytics.nn.modules.block import SGCB
    # Ensure SGCB is available in the module where torch looks for it
    setattr(block, 'SGCB', SGCB)
    print("✅ Registered SGCB module")
except ImportError:
    print("⚠️ SGCB module not found in local ultralytics. Check paths.")

# Monkeypatch requirements check to avoid ONNX version issues
import ultralytics.utils.checks as checks
checks.check_requirements = lambda *args, **kwargs: True
print("✅ Monkeypatched check_requirements")

from ultralytics import YOLO

def bulk_convert_onnx(repo_root):
    print(f"\n🔍 Searching for experiment folders in: {repo_root}")
    
    # Find all directories starting with "Comparison_Exp"
    exp_dirs = [d for d in repo_root.iterdir() if d.is_dir() and d.name.startswith("Comparison_Exp")]
    
    if not exp_dirs:
        print("⚠️ No experiment folders found starting with 'Comparison_Exp'.")
        return

    for exp_dir in exp_dirs:
        # Determine image size from folder name (default to 640)
        imgsz = 384 if "img384" in exp_dir.name else 640
        
        print(f"\n" + "="*80)
        print(f"🚀 Processing Experiment Folder: {exp_dir.name}")
        print(f"   Inferred ONNX input resolution: {imgsz}px")
        print("="*80)
        
        # Target model subdirectories inside the experiment folder
        subdirs = [d for d in exp_dir.iterdir() if d.is_dir() and d.name != "ONNX_weights"]
        
        for subdir in subdirs:
            print(f"\n📂 Model: {subdir.name}")
            weights_dir = subdir / "weights"
            if not weights_dir.exists():
                print(f"   ⚠️ Weights directory not found, skipping.")
                continue
                
            # Find best.pt or last.pt
            model_pt = weights_dir / "best.pt"
            if not model_pt.exists():
                model_pt = weights_dir / "last.pt"
                
            if not model_pt.exists():
                print(f"   ⚠️ No .pt models found in {weights_dir}, skipping.")
                continue
                
            # Create ONNX_weights directory
            onnx_dir = subdir / "ONNX_weights"
            os.makedirs(onnx_dir, exist_ok=True)
            
            print(f"   📥 Loading model: {model_pt}")
            try:
                model = YOLO(str(model_pt))
                
                print(f"   ⚙️ Exporting to ONNX ({imgsz}px)...")
                # Export without simplify to avoid external dependencies
                onnx_path = model.export(format='onnx', imgsz=imgsz, opset=12)
                
                # Move result to ONNX_weights
                final_onnx = onnx_dir / Path(onnx_path).name
                if final_onnx.exists():
                    os.remove(final_onnx)
                os.rename(onnx_path, final_onnx)
                
                print(f"   ✅ Exported to: {final_onnx}")
                
            except Exception as e:
                print(f"   ❌ Export failed for {subdir.name}: {e}")

if __name__ == "__main__":
    bulk_convert_onnx(repo_root)
