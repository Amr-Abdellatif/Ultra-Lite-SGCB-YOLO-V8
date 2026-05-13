import time
import torch
import argparse
from ultralytics import YOLO
from ultralytics.utils.torch_utils import select_device
from copy import deepcopy

def benchmark_model(model_path, imgsz=640, device='0', half=True, num_samples=100):
    """
    Benchmark a model for Parameters, FLOPs, and Inference Speed.
    """
    print(f"\n🚀 Benchmarking Model: {model_path}")
    print(f"   Image Size: {imgsz}")
    print(f"   Device: {device}")
    
    # Select device
    device = select_device(device)
    
    # Load model
    model = YOLO(model_path)
    model.to(device)
    
    # 1. Calculate Parameters and FLOPs
    print("\n📊 Calculating Model Complexity...")
    # Get model info
    # results is a list [layers, params, gradients, flops]
    try:
        results = model.info(imgsz=imgsz, verbose=False)
        layers, params, gradients, flops = results
    except Exception as e:
        print(f"⚠️ Could not get model info via .info(): {e}")
        params = sum(p.numel() for p in model.model.parameters())
        flops = 0 # Placeholder if info() fails
        layers = 0

    print(f"   - Layers: {layers}")
    print(f"   - Parameters: {params/1e6:.2f} M")
    print(f"   - FLOPs: {flops:.2f} G")
    
    # 2. Benchmark Inference Speed
    print(f"\n⚡ Benchmarking Inference Speed ({num_samples} samples)...")
    
    # Warmup
    dummy_input = torch.randn(1, 3, imgsz, imgsz).to(device)
    if half and device.type != 'cpu':
        dummy_input = dummy_input.half()
        model.model.half()
    
    # Warmup runs
    for _ in range(20):
        model(dummy_input, verbose=False)
    
    # Synchronize if using CUDA
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    # Measure
    start_time = time.time()
    for _ in range(num_samples):
        model(dummy_input, verbose=False)
        
    if device.type == 'cuda':
        torch.cuda.synchronize()
        
    end_time = time.time()
    
    total_time = end_time - start_time
    avg_latency = (total_time / num_samples) * 1000  # ms
    fps = 1000 / avg_latency
    
    print(f"   - Avg Latency: {avg_latency:.2f} ms")
    print(f"   - Throughput: {fps:.2f} FPS")
    print(f"   - Hardware: {torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'}")
    
    return {
        'model': model_path,
        'params_m': params/1e6,
        'flops_g': flops,
        'latency_ms': avg_latency,
        'fps': fps,
        'hardware': torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'
    }

def main():
    parser = argparse.ArgumentParser(description='Edge Deployment Benchmark')
    parser.add_argument('--model', type=str, required=True, help='Path to model weights (.pt)')
    parser.add_argument('--imgsz', type=int, default=640, help='Inference image size')
    parser.add_argument('--device', type=str, default='0', help='Device (0, 1, cpu)')
    parser.add_argument('--samples', type=int, default=100, help='Number of samples for benchmarking')
    parser.add_argument('--no-half', action='store_true', help='Disable FP16 inference')
    
    args = parser.parse_args()
    
    benchmark_model(
        model_path=args.model,
        imgsz=args.imgsz,
        device=args.device,
        half=not args.no_half,
        num_samples=args.samples
    )

if __name__ == '__main__':
    main()
