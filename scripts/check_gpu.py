import sys
import torch
import traceback

def check_gpu():
    print(f"Python: {sys.version.split()[0]}")
    
    try:
        import torch
        print(f"PyTorch: {torch.__version__}")
        
        cuda_available = torch.cuda.is_available()
        print(f"CUDA availability: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA version reported by PyTorch: {torch.version.cuda}")
            print(f"GPU name: {torch.cuda.get_device_name(0)}")
            total_memory = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            print(f"GPU VRAM: {total_memory:.2f} GB")
        else:
            print("LOCAL GPU VALIDATION NOT AVAILABLE")
    except ImportError:
        print("PyTorch: NOT INSTALLED")
        
    try:
        import bitsandbytes
        print(f"BitsAndBytes availability: INSTALLED")
    except ImportError:
        print("BitsAndBytes availability: NOT INSTALLED")

if __name__ == "__main__":
    check_gpu()
