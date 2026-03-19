import torch
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}, SM {torch.cuda.get_device_capability(0)}")
import importlib
for pkg in ["triton", "fla", "causal_conv1d"]:
    try:
        m = importlib.import_module(pkg)
        print(f"  {pkg}: {getattr(m, '__version__', 'installed')}")
    except ImportError:
        print(f"  {pkg}: NOT installed")
