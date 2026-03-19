"""Test if flash-linear-attention works for Qwen3.5 forward pass."""
import torch
print(f"PyTorch {torch.__version__}, CUDA {torch.version.cuda}")

# Check if fla is detected by transformers
try:
    import fla
    print(f"fla: {fla.__version__}")
except ImportError:
    print("fla: NOT installed")

try:
    import causal_conv1d
    print(f"causal_conv1d: {causal_conv1d.__version__}")
except ImportError:
    print("causal_conv1d: NOT installed")

# Try loading the model
from transformers import AutoModelForCausalLM
print("\nLoading Qwen3.5-0.8B...")
model = AutoModelForCausalLM.from_pretrained(
    "/home/lapo/models/hf/Qwen3.5-0.8B",
    dtype=torch.bfloat16,
    trust_remote_code=True,
).cuda()
model.eval()

# Test forward
for batch, seq in [(1, 512), (4, 512), (16, 512), (32, 512), (64, 512)]:
    torch.cuda.reset_peak_memory_stats()
    print(f"Running forward pass (batch={batch}, seq={seq})...")
    ids = torch.randint(0, 1000, (batch, seq), device="cuda")
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model.model(ids, use_cache=False)
        h = out.last_hidden_state.clone()
        del out
        torch.cuda.empty_cache()
        print(f"  Hidden shape: {h.shape}")

    mem = torch.cuda.max_memory_allocated() / 1e9
    print(f"  Peak GPU memory: {mem:.2f} GB")
    del h, ids
    torch.cuda.empty_cache()

print("SUCCESS")
