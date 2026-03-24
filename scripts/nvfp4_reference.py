"""Reference inference for Qwen3.5-4B-NVFP4: dequantize FP4→BF16, run through HF model.

Loads the NVFP4 safetensors checkpoint, manually dequantizes all FP4 weights
back to BF16 using the formula: w = fp4_val * block_scale_e4m3 * global_scale_f32,
then injects them into the standard HF Qwen3.5 model and runs inference.

This gives us a ground-truth reference for what the NVFP4 checkpoint should produce.
"""
import torch
import torch.nn.functional as F
import numpy as np
from safetensors import safe_open
from pathlib import Path
import json
import struct
import argparse

# FP4 E2M1 lookup table (16 values, unsigned)
# E2M1: 1 sign bit handled separately, 2 exponent bits, 1 mantissa bit
# Values: 0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, 0.5, 1, 1.5, 2, 3, 4, 6
# (repeated for sign=0 and sign=1, but we handle sign separately)
FP4_LUT = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,   # positive (sign=0)
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,  # negative (sign=1)
], dtype=torch.float32)


def dequantize_fp4_tensor(data: torch.Tensor, scales: torch.Tensor,
                           global_scale: float, shape: list) -> torch.Tensor:
    """Dequantize a packed FP4 tensor to float32.

    Args:
        data: uint8 packed FP4 data [out_features, in_features/2]
        scales: uint8 E4M3 block scales [out_features, in_features/16]
        global_scale: float32 per-tensor scale
        shape: [out_features, in_features] (unpacked shape)
    """
    out_features = shape[0]
    in_features = shape[1]

    # Unpack FP4: low nibble = even element, high nibble = odd element
    low_nibbles = (data.to(torch.int32) & 0x0F).long()
    high_nibbles = (data.to(torch.int32) >> 4).long()

    unpacked = torch.zeros(out_features, in_features, dtype=torch.float32)
    unpacked[:, 0::2] = FP4_LUT[low_nibbles]
    unpacked[:, 1::2] = FP4_LUT[high_nibbles]

    # Convert E4M3 scales using PyTorch native float8_e4m3fn
    scale_floats = scales.view(torch.float8_e4m3fn).float()

    # Each scale covers 16 consecutive elements in the in_features dimension
    # scales shape: [out_features, in_features/16]
    # Use repeat_interleave for vectorized application
    scale_expanded = scale_floats.reshape(out_features, -1).repeat_interleave(16, dim=1)
    # Trim in case in_features is not perfectly divisible
    scale_expanded = scale_expanded[:, :in_features]

    unpacked *= scale_expanded
    unpacked *= global_scale

    return unpacked


DEQUANT_CACHE = Path("~/models/Qwen3.5-4B-NVFP4-dequant.safetensors").expanduser()


def load_and_dequantize(model_path: str):
    """Load NVFP4 safetensors and dequantize all weights to BF16.

    Caches the dequantized weights to DEQUANT_CACHE so subsequent runs skip dequantization.
    """
    model_dir = Path(model_path)

    # Load config (always needed)
    with open(model_dir / "config.json") as f:
        config = json.load(f)
    text_cfg = config.get("text_config", config)
    n_layers = text_cfg["num_hidden_layers"]

    # Check for cached dequantized weights
    if DEQUANT_CACHE.exists():
        print(f"Loading cached dequantized weights from {DEQUANT_CACHE}")
        from safetensors.torch import load_file
        weights = load_file(str(DEQUANT_CACHE))
        print(f"Loaded {len(weights)} tensors from cache")
        return weights, config

    print(f"Model: {n_layers} layers, hidden_size={text_cfg['hidden_size']}")

    # Open safetensors
    st_files = sorted(model_dir.glob("model*.safetensors"))
    print(f"Found {len(st_files)} safetensors file(s)")

    weights = {}  # HF tensor name -> torch tensor (float32 or bf16)

    for st_path in st_files:
        with safe_open(str(st_path), framework="pt") as f:
            keys = list(f.keys())

            for k in sorted(keys):
                # Skip visual encoder
                if "visual" in k:
                    continue
                # Skip scale/input_scale (handled with base weight)
                if k.endswith(".weight_scale") or k.endswith(".weight_scale_2") or k.endswith(".input_scale"):
                    continue

                # Bare tensors (A_log, dt_bias, etc.)
                if not k.endswith(".weight") and not k.endswith("_scale") and not k.endswith("_scale_2"):
                    t = f.get_tensor(k)
                    weights[k] = t.float()
                    print(f"  F32 bare: {k} {list(t.shape)}")
                    continue

                if k.endswith(".weight"):
                    base = k[:-7]
                    t = f.get_tensor(k)
                    scale_key = base + ".weight_scale"
                    scale2_key = base + ".weight_scale_2"

                    if scale_key in keys:
                        # FP4 quantized — dequantize
                        scales = f.get_tensor(scale_key)
                        scale2 = float(f.get_tensor(scale2_key).item())

                        data = t.squeeze().contiguous()
                        packed_shape = list(data.shape)
                        real_shape = [packed_shape[0], packed_shape[1] * 2]

                        dequantized = dequantize_fp4_tensor(
                            data.to(torch.uint8),
                            scales.contiguous().view(torch.uint8),
                            scale2,
                            real_shape
                        )
                        weights[k] = dequantized.bfloat16()
                        print(f"  FP4→BF16: {k} {real_shape} "
                              f"scales={list(scales.shape)} scale2={scale2:.6f} "
                              f"range=[{dequantized.min():.4f}, {dequantized.max():.4f}]")
                    else:
                        # Non-quantized
                        weights[k] = t
                        print(f"  Native: {k} {list(t.shape)} dtype={t.dtype}")

    print(f"\nDequantized {len(weights)} tensors")

    # Save cache
    from safetensors.torch import save_file
    print(f"Saving dequantized cache to {DEQUANT_CACHE}...")
    save_file(weights, str(DEQUANT_CACHE))
    print(f"Saved ({DEQUANT_CACHE.stat().st_size / 1024 / 1024:.0f} MB)")

    return weights, config


def inject_weights(model, weights):
    """Inject dequantized weights into HF model, handling the (1+weight) norm convention."""
    state_dict = model.state_dict()
    injected = 0
    skipped = []

    for name, param in state_dict.items():
        # Map from model state_dict name to checkpoint name
        # HF model uses "model.layers.X...." but checkpoint uses "model.language_model.layers.X...."
        # Try both prefixes
        candidates = [name]
        if name.startswith("model."):
            candidates.append("model.language_model." + name[6:])

        found = False
        for ck in candidates:
            if ck in weights:
                w = weights[ck]
                if w.shape != param.shape:
                    print(f"  Shape mismatch: {name} model={param.shape} ckpt={w.shape}")
                    skipped.append(name)
                else:
                    param.data.copy_(w.to(param.dtype))
                    injected += 1
                found = True
                break

        if not found:
            skipped.append(name)

    print(f"Injected {injected} tensors, skipped {len(skipped)}")
    if skipped:
        print(f"Skipped (first 10): {skipped[:10]}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default="/home/lapo/models/Qwen3.5-4B-NVFP4",
                        help="Path to NVFP4 checkpoint")
    parser.add_argument("--prompt", default="What is the capital of France?",
                        help="Prompt to test")
    parser.add_argument("--max-tokens", type=int, default=100)
    args = parser.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
    from pathlib import Path

    # Build config + tokenizer (always needed)
    with open(Path(args.model_path) / "config.json") as f:
        config = json.load(f)
    text_cfg = config.get("text_config", config)
    text_cfg["model_type"] = "qwen3_5"
    if "vocab_size" not in text_cfg:
        text_cfg["vocab_size"] = 248320
    text_cfg["architectures"] = ["Qwen3_5ForCausalLM"]
    text_cfg["tie_word_embeddings"] = True
    hf_config = AutoConfig.for_model(**text_cfg)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    # Fast path: load pre-saved full model
    model_cache = Path("~/models/Qwen3.5-4B-NVFP4-bf16.pt").expanduser()
    if model_cache.exists():
        print(f"Loading cached model from {model_cache}...")
        model = torch.load(str(model_cache), map_location="cuda", weights_only=False)
        model.eval()
    else:
        print("Dequantizing FP4 weights...")
        weights, _ = load_and_dequantize(args.model_path)

        model = AutoModelForCausalLM.from_config(
            hf_config, dtype=torch.bfloat16, trust_remote_code=True)
        model = model.to("cuda")
        model.eval()
        model = inject_weights(model, weights)

        # Save full model for fast loading next time
        print(f"Saving model to {model_cache}...")
        torch.save(model.cpu(), str(model_cache))
        model = model.to("cuda")
        print(f"Saved ({model_cache.stat().st_size / 1024 / 1024:.0f} MB)")

    print("\n" + "=" * 60)
    print("Step 3: Run inference")
    print("=" * 60)

    # ChatML format
    prompt = (f"<|im_start|>user\n{args.prompt}<|im_end|>\n"
              f"<|im_start|>assistant\n<think>\n\n</think>\n\n")
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(f"Prompt: {repr(prompt)}")
    print(f"Input tokens: {inputs.input_ids.shape}")

    # Greedy decode
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=False,
        )

    generated = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:],
                                  skip_special_tokens=True)
    print(f"\n=== Generated (greedy) ===")
    print(generated)

    # Also try with temperature
    with torch.no_grad():
        outputs2 = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )

    generated2 = tokenizer.decode(outputs2[0][inputs.input_ids.shape[1]:],
                                   skip_special_tokens=True)
    print(f"\n=== Generated (temp=0.7) ===")
    print(generated2)


if __name__ == "__main__":
    main()
