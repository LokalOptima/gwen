"""
MTP (Multi-Token Prediction) head for Qwen3.5-0.8B.

Architecture matches the pre-trained MTP weights exactly:
    concat(RMSNorm(embed[t+1]), RMSNorm(hidden[t])) → FC(2048→1024)
    → 1 GQA layer (8Q+gate, 2KV, head_dim=256, QK-Norm)
    → SwiGLU FFN (1024→3584→1024)
    → RMSNorm → LM head (1024→K)

The LM head outputs over a restricted vocab of size K (not full 248K).
Pre-trained weights loaded from safetensors, lm_head initialized fresh.
"""

import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMSNorm with learnable scale. Weight initialized to 1 (identity)."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * rms).to(x.dtype) * self.weight


def apply_rope(x: torch.Tensor, positions: torch.Tensor,
               theta: float = 10000000.0, rope_dim: int = 64) -> torch.Tensor:
    """Apply interleaved RoPE to first rope_dim dims of each head.

    Args:
        x: [B, n_heads, L, head_dim]
        positions: [B, L] or [L] int positions
    Returns:
        x with RoPE applied to first rope_dim dimensions.
    """
    B, H, L, D = x.shape
    n_pairs = rope_dim // 2

    if positions.dim() == 1:
        positions = positions.unsqueeze(0).expand(B, -1)

    # freq[pair] = pos * theta^(-2*pair/rope_dim)
    pair_idx = torch.arange(n_pairs, device=x.device, dtype=torch.float32)
    freq_exp = -2.0 * pair_idx / rope_dim  # [n_pairs]
    inv_freq = theta ** freq_exp  # [n_pairs]

    # [B, L] x [n_pairs] → [B, L, n_pairs]
    freqs = positions.float().unsqueeze(-1) * inv_freq.unsqueeze(0).unsqueeze(0)
    cos_val = freqs.cos()  # [B, L, n_pairs]
    sin_val = freqs.sin()

    # Reshape for broadcast: [B, 1, L, n_pairs]
    cos_val = cos_val.unsqueeze(1)
    sin_val = sin_val.unsqueeze(1)

    # Interleaved pairs: (0,1), (2,3), ...
    x_rope = x[..., :rope_dim].reshape(B, H, L, n_pairs, 2)
    x0 = x_rope[..., 0]  # [B, H, L, n_pairs]
    x1 = x_rope[..., 1]

    r0 = x0 * cos_val - x1 * sin_val
    r1 = x0 * sin_val + x1 * cos_val

    rotated = torch.stack([r0, r1], dim=-1).reshape(B, H, L, rope_dim)
    out = x.clone()
    out[..., :rope_dim] = rotated
    return out


class GatedSelfAttention(nn.Module):
    """Qwen3.5-style gated GQA attention (attn_output_gate=True, with RoPE)."""

    def __init__(
        self,
        hidden_size: int = 1024,
        num_q_heads: int = 8,
        num_kv_heads: int = 2,
        head_dim: int = 256,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_kv_groups = num_q_heads // num_kv_heads

        # Q projection includes output gate: 2 * num_q_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, 2 * num_q_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(hidden_size, num_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(num_q_heads * head_dim, hidden_size, bias=False)

        # QK-Norm (shared across heads, applied per head_dim)
        self.q_norm = RMSNorm(head_dim, eps=eps)
        self.k_norm = RMSNorm(head_dim, eps=eps)

    def forward(self, x: torch.Tensor, positions: torch.Tensor | None = None) -> torch.Tensor:
        B, L, _ = x.shape
        h, d = self.num_q_heads, self.head_dim

        # Q + gate: interleaved as [Q_h0(256), G_h0(256), Q_h1(256), G_h1(256), ...]
        qg = self.q_proj(x)  # [B, L, 2*h*d = 4096]
        qg = qg.view(B, L, h, 2, d)  # [B, L, h, 2, d]
        q = qg[:, :, :, 0, :]     # [B, L, h, d]
        gate = qg[:, :, :, 1, :]   # [B, L, h, d]

        k = self.k_proj(x).view(B, L, self.num_kv_heads, d)
        v = self.v_proj(x).view(B, L, self.num_kv_heads, d)

        # QK-Norm
        q = self.q_norm(q)
        k = self.k_norm(k)

        # Expand KV for GQA (2 KV heads → 8 Q heads, ratio 4:1)
        if self.num_kv_groups > 1:
            k = k[:, :, :, None, :].expand(-1, -1, -1, self.num_kv_groups, -1)
            k = k.reshape(B, L, h, d)
            v = v[:, :, :, None, :].expand(-1, -1, -1, self.num_kv_groups, -1)
            v = v.reshape(B, L, h, d)

        # [B, heads, L, head_dim]
        q, k, v = (t.transpose(1, 2) for t in (q, k, v))

        # RoPE (interleaved, first 64 of 256 dims)
        if positions is None:
            positions = torch.arange(L, device=x.device)
        q = apply_rope(q, positions)
        k = apply_rope(k, positions)

        # Scaled dot-product attention with causal mask
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        # [B, L, heads * head_dim]
        attn_out = attn_out.transpose(1, 2).reshape(B, L, h * d)

        # Output gate: sigmoid(gate) * attention_output
        gate = torch.sigmoid(gate.reshape(B, L, h * d))
        attn_out = attn_out * gate

        return self.o_proj(attn_out)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int = 1024, intermediate_size: int = 3584):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class MTPHead(nn.Module):
    """Multi-Token Prediction head for restricted-vocab speculative decoding.

    At position t, predicts token[t+2] from:
        - embed[t+1]: token embedding at position t+1
        - hidden[t]: main model's last-layer hidden state at position t

    Output is over a restricted vocab of size K (mapped from full 248K vocab).
    """

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 1024,
        intermediate_size: int = 3584,
        num_q_heads: int = 8,
        num_kv_heads: int = 2,
        head_dim: int = 256,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        # Input norms
        self.pre_fc_norm_embedding = RMSNorm(hidden_size, eps=eps)
        self.pre_fc_norm_hidden = RMSNorm(hidden_size, eps=eps)

        # FC projection: concat(norm_embed, norm_hidden) → hidden
        self.fc = nn.Linear(hidden_size * 2, hidden_size, bias=False)

        # Single transformer layer
        self.input_layernorm = RMSNorm(hidden_size, eps=eps)
        self.self_attn = GatedSelfAttention(
            hidden_size, num_q_heads, num_kv_heads, head_dim, eps
        )
        self.post_attention_layernorm = RMSNorm(hidden_size, eps=eps)
        self.mlp = SwiGLU(hidden_size, intermediate_size)

        # Output
        self.norm = RMSNorm(hidden_size, eps=eps)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(
        self,
        embeddings: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            embeddings: [B, L, 1024] — token embeddings at positions t+1
            hidden_states: [B, L, 1024] — main model hidden states at positions t

        Returns:
            logits: [B, L, K] — logits over restricted vocab
        """
        # Normalize and concatenate
        x = torch.cat(
            [
                self.pre_fc_norm_embedding(embeddings),
                self.pre_fc_norm_hidden(hidden_states),
            ],
            dim=-1,
        )  # [B, L, 2048]

        # FC projection
        x = self.fc(x)  # [B, L, 1024]

        # Transformer layer with residuals
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attn(x)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        # Output
        x = self.norm(x)
        return self.lm_head(x)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        vocab_size: int,
        device: str = "cpu",
        vocab_ids: list[int] | None = None,
    ) -> "MTPHead":
        """Load pre-trained MTP weights from Qwen3.5-0.8B safetensors.

        If vocab_ids is provided (list of full-vocab token IDs for the restricted
        vocab), initializes lm_head from the corresponding rows of the pre-trained
        lm_head. Otherwise lm_head is randomly initialized.
        """
        from safetensors import safe_open

        model_dir = Path(model_dir)
        model = cls(vocab_size=vocab_size)

        # Find safetensors files
        index_path = model_dir / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                index = json.load(f)
            shard_files = sorted(set(index["weight_map"].values()))
            paths = [model_dir / f for f in shard_files]
        else:
            paths = sorted(model_dir.glob("*.safetensors"))

        # Load MTP tensors from safetensors
        mtp_tensors = {}
        for st_path in paths:
            with safe_open(str(st_path), framework="pt") as f:
                for name in f.keys():
                    # Handle both "model.mtp.X" and "mtp.X" naming
                    if "mtp." in name:
                        # Normalize to "mtp.X"
                        idx = name.index("mtp.")
                        clean = name[idx:]
                        mtp_tensors[clean] = f.get_tensor(name)

        if not mtp_tensors:
            raise ValueError(f"No MTP tensors found in {model_dir}")

        # Map safetensors names → our state dict keys
        # Safetensors: "mtp.layers.0.self_attn.q_proj.weight"
        # Our model:   "self_attn.q_proj.weight"
        name_map = {
            "mtp.fc.weight": "fc.weight",
            "mtp.pre_fc_norm_embedding.weight": "pre_fc_norm_embedding.weight",
            "mtp.pre_fc_norm_hidden.weight": "pre_fc_norm_hidden.weight",
            "mtp.layers.0.self_attn.q_proj.weight": "self_attn.q_proj.weight",
            "mtp.layers.0.self_attn.k_proj.weight": "self_attn.k_proj.weight",
            "mtp.layers.0.self_attn.v_proj.weight": "self_attn.v_proj.weight",
            "mtp.layers.0.self_attn.o_proj.weight": "self_attn.o_proj.weight",
            "mtp.layers.0.self_attn.q_norm.weight": "self_attn.q_norm.weight",
            "mtp.layers.0.self_attn.k_norm.weight": "self_attn.k_norm.weight",
            "mtp.layers.0.input_layernorm.weight": "input_layernorm.weight",
            "mtp.layers.0.post_attention_layernorm.weight": "post_attention_layernorm.weight",
            "mtp.layers.0.mlp.gate_proj.weight": "mlp.gate_proj.weight",
            "mtp.layers.0.mlp.up_proj.weight": "mlp.up_proj.weight",
            "mtp.layers.0.mlp.down_proj.weight": "mlp.down_proj.weight",
            "mtp.norm.weight": "norm.weight",
        }

        state_dict = model.state_dict()
        loaded = 0
        for st_name, our_name in name_map.items():
            if st_name not in mtp_tensors:
                print(f"  Warning: {st_name} not found in safetensors")
                continue
            tensor = mtp_tensors[st_name]
            # Qwen3.5 norm convention: stored weight w, applied as x * (1 + w)
            # Our RMSNorm applies x * weight, so load as (1 + w)
            if "norm" in our_name:
                state_dict[our_name] = tensor.float() + 1.0
            else:
                state_dict[our_name] = tensor
            loaded += 1

        # Initialize lm_head from pre-trained weights if vocab mapping provided.
        # Qwen3.5 ties lm_head = embed_tokens, and MTP shares it with the main model.
        # So we load from embed_tokens and slice to the restricted vocab.
        if vocab_ids is not None:
            # Find embedding tensor
            emb_keys = ["model.language_model.embed_tokens.weight", "model.embed_tokens.weight"]
            full_lm_head = None
            for st_path in paths:
                with safe_open(str(st_path), framework="pt") as f:
                    for ek in emb_keys:
                        if ek in f.keys():
                            full_lm_head = f.get_tensor(ek)
                            break
                if full_lm_head is not None:
                    break
            if full_lm_head is not None:
                import numpy as np
                ids = np.array(vocab_ids, dtype=np.int64)
                state_dict["lm_head.weight"] = full_lm_head[ids]  # [K, 1024]
                loaded += 1
                print(f"Loaded {loaded}/16 pre-trained MTP tensors (lm_head from embed_tokens top-{vocab_size} slice)")
            else:
                print(f"Loaded {loaded}/15 pre-trained MTP tensors (lm_head: embed_tokens not found, random init)")
        else:
            print(f"Loaded {loaded}/15 pre-trained MTP tensors (lm_head initialized randomly)")

        model.load_state_dict(state_dict, strict=False)
        # Keep FP32 for training (GradScaler needs FP32 gradients).
        # autocast handles FP16 forward pass. Export converts to FP16.
        return model.to(device)

    def param_count(self) -> dict[str, int]:
        """Count parameters by component."""
        groups = {
            "input_norms": 0,
            "fc": 0,
            "attention": 0,
            "ffn": 0,
            "output_norm": 0,
            "lm_head": 0,
        }
        for name, p in self.named_parameters():
            n = p.numel()
            if "pre_fc_norm" in name:
                groups["input_norms"] += n
            elif name.startswith("fc."):
                groups["fc"] += n
            elif "self_attn" in name or "input_layernorm" in name:
                groups["attention"] += n
            elif "mlp" in name or "post_attention" in name:
                groups["ffn"] += n
            elif name == "norm.weight":
                groups["output_norm"] += n
            elif "lm_head" in name:
                groups["lm_head"] += n
        groups["total"] = sum(groups.values())
        return groups
