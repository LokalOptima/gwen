#!/usr/bin/env python3
"""
Analyze theoretical MTP accuracy with different restricted vocab sizes.

For each K (restricted vocab size), computes:
  1. Token coverage: P(token ∈ top-K)
  2. Pair coverage: P(token[t] ∈ top-K AND token[t+1] ∈ top-K) — relevant for MTP
  3. Consecutive run lengths before hitting OOV (how many accepts before forced reject)
  4. Theoretical MTP ceiling = pair coverage (both draft token and verification must be in-vocab)
  5. Expected throughput gain at various actual-accuracy levels

Uses pre-cached tokenized data from prepare_training_data.py.

Usage:
    uv run --with numpy scripts/analyze_restricted_vocab.py
    uv run --with numpy scripts/analyze_restricted_vocab.py --data data/train_tokens.bin
"""

import argparse
from pathlib import Path

import numpy as np

VOCAB_SIZE = 248320
SENTINEL = 0xFFFFFFFF
DATA_DIR = Path(__file__).parent.parent / "data"


def load_top_k_sets(counts_path: Path, k_values: list[int]) -> dict[int, set]:
    """Build top-K token sets from frequency counts."""
    counts = np.fromfile(str(counts_path), dtype=np.int64)
    if len(counts) < VOCAB_SIZE:
        counts = np.pad(counts, (0, VOCAB_SIZE - len(counts)))
    sorted_ids = np.argsort(-counts)
    return {k: set(sorted_ids[:k].tolist()) for k in k_values}


def load_top_k_masks(counts_path: Path, k_values: list[int]) -> dict[int, np.ndarray]:
    """Build boolean masks for top-K tokens (faster than sets for vectorized ops)."""
    counts = np.fromfile(str(counts_path), dtype=np.int64)
    if len(counts) < VOCAB_SIZE:
        counts = np.pad(counts, (0, VOCAB_SIZE - len(counts)))
    sorted_ids = np.argsort(-counts)
    masks = {}
    for k in k_values:
        mask = np.zeros(VOCAB_SIZE, dtype=bool)
        mask[sorted_ids[:k]] = True
        masks[k] = mask
    return masks


def analyze_sequences(tokens: np.ndarray, masks: dict[int, np.ndarray],
                      k_values: list[int]) -> dict:
    """Analyze token sequences for restricted vocab statistics."""
    # Remove sentinels, split into sequences
    is_sentinel = tokens == SENTINEL
    valid_mask = ~is_sentinel

    # For coverage stats, use all valid tokens
    valid_tokens = tokens[valid_mask]
    n_valid = len(valid_tokens)
    print(f"Total valid tokens: {n_valid:,}")

    results = {}

    for k in k_values:
        mask = masks[k]

        # 1. Token coverage: P(token ∈ top-K)
        in_vocab = mask[valid_tokens]
        coverage = in_vocab.mean()

        # 2. Pair coverage: P(token[t] AND token[t+1] both ∈ top-K)
        # This is what matters for MTP: the draft (token[t+1]) and the actual
        # next-next token (token[t+2]) both need to be predictable
        pair_both = in_vocab[:-1] & in_vocab[1:]
        pair_coverage = pair_both.mean()

        # 3. Triplet coverage: P(token[t], token[t+1], token[t+2] all ∈ top-K)
        triplet = in_vocab[:-2] & in_vocab[1:-1] & in_vocab[2:]
        triplet_coverage = triplet.mean()

        # 4. Consecutive run lengths (in-vocab streaks)
        # Find boundaries between sequences (sentinels)
        # Work on full token array including sentinels
        in_vocab_full = np.zeros(len(tokens), dtype=bool)
        in_vocab_full[valid_mask] = mask[tokens[valid_mask]]
        # Sentinels break runs
        in_vocab_full[is_sentinel] = False

        # Compute run lengths of consecutive in-vocab tokens
        changes = np.diff(in_vocab_full.astype(np.int8))
        run_starts = np.where(changes == 1)[0] + 1  # transitions 0→1
        run_ends = np.where(changes == -1)[0] + 1    # transitions 1→0
        # Handle edge cases
        if in_vocab_full[0]:
            run_starts = np.concatenate([[0], run_starts])
        if in_vocab_full[-1]:
            run_ends = np.concatenate([run_ends, [len(in_vocab_full)]])
        run_lengths = run_ends[:len(run_starts)] - run_starts[:len(run_ends)]

        if len(run_lengths) > 0:
            avg_run = run_lengths.mean()
            median_run = np.median(run_lengths)
            p90_run = np.percentile(run_lengths, 90)
            max_run = run_lengths.max()
        else:
            avg_run = median_run = p90_run = max_run = 0

        results[k] = {
            "coverage": coverage,
            "pair_coverage": pair_coverage,
            "triplet_coverage": triplet_coverage,
            "avg_run": avg_run,
            "median_run": median_run,
            "p90_run": p90_run,
            "max_run": max_run,
            "n_runs": len(run_lengths),
        }

    return results


def estimate_throughput(coverage, mtp_accuracy_within_vocab, base_tps=599,
                        mtp_overhead_ms=0.15, forward_ms=1.68):
    """Estimate throughput with restricted-vocab MTP.

    Args:
        coverage: P(next-next token ∈ top-K)
        mtp_accuracy_within_vocab: P(MTP correct | token ∈ top-K)
        base_tps: baseline tokens/sec without MTP
        mtp_overhead_ms: time for MTP forward pass
        forward_ms: time for main model forward pass
    """
    # Effective acceptance rate
    accept_rate = coverage * mtp_accuracy_within_vocab

    # Time per speculation cycle:
    # Accept: forward_2tok (≈ forward_ms * 1.05) + mtp_overhead → 2 tokens
    # Reject: forward_2tok + undo (0.04ms) + mtp_overhead → 1 token
    t_2tok = forward_ms * 1.05  # forward_2tok is ~5% more than single forward
    t_accept = t_2tok + mtp_overhead_ms  # ms for 2 tokens
    t_reject = t_2tok + 0.04 + mtp_overhead_ms  # ms for 1 token

    # Expected tokens per cycle
    tokens_per_cycle = accept_rate * 2 + (1 - accept_rate) * 1
    time_per_cycle = accept_rate * t_accept + (1 - accept_rate) * t_reject

    tps = tokens_per_cycle / (time_per_cycle / 1000)
    speedup = tps / base_tps

    return tps, speedup


def main():
    parser = argparse.ArgumentParser(description="Analyze restricted vocab MTP ceiling")
    parser.add_argument("--data", type=Path, default=DATA_DIR / "train_tokens.bin",
                        help="Tokenized data file (uint32)")
    parser.add_argument("--counts", type=Path, default=DATA_DIR / "token_counts.bin",
                        help="Token frequency counts (int64)")
    args = parser.parse_args()

    data_path = args.data.expanduser().resolve()
    counts_path = args.counts.expanduser().resolve()

    k_values = [1000, 2000, 5000, 10000, 15000, 20000, 30000, 50000]

    print(f"Data: {data_path}")
    print(f"Counts: {counts_path}")
    print()

    # Load data
    print("Loading tokenized data...")
    tokens = np.fromfile(str(data_path), dtype=np.uint32)
    print(f"  Loaded {len(tokens):,} entries")

    print("Building top-K masks...")
    masks = load_top_k_masks(counts_path, k_values)

    print("\nAnalyzing...")
    results = analyze_sequences(tokens, masks, k_values)

    # === Report ===
    print(f"\n{'='*90}")
    print("THEORETICAL MTP ACCURACY WITH RESTRICTED VOCAB")
    print(f"{'='*90}")

    # Table 1: Coverage
    fmt = "%-8s %10s %10s %10s %10s %10s %10s"
    print(f"\n{fmt % ('K', 'Coverage', 'Pair Cov', 'Triplet', 'Avg Run', 'Med Run', 'P90 Run')}")
    print("-" * 90)
    for k in k_values:
        r = results[k]
        print(fmt % (
            f"{k:,}",
            f"{100*r['coverage']:.2f}%",
            f"{100*r['pair_coverage']:.2f}%",
            f"{100*r['triplet_coverage']:.2f}%",
            f"{r['avg_run']:.1f}",
            f"{r['median_run']:.0f}",
            f"{r['p90_run']:.0f}",
        ))

    # Table 2: Throughput estimates
    print(f"\n{'='*90}")
    print("ESTIMATED THROUGHPUT (base=599 tok/s)")
    print(f"{'='*90}")

    # Different assumptions about MTP accuracy within the restricted vocab
    accuracy_scenarios = [0.60, 0.70, 0.80, 0.90, 1.00]

    fmt2 = "%-8s" + " %12s" * len(accuracy_scenarios)
    headers = [f"α={a:.0%}" for a in accuracy_scenarios]
    print(f"\n{fmt2 % tuple(['K'] + headers)}")
    print("-" * (8 + 13 * len(accuracy_scenarios)))

    for k in k_values:
        r = results[k]
        cells = []
        for alpha in accuracy_scenarios:
            tps, speedup = estimate_throughput(r['coverage'], alpha)
            cells.append(f"{tps:.0f} ({speedup:.2f}x)")
        print(fmt2 % tuple([f"{k:,}"] + cells))

    print(f"""
Notes:
  Coverage  = P(token ∈ top-K) — ceiling if MTP were perfect
  Pair Cov  = P(token[t] AND token[t+1] both ∈ top-K) — consecutive in-vocab
  Triplet   = P(three consecutive tokens all ∈ top-K)
  Avg/Med/P90 Run = consecutive in-vocab tokens before hitting OOV
  α = MTP accuracy within the restricted vocab (P(correct | token ∈ top-K))
  Current MTP: ~55% acceptance on full 248K vocab → α ≈ 55% at K=248K

  Key insight: fine-tuning MTP on English with restricted vocab should push α
  from ~55% toward 70-80%+, while coverage provides the ceiling.
  Break-even is ~65% acceptance → ~1.00x speedup.
""")


if __name__ == "__main__":
    main()
