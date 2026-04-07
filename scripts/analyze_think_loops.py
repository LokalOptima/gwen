#!/usr/bin/env python3
"""Analyze model outputs for thinking loops.

Reads generated text files, extracts <think> blocks, and detects
repeated n-gram patterns indicating the model got stuck in a loop.

Usage:
    python3 analyze_think_loops.py output_dir/
    python3 analyze_think_loops.py file1.txt file2.txt ...
"""
import sys
import os
import re


def extract_think_block(text):
    """Extract content of the first <think> block.

    Returns (content, closed) where closed=False means the model
    hit the token limit while still thinking.
    """
    start = text.find("<think>")
    if start == -1:
        return None, True

    start += len("<think>")
    end = text.find("</think>", start)
    if end == -1:
        # Unclosed: model was still thinking when generation stopped
        return text[start:].strip(), False
    return text[start:end].strip(), True


def detect_loops(words, min_period=5, max_period=300, min_consecutive=3):
    """Find repeated n-gram runs in a word list.

    Returns list of findings sorted by severity (most repeated words first).
    Each finding: {period, consecutive, onset, repeated_words, sample}.

    Uses two strategies:
    1. N-gram hashing: fast detection via position tracking (original)
    2. Forward scan: for each position, check if the next P words repeat
       at period P. Catches loops even when the first cycle differs.
    """
    n = len(words)
    if n < min_period * min_consecutive:
        return []

    findings = []
    seen_onsets = set()

    # Check every period. For typical think blocks (~5000 words) and
    # max_period=300, the forward scan is O(n * max_period) ≈ 1.5M — fast enough.
    periods = range(min_period, min(max_period, n // min_consecutive) + 1)

    for period in periods:
        # Forward scan: at each candidate onset, check how many times the
        # next `period` words repeat consecutively.  Stride by period//2 to
        # keep runtime O(n * max_period) across all periods.
        stride = max(1, period // 2)
        for start in range(0, n - period * min_consecutive + 1, stride):
            pattern = words[start : start + period]
            repeats = 1
            pos = start + period
            while pos + period <= n and words[pos : pos + period] == pattern:
                repeats += 1
                pos += period
            if repeats >= min_consecutive:
                if any(abs(start - s) < period for s in seen_onsets):
                    continue
                seen_onsets.add(start)
                sample = " ".join(pattern[:25])
                if period > 25:
                    sample += " ..."
                findings.append(
                    {
                        "period": period,
                        "consecutive": repeats,
                        "onset": start,
                        "repeated_words": repeats * period,
                        "sample": sample,
                    }
                )

    findings.sort(key=lambda f: -f["repeated_words"])
    return findings[:5]


def analyze_file(filepath):
    """Analyze a single output file. Returns a result dict."""
    with open(filepath) as f:
        text = f.read()

    total_chars = len(text)
    think_content, closed = extract_think_block(text)

    if think_content is not None:
        analyze_text = think_content
        label = "think"
    else:
        # No <think> block — analyze the full text for repetition
        analyze_text = text
        label = "full"
        closed = True

    words = analyze_text.split()
    think_words = len(words)
    loops = detect_loops(words)

    if not loops:
        verdict = "OK"
    else:
        worst = loops[0]
        frac = worst["repeated_words"] / max(think_words, 1)
        if frac > 0.3:
            verdict = "LOOP"
        elif worst["consecutive"] >= 5:
            verdict = "LOOP"
        else:
            verdict = "REPETITIVE"

    return {
        "file": os.path.basename(filepath),
        "think_words": think_words,
        "closed": closed,
        "loops": loops,
        "verdict": verdict,
        "label": label,
    }


def print_report(results):
    """Print a human-readable report."""
    w = max(len(r["file"]) for r in results) if results else 20

    print()
    print("=" * 72)
    print("  THINKING LOOP ANALYSIS")
    print("=" * 72)

    n_loop = 0
    n_repetitive = 0
    n_ok = 0
    n_no_think = 0

    for r in results:
        verdict = r["verdict"]
        if verdict == "LOOP":
            tag = "LOOP"
            n_loop += 1
        elif verdict == "REPETITIVE":
            tag = "WARN"
            n_repetitive += 1
        elif verdict == "NO_THINK":
            tag = "SKIP"
            n_no_think += 1
        else:
            tag = " OK "
            n_ok += 1

        label = r.get("label", "think")
        closed_str = "" if r["closed"] else f" [unclosed <{label}>]"
        print(
            f"  [{tag}] {r['file']:<{w}}  "
            f"{label}={r['think_words']:>5} words{closed_str}"
        )

        for i, loop in enumerate(r["loops"]):
            onset_pct = 100.0 * loop["onset"] / max(r["think_words"], 1)
            print(
                f"         {'└' if i == len(r['loops'])-1 else '├'}"
                f" period={loop['period']} words × {loop['consecutive']} repeats"
                f"  onset=word {loop['onset']} ({onset_pct:.0f}%)"
            )
            # Truncate sample to terminal width
            sample = loop["sample"]
            if len(sample) > 90:
                sample = sample[:87] + "..."
            print(f"           \"{sample}\"")

    print()
    print("-" * 72)
    print(
        f"  LOOP: {n_loop}  |  WARN: {n_repetitive}  |  OK: {n_ok}"
        f"  |  NO_THINK: {n_no_think}  |  total: {len(results)}"
    )
    if n_loop > 0:
        print("  *** Thinking loops detected — consider --dry-multiplier or --reasoning-budget ***")
    print("=" * 72)
    print()

    return n_loop


def main():
    paths = []
    for arg in sys.argv[1:]:
        if os.path.isdir(arg):
            for f in sorted(os.listdir(arg)):
                if f.endswith(".txt"):
                    paths.append(os.path.join(arg, f))
        elif os.path.isfile(arg):
            paths.append(arg)
        else:
            print(f"WARNING: skipping {arg} (not found)", file=sys.stderr)

    if not paths:
        print("Usage: analyze_think_loops.py <output_dir_or_files...>", file=sys.stderr)
        sys.exit(1)

    results = [analyze_file(p) for p in paths]
    n_loop = print_report(results)
    sys.exit(1 if n_loop > 0 else 0)


if __name__ == "__main__":
    main()
