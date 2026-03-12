"""
Dataset for MTP training on pre-tokenized binary data.

Data format: uint32 tokens separated by SENTINEL (0xFFFFFFFF) between documents.
Produced by scripts/prepare_training_data.py.

Restricted vocab: maps full vocab IDs to [0, K) indices. OOV tokens get target=-100
(PyTorch ignore_index), so they don't contribute to the loss.
"""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


SENTINEL = 0xFFFFFFFF
VOCAB_SIZE = 248320


class RestrictedVocab:
    """Maps full vocab IDs (248K) to restricted vocab indices (K).

    Built from token frequency counts. OOV tokens map to -1.
    """

    def __init__(self, counts_path: Path, k: int):
        counts = np.fromfile(str(counts_path), dtype=np.int64)
        if len(counts) < VOCAB_SIZE:
            counts = np.pad(counts, (0, VOCAB_SIZE - len(counts)))

        sorted_ids = np.argsort(-counts)
        self.top_k_ids = sorted_ids[:k].astype(np.int32)

        # Build mapping: full_id → restricted_index (or -1)
        self.full_to_restricted = np.full(VOCAB_SIZE, -1, dtype=np.int32)
        for i, token_id in enumerate(self.top_k_ids):
            self.full_to_restricted[token_id] = i

        # Sort for cache-friendly access at inference
        self.top_k_ids_sorted = np.sort(self.top_k_ids)

        # Stats
        total = counts.sum()
        coverage = counts[self.top_k_ids].sum() / total if total > 0 else 0
        self.k = k
        self.coverage = coverage

    def map_targets(self, token_ids: np.ndarray) -> np.ndarray:
        """Map full vocab IDs to restricted indices. OOV → -100 (ignore)."""
        restricted = self.full_to_restricted[token_ids]
        # PyTorch cross-entropy ignore_index = -100
        restricted[restricted == -1] = -100
        return restricted

    def __len__(self) -> int:
        return self.k


class TokenSequenceDataset(Dataset):
    """Dataset of fixed-length token sequences from pre-tokenized binary data.

    Splits the token stream into documents (at SENTINEL boundaries), then
    creates fixed-length chunks. Short documents are padded; chunks that
    cross document boundaries are split.

    Each item returns:
        token_ids: [seq_len] int64 — full vocab token IDs
        targets:   [seq_len-2] int32 — restricted vocab targets for MTP
                   (target[i] = restricted_vocab[token_ids[i+2]], or -100 if OOV)
        length:    int — actual valid length (before padding)
    """

    def __init__(
        self,
        data_path: Path,
        vocab: RestrictedVocab,
        seq_len: int = 512,
        min_doc_len: int = 8,
    ):
        self.seq_len = seq_len
        self.vocab = vocab

        # Load and split by sentinels
        raw = np.fromfile(str(data_path), dtype=np.uint32)
        sentinel_mask = raw == SENTINEL
        splits = np.where(sentinel_mask)[0]

        # Extract documents
        chunks = []
        start = 0
        for end in splits:
            if end - start >= min_doc_len:
                doc = raw[start:end]
                # Split long documents into seq_len chunks
                for i in range(0, len(doc), seq_len):
                    chunk = doc[i : i + seq_len]
                    if len(chunk) >= min_doc_len:
                        chunks.append(chunk)
            start = end + 1

        # Handle trailing tokens (no final sentinel)
        if start < len(raw) and len(raw) - start >= min_doc_len:
            doc = raw[start:]
            doc = doc[~(doc == SENTINEL)]  # remove any stray sentinels
            for i in range(0, len(doc), seq_len):
                chunk = doc[i : i + seq_len]
                if len(chunk) >= min_doc_len:
                    chunks.append(chunk)

        self.chunks = chunks
        self.n_tokens = sum(len(c) for c in chunks)

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        chunk = self.chunks[idx]
        length = len(chunk)

        # Pad to seq_len if needed
        if length < self.seq_len:
            padded = np.zeros(self.seq_len, dtype=np.uint32)
            padded[:length] = chunk
            chunk = padded

        token_ids = torch.from_numpy(chunk.astype(np.int64))

        # MTP targets: token_ids[t+2] mapped to restricted vocab
        # For positions t=0..L-3, target = restricted(token_ids[t+2])
        target_ids = chunk[2:length] if length > 2 else np.array([], dtype=np.uint32)
        targets = self.vocab.map_targets(target_ids)

        # Pad targets to seq_len-2
        target_padded = np.full(self.seq_len - 2, -100, dtype=np.int32)
        target_padded[: len(targets)] = targets

        return {
            "token_ids": token_ids,
            "targets": torch.from_numpy(target_padded).long(),
            "length": length,
        }


def make_splits(
    data_path: Path,
    vocab: RestrictedVocab,
    seq_len: int = 512,
    val_fraction: float = 0.05,
    seed: int = 42,
) -> tuple[TokenSequenceDataset, TokenSequenceDataset]:
    """Create train/val split from a single data file."""
    full = TokenSequenceDataset(data_path, vocab, seq_len)

    n = len(full)
    n_val = max(1, int(n * val_fraction))
    n_train = n - n_val

    rng = torch.Generator().manual_seed(seed)
    indices = torch.randperm(n, generator=rng).tolist()

    train_ds = _SubsetDataset(full, indices[:n_train])
    val_ds = _SubsetDataset(full, indices[n_train:])

    return train_ds, val_ds


class _SubsetDataset(Dataset):
    """Thin wrapper for index-based subset."""

    def __init__(self, dataset: TokenSequenceDataset, indices: list[int]):
        self.dataset = dataset
        self.indices = indices

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[self.indices[idx]]
