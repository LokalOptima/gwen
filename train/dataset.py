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
from torch.utils.data import Dataset, Sampler


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
        self._lengths = [len(c) for c in chunks]

    @property
    def lengths(self) -> list[int]:
        return self._lengths

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        chunk = self.chunks[idx]
        length = len(chunk)

        token_ids = torch.from_numpy(chunk.astype(np.int64))

        # MTP targets: token_ids[t+2] mapped to restricted vocab
        target_ids = chunk[2:length] if length > 2 else np.array([], dtype=np.uint32)
        targets = torch.from_numpy(self.vocab.map_targets(target_ids)).long()

        return {
            "token_ids": token_ids,
            "targets": targets,
            "length": length,
        }


def mtp_collate(batch: list[dict]) -> dict:
    """Collate variable-length sequences, padding to max length in batch.

    Pads token_ids with 0, targets with -100 (ignore_index).
    This ensures compute scales with actual sequence lengths, not seq_len.
    """
    max_len = max(item["length"] for item in batch)
    max_target_len = max(len(item["targets"]) for item in batch)

    token_ids = torch.zeros(len(batch), max_len, dtype=torch.long)
    targets = torch.full((len(batch), max_target_len), -100, dtype=torch.long)

    for i, item in enumerate(batch):
        L = item["length"]
        token_ids[i, :L] = item["token_ids"]
        tgt_len = len(item["targets"])
        targets[i, :tgt_len] = item["targets"]

    return {"token_ids": token_ids, "targets": targets}


class TokenBatchSampler(Sampler):
    """Batch sampler that targets a constant token budget per batch.

    Sorts sequences by length, packs similar-length sequences into batches
    targeting `max_tokens` total tokens per batch. Shuffles batch order each
    epoch (deterministic, seeded).

    Benefits over fixed batch_size:
    - Consistent GPU memory/compute per batch
    - Minimal padding (similar-length sequences grouped)
    - More training signal from short sequences
    """

    def __init__(
        self,
        lengths: list[int],
        max_tokens: int,
        shuffle: bool = True,
        seed: int = 42,
        drop_last: bool = True,
    ):
        self.max_tokens = max_tokens
        self.shuffle = shuffle
        self.seed = seed
        self.drop_last = drop_last
        self.epoch = 0

        # Sort indices by length (longest first for greedy packing)
        sorted_indices = sorted(range(len(lengths)), key=lambda i: -lengths[i])

        # Greedily pack into batches targeting max_tokens
        self.batches = []
        batch = []
        batch_max_len = 0
        for idx in sorted_indices:
            seq_len = lengths[idx]
            new_max = max(batch_max_len, seq_len)
            # If adding this sequence would exceed budget, flush
            if batch and new_max * (len(batch) + 1) > max_tokens:
                self.batches.append(batch)
                batch = [idx]
                batch_max_len = seq_len
            else:
                batch.append(idx)
                batch_max_len = new_max
        if batch and (not drop_last or len(batch) > 1):
            self.batches.append(batch)

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self.shuffle:
            rng = torch.Generator().manual_seed(self.seed + self.epoch)
            order = torch.randperm(len(self.batches), generator=rng).tolist()
        else:
            order = list(range(len(self.batches)))
        for i in order:
            yield self.batches[i]

    def __len__(self):
        return len(self.batches)


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
        self._lengths = [dataset.lengths[i] for i in indices]

    @property
    def lengths(self) -> list[int]:
        return self._lengths

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> dict:
        return self.dataset[self.indices[idx]]
