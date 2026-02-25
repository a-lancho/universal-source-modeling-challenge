"""Utilities for comparing model code-length estimates to standard compressors.

This module is optional and purely educational. It must not be used as the
competition leaderboard metric.
"""

from __future__ import annotations

import bz2
import lzma
from pathlib import Path
import zlib

import numpy as np


def load_sequence(data_dir: str | Path, split: str) -> np.ndarray:
    """Load `train.npy` or `test.npy` from a data directory as int64."""

    if split not in {"train", "test"}:
        raise ValueError("split must be one of {'train', 'test'}.")
    path = Path(data_dir) / f"{split}.npy"
    if not path.exists():
        raise FileNotFoundError(f"Missing sequence file: {path}")
    arr = np.load(path)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D sequence array at {path}, got shape {arr.shape}.")
    return np.asarray(arr, dtype=np.int64)


def sequence_to_bytes(seq: np.ndarray, alphabet_size: int | None = None) -> bytes:
    """Convert an integer symbol sequence to bytes for standard compressors.

    Encoding:
    - `alphabet_size <= 256`: one byte per symbol (`uint8`)
    - `alphabet_size > 256`: two bytes per symbol (`uint16`, little-endian)
    """

    arr = np.asarray(seq, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D sequence, got shape {arr.shape}.")
    if arr.size == 0:
        return b""
    if np.any(arr < 0):
        raise ValueError("Sequence contains negative symbols; cannot encode to bytes.")

    inferred_max = int(arr.max())
    if alphabet_size is None:
        alphabet_size = inferred_max + 1
    alphabet_size = int(alphabet_size)
    if alphabet_size <= 0:
        raise ValueError("alphabet_size must be positive.")
    if inferred_max >= alphabet_size:
        raise ValueError(
            f"Sequence contains symbol {inferred_max} but alphabet_size={alphabet_size}."
        )

    if alphabet_size <= 256:
        return np.asarray(arr, dtype=np.uint8).tobytes(order="C")

    if alphabet_size > 65536:
        raise ValueError("alphabet_size > 65536 is not supported by uint16 packing.")
    return np.asarray(arr, dtype="<u2").tobytes(order="C")


def compress_bytes(payload: bytes) -> dict[str, int]:
    """Compress payload with standard-library compressors and return sizes in bytes."""

    return {
        "zlib": len(zlib.compress(payload, level=9)),
        "lzma": len(lzma.compress(payload, preset=9)),
        "bz2": len(bz2.compress(payload, compresslevel=9)),
    }


def summarize_compression(
    seq: np.ndarray,
    *,
    alphabet_size: int,
    model_bits: float | None = None,
) -> dict:
    """Return a structured comparison of raw/compressed/model code lengths."""

    arr = np.asarray(seq, dtype=np.int64)
    if arr.ndim != 1:
        raise ValueError(f"Expected a 1D sequence, got shape {arr.shape}.")
    n = int(arr.size)
    payload = sequence_to_bytes(arr, alphabet_size=alphabet_size)
    raw_bytes = len(payload)
    compressed = compress_bytes(payload)

    def _bps(num_bytes: int) -> float:
        return float("nan") if n == 0 else (8.0 * float(num_bytes) / float(n))

    summary: dict[str, object] = {
        "num_symbols": n,
        "alphabet_size": int(alphabet_size),
        "encoding": "uint8" if alphabet_size <= 256 else "uint16_le",
        "raw_bytes": raw_bytes,
        "raw_bps": _bps(raw_bytes),
        "compressors": {
            name: {
                "compressed_bytes": size_bytes,
                "compressed_bps": _bps(size_bytes),
            }
            for name, size_bytes in compressed.items()
        },
    }

    if model_bits is not None:
        model_bits = float(model_bits)
        model_bps = float("nan") if n == 0 else (model_bits / float(n))
        summary["model"] = {
            "total_bits": model_bits,
            "bits_per_symbol": model_bps,
        }
        summary["deltas_vs_model_bps"] = {
            "raw_minus_model": summary["raw_bps"] - model_bps,  # type: ignore[operator]
            "zlib_minus_model": summary["compressors"]["zlib"]["compressed_bps"] - model_bps,  # type: ignore[index]
            "lzma_minus_model": summary["compressors"]["lzma"]["compressed_bps"] - model_bps,  # type: ignore[index]
            "bz2_minus_model": summary["compressors"]["bz2"]["compressed_bps"] - model_bps,  # type: ignore[index]
        }

    return summary
