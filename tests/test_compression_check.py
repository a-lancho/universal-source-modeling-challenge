"""Lightweight tests for optional compression sanity-check utilities."""

from __future__ import annotations

import numpy as np

from src.sanity_checks.compression_check import compress_bytes, sequence_to_bytes


def test_sequence_to_bytes_roundtrip_like_behavior() -> None:
    seq_small = np.array([0, 1, 2, 255, 3], dtype=np.int64)
    payload_small = sequence_to_bytes(seq_small, alphabet_size=256)
    assert isinstance(payload_small, bytes)
    assert len(payload_small) == len(seq_small)

    seq_large = np.array([0, 1, 256, 1024, 4095], dtype=np.int64)
    payload_large = sequence_to_bytes(seq_large, alphabet_size=5000)
    assert isinstance(payload_large, bytes)
    assert len(payload_large) == 2 * len(seq_large)


def test_compress_bytes_returns_all_keys() -> None:
    payload = b"abcabcabcabc" * 100
    out = compress_bytes(payload)

    assert set(out.keys()) == {"zlib", "lzma", "bz2"}
    assert all(isinstance(v, int) and v > 0 for v in out.values())

