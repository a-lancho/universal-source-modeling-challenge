"""Instructor tool to release a public practice test set (seed is public)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from competition.release_test import LOCKED_CONFIG, _sha256_file
from src.data.synthetic_source import build_random_hmm_source


def _build_practice_metadata(*, test_length: int, seed: int) -> dict[str, Any]:
    run_id = f"practice_a16_t{int(test_length)}_seed{int(seed)}"
    return {
        "test_length": int(test_length),
        "alphabet_size": int(LOCKED_CONFIG.alphabet_size),
        "seed": int(seed),
        "run_id": run_id,
    }


def release_public_practice_set(
    *,
    test_length: int = 300_000,
    seed: int,
    output_dir: str | Path = "data/public_practice",
) -> Path:
    """Generate and save a public practice test set using the locked HMM config."""

    if test_length <= 0:
        raise ValueError("test_length must be positive.")

    source = build_random_hmm_source(LOCKED_CONFIG)
    test_seq, _predictive_bits = source.sample(
        test_length,
        seed=seed,
        burn_in=LOCKED_CONFIG.burn_in,
        return_states=False,
    )

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    test_path = out_dir / "test.npy"
    np.save(test_path, test_seq)

    metadata = _build_practice_metadata(test_length=test_length, seed=seed)
    with (out_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    checksum = _sha256_file(test_path)
    with (out_dir / "sha256.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"{checksum}  test.npy\n")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Release a public practice test set.")
    parser.add_argument("--test-length", type=int, default=300_000)
    parser.add_argument("--seed", type=int, required=True, help="Public practice seed (shareable).")
    parser.add_argument("--output-dir", type=str, default="data/public_practice")
    args = parser.parse_args()

    out_dir = release_public_practice_set(
        test_length=args.test_length,
        seed=args.seed,
        output_dir=args.output_dir,
    )
    print(f"Released public practice set to {out_dir}")
    print(f"  files: {out_dir / 'test.npy'}, {out_dir / 'metadata.json'}, {out_dir / 'sha256.txt'}")


if __name__ == "__main__":
    main()

