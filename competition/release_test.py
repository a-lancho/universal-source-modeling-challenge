"""Instructor tool to release the hidden test set for live competition day."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.data.synthetic_source import HMMSourceConfig, build_random_hmm_source


LOCKED_CONFIG = HMMSourceConfig(
    num_states=8,
    alphabet_size=16,
    self_transition_bias=12.0,
    emission_concentration=0.20,
    transition_concentration=0.5,
    burn_in=4096,
    seed=0,  # HMM parameter seed is locked; runtime sample seed is provided at release time.
)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _seed_hash(seed: int) -> str:
    return _sha256_bytes(str(int(seed)).encode("utf-8"))


def _build_public_metadata(*, test_length: int, seed: int) -> dict[str, Any]:
    seed_digest = _seed_hash(seed)
    run_id = f"live_a16_t{test_length}_{seed_digest[:12]}"
    return {
        "test_length": int(test_length),
        "alphabet_size": int(LOCKED_CONFIG.alphabet_size),
        "run_id": run_id,
        "seed_hash": seed_digest,
    }


def release_test_set(
    *,
    test_length: int,
    seed: int,
    output_dir: str | Path = "data/live_release",
    include_private_details: bool = False,
) -> Path:
    """Generate and save a live-release test set using the locked HMM config."""

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

    metadata = _build_public_metadata(test_length=test_length, seed=seed)
    if include_private_details:
        metadata.update(
            {
                "locked_config": {
                    "num_states": LOCKED_CONFIG.num_states,
                    "alphabet_size": LOCKED_CONFIG.alphabet_size,
                    "self_transition_bias": LOCKED_CONFIG.self_transition_bias,
                    "emission_concentration": LOCKED_CONFIG.emission_concentration,
                    "transition_concentration": LOCKED_CONFIG.transition_concentration,
                    "burn_in": LOCKED_CONFIG.burn_in,
                    "hmm_parameter_seed": LOCKED_CONFIG.seed,
                    "sample_seed": int(seed),
                },
                "transition_matrix": source.transition_matrix.tolist(),
                "emission_matrix": source.emission_matrix.tolist(),
                "initial_distribution": source.initial_distribution.tolist(),
            }
        )

    with (out_dir / "metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, indent=2)

    checksum = _sha256_file(test_path)
    with (out_dir / "sha256.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"{checksum}  test.npy\n")

    return out_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Release live competition test set only.")
    parser.add_argument("--test-length", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=str, default="data/live_release")
    parser.add_argument(
        "--include-private-details",
        action="store_true",
        help="Include hidden HMM parameters in metadata.json (instructor-only; do not use for live release).",
    )
    args = parser.parse_args()

    out_dir = release_test_set(
        test_length=args.test_length,
        seed=args.seed,
        output_dir=args.output_dir,
        include_private_details=args.include_private_details,
    )
    print(f"Released test set to {out_dir}")
    print(f"  files: {out_dir / 'test.npy'}, {out_dir / 'metadata.json'}, {out_dir / 'sha256.txt'}")


if __name__ == "__main__":
    main()

