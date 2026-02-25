"""Smoke test for the baseline benchmark table generator."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import numpy as np

from src.data.synthetic_source import HMMSourceConfig, build_random_hmm_source


def test_benchmark_baselines_smoke(tmp_path: Path) -> None:
    config = HMMSourceConfig(
        num_states=4,
        alphabet_size=8,
        transition_concentration=0.7,
        emission_concentration=0.3,
        self_transition_bias=4.0,
        burn_in=256,
        seed=17,
    )
    source = build_random_hmm_source(config)
    split = source.generate_train_test_split(
        train_length=4000,
        test_length=4000,
        seed=9,
        burn_in=config.burn_in,
    )

    train_path = tmp_path / "train.npy"
    test_path = tmp_path / "test.npy"
    np.save(train_path, split.train)
    np.save(test_path, split.test)

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "competition.benchmark_baselines",
            "--train-path",
            str(train_path),
            "--test-path",
            str(test_path),
            "--num-tokens",
            "2000",
            "--time-limit-seconds",
            "30",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    assert "# Baseline Benchmark Table" in proc.stdout
    assert "bits_per_symbol" in proc.stdout
    assert "| baseline" in proc.stdout

