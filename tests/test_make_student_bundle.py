"""Lightweight integration test for student bundle assembly."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_make_student_bundle_with_temp_inputs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[1]

    # Create small dummy train.npy and public practice set in temp locations.
    train_dir = tmp_path / "inputs" / "generator"
    train_dir.mkdir(parents=True, exist_ok=True)
    train_path = train_dir / "train.npy"
    np.save(train_path, np.array([0, 1, 2, 3] * 10, dtype=np.int64))

    practice_dir = tmp_path / "inputs" / "public_practice"
    practice_dir.mkdir(parents=True, exist_ok=True)
    np.save(practice_dir / "test.npy", np.array([0, 1, 2, 3] * 20, dtype=np.int64))
    (practice_dir / "metadata.json").write_text(
        json.dumps(
            {
                "test_length": 80,
                "alphabet_size": 16,
                "seed": 4242,
                "run_id": "practice_test",
            }
        ),
        encoding="utf-8",
    )
    (practice_dir / "sha256.txt").write_text("dummyhash  test.npy\n", encoding="utf-8")

    out_dir = tmp_path / "student_bundle"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "competition.make_student_bundle",
            "--output-dir",
            str(out_dir),
            "--train-path",
            str(train_path),
            "--practice-dir",
            str(practice_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Student bundle created at" in proc.stdout

    # Key files from repo
    assert (out_dir / "COMPETITION_RULES.md").exists()
    assert (out_dir / "competition" / "run_live_eval.py").exists()
    assert (out_dir / "competition" / "config.yaml").exists()
    assert (out_dir / "competition" / "verify_hashes.py").exists()
    assert (out_dir / "submissions" / "template_predictor.py").exists()
    assert (out_dir / "notebooks" / "colab_starter.ipynb").exists()

    # Copied data files at expected relative destinations
    assert (out_dir / "data" / "generator" / "train.npy").exists()
    assert (out_dir / "data" / "public_practice" / "test.npy").exists()
    assert (out_dir / "data" / "public_practice" / "metadata.json").exists()
    assert (out_dir / "data" / "public_practice" / "sha256.txt").exists()

