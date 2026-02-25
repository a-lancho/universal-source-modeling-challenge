"""Lightweight integration test for public practice release + hash verification."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def test_release_practice_creates_files_and_hashes_verify(tmp_path: Path) -> None:
    out_dir = tmp_path / "public_practice"
    repo_root = Path(__file__).resolve().parents[1]

    release_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "competition.release_practice",
            "--test-length",
            "2000",
            "--seed",
            "4242",
            "--output-dir",
            str(out_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "Released public practice set" in release_proc.stdout

    test_path = out_dir / "test.npy"
    metadata_path = out_dir / "metadata.json"
    sha_path = out_dir / "sha256.txt"
    assert test_path.exists()
    assert metadata_path.exists()
    assert sha_path.exists()

    arr = np.load(test_path)
    assert arr.shape == (2000,)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["test_length"] == 2000
    assert metadata["alphabet_size"] == 16
    assert metadata["seed"] == 4242
    assert "run_id" in metadata

    verify_proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "competition.verify_hashes",
            "--dir",
            str(out_dir),
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )
    assert "STATUS: PASS" in verify_proc.stdout

