"""Lightweight tests for live competition evaluation tooling."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np


def test_run_live_eval_imports_temp_uniform_predictor_and_prints_final_score(
    tmp_path: Path,
) -> None:
    test_seq = np.array([0, 1, 2, 3] * 100, dtype=np.int64)
    test_path = tmp_path / "test.npy"
    np.save(test_path, test_seq)

    predictor_path = tmp_path / "predictor.py"
    predictor_path.write_text(
        "\n".join(
            [
                "from src.predictors.uniform import UniformPredictor",
                "",
                "def build_predictor(alphabet_size: int, max_context_length: int):",
                "    return UniformPredictor(",
                "        alphabet_size=alphabet_size,",
                "        max_context_length=max_context_length,",
                "    )",
                "",
            ]
        ),
        encoding="utf-8",
    )

    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "competition.run_live_eval",
            "--test-path",
            str(test_path),
            "--predictor-path",
            str(predictor_path),
            "--num-tokens",
            "200",
            "--time-limit-seconds",
            "10",
        ],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    )

    lines = [line for line in proc.stdout.strip().splitlines() if line.strip()]
    assert len(lines) == 1
    line = lines[0]
    assert line.startswith("FINAL_SCORE ")
    assert "bits_per_symbol=" in line
    assert "elapsed_seconds=" in line
    assert "timed_out=" in line
    assert "evaluated_tokens=" in line

