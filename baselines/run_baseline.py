"""Run a baseline predictor on generated train/test splits.

Usage (from repo root):
    python -m baselines.run_baseline --data-dir data/generator --baseline uniform
    python -m baselines.run_baseline --data-dir data/generator --baseline ngram --ngram-n 3
    python -m baselines.run_baseline --data-dir data/generator --baseline ngram_threshold --ngram-n 5 --min-count 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.harness import EvaluationResult, evaluate_sequence
from src.predictors.ngram import NGramPredictor
from src.predictors.ngram_threshold import CountThresholdNGramPredictor
from src.predictors.uniform import UniformPredictor


def _load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    array = np.load(path)
    if array.ndim != 1:
        raise ValueError(f"Expected a 1D sequence array at {path}, got shape {array.shape}.")
    return np.asarray(array, dtype=np.int64)


def _load_metadata(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _infer_alphabet_size(train: np.ndarray, test: np.ndarray, metadata: dict[str, Any]) -> int:
    if "alphabet_size" in metadata:
        alphabet_size = int(metadata["alphabet_size"])
        if alphabet_size <= 0:
            raise ValueError("metadata.json contains an invalid alphabet_size.")
        return alphabet_size

    max_symbol = -1
    if train.size:
        max_symbol = max(max_symbol, int(train.max()))
    if test.size:
        max_symbol = max(max_symbol, int(test.max()))
    if max_symbol < 0:
        raise ValueError("Cannot infer alphabet size from empty train/test sequences.")
    return max_symbol + 1


def _build_predictor(
    *,
    baseline: str,
    alphabet_size: int,
    max_context_length: int,
    ngram_n: int,
    laplace: float,
    min_count: int,
):
    if baseline == "uniform":
        return UniformPredictor(
            alphabet_size=alphabet_size,
            max_context_length=max_context_length,
        )
    if baseline == "ngram":
        return NGramPredictor(
            alphabet_size=alphabet_size,
            n=ngram_n,
            laplace=laplace,
            max_context_length=max_context_length,
            adapt_online=True,
        )
    if baseline == "ngram_threshold":
        return CountThresholdNGramPredictor(
            alphabet_size=alphabet_size,
            n=ngram_n,
            laplace=laplace,
            min_count=min_count,
            max_context_length=max_context_length,
            adapt_online=True,
        )
    raise ValueError(f"Unknown baseline: {baseline}")


def _print_summary(
    *,
    baseline: str,
    alphabet_size: int,
    max_context_length: int,
    result: EvaluationResult,
    ngram_n: int | None = None,
    laplace: float | None = None,
    min_count: int | None = None,
) -> None:
    print("Baseline Evaluation Summary")
    print(f"  baseline: {baseline}")
    print(f"  alphabet size: {alphabet_size}")
    print(f"  max_context_length: {max_context_length}")
    if ngram_n is not None:
        print(f"  ngram_n: {ngram_n}")
    if laplace is not None:
        print(f"  laplace: {laplace}")
    if min_count is not None:
        print(f"  min_count: {min_count}")
    print(f"  num_tokens_evaluated: {result.num_tokens}")
    print(f"  total_bits: {result.total_bits:.6f}")
    print(f"  bits_per_symbol: {result.bits_per_symbol:.6f}")
    print(f"  elapsed_seconds: {result.elapsed_seconds:.6f}")
    print(f"  tokens_per_second: {result.tokens_per_second:.2f}")
    print(f"  timed_out: {result.timed_out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a baseline predictor on train/test splits.")
    parser.add_argument("--data-dir", type=str, default="data/generator")
    parser.add_argument("--baseline", choices=("uniform", "ngram", "ngram_threshold"), required=True)
    parser.add_argument("--max-context-length", type=int, default=256)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--ngram-n", type=int, default=3)
    parser.add_argument("--laplace", type=float, default=1.0)
    parser.add_argument("--min-count", type=int, default=5)
    parser.add_argument(
        "--validate-probabilities",
        action="store_true",
        help="Enable probability normalization checks during evaluation (slower).",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train = _load_array(data_dir / "train.npy")
    test = _load_array(data_dir / "test.npy")
    metadata = _load_metadata(data_dir / "metadata.json")
    alphabet_size = _infer_alphabet_size(train, test, metadata)

    predictor = _build_predictor(
        baseline=args.baseline,
        alphabet_size=alphabet_size,
        max_context_length=args.max_context_length,
        ngram_n=args.ngram_n,
        laplace=args.laplace,
        min_count=args.min_count,
    )

    if isinstance(predictor, NGramPredictor):
        predictor.fit(train)

    result = evaluate_sequence(
        predictor,
        test,
        max_context_length=args.max_context_length,
        max_seconds=args.max_seconds,
        validate_probabilities=args.validate_probabilities,
    )

    _print_summary(
        baseline=args.baseline,
        alphabet_size=alphabet_size,
        max_context_length=args.max_context_length,
        result=result,
        ngram_n=args.ngram_n if args.baseline in {"ngram", "ngram_threshold"} else None,
        laplace=args.laplace if args.baseline in {"ngram", "ngram_threshold"} else None,
        min_count=args.min_count if args.baseline == "ngram_threshold" else None,
    )


if __name__ == "__main__":
    main()
