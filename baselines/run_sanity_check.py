"""Optional sanity check: compare model bits to standard compressors.

This tool is educational only and does not affect leaderboard scoring.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.evaluation.harness import evaluate_sequence
from src.predictors.ngram import NGramPredictor
from src.sanity_checks.compression_check import (
    load_sequence,
    summarize_compression,
)

# Reuse baseline predictor construction and metadata helpers.
from baselines.run_baseline import _build_predictor, _infer_alphabet_size, _load_metadata


def _print_summary(
    *,
    baseline: str,
    split: str,
    compression_summary: dict,
    eval_summary: dict,
    ngram_n: int | None = None,
    laplace: float | None = None,
    min_count: int | None = None,
) -> None:
    print("Sanity Check: Prediction vs Compression (Bonus)")
    print("  leaderboard_metric_affected: False")
    print(f"  split: {split}")
    print(f"  baseline: {baseline}")
    if ngram_n is not None:
        print(f"  ngram_n: {ngram_n}")
    if laplace is not None:
        print(f"  laplace: {laplace}")
    if min_count is not None:
        print(f"  min_count: {min_count}")
    print(f"  num_symbols: {compression_summary['num_symbols']}")
    print(f"  alphabet_size: {compression_summary['alphabet_size']}")
    print(f"  encoding: {compression_summary['encoding']}")
    print(f"  raw_bps: {compression_summary['raw_bps']:.6f}")

    compressors = compression_summary["compressors"]
    for name in ("zlib", "lzma", "bz2"):
        stats = compressors[name]
        print(
            f"  {name}_bps: {stats['compressed_bps']:.6f} "
            f"({stats['compressed_bytes']} bytes)"
        )

    model = compression_summary.get("model")
    if model is not None:
        print(f"  model_bps: {model['bits_per_symbol']:.6f}")
        deltas = compression_summary.get("deltas_vs_model_bps", {})
        print(f"  delta_raw_minus_model_bps: {deltas.get('raw_minus_model', float('nan')):.6f}")
        print(f"  delta_zlib_minus_model_bps: {deltas.get('zlib_minus_model', float('nan')):.6f}")
        print(f"  delta_lzma_minus_model_bps: {deltas.get('lzma_minus_model', float('nan')):.6f}")
        print(f"  delta_bz2_minus_model_bps: {deltas.get('bz2_minus_model', float('nan')):.6f}")
    else:
        print("  model_bps: unavailable for full-sequence comparison (evaluation timed out)")

    print(f"  eval_num_tokens: {eval_summary['num_tokens']}")
    print(f"  eval_bits_per_symbol_partial: {eval_summary['bits_per_symbol']:.6f}")
    print(f"  eval_elapsed_seconds: {eval_summary['elapsed_seconds']:.6f}")
    print(f"  eval_tokens_per_second: {eval_summary['tokens_per_second']:.2f}")
    print(f"  eval_timed_out: {eval_summary['timed_out']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bonus sanity check: compare model cross-entropy to standard compressors."
    )
    parser.add_argument("--data-dir", type=str, default="data/generator")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument(
        "--baseline", choices=("uniform", "ngram", "ngram_threshold"), required=True
    )
    parser.add_argument("--max-context-length", type=int, default=256)
    parser.add_argument("--max-seconds", type=float, default=None)
    parser.add_argument("--ngram-n", type=int, default=4)
    parser.add_argument("--laplace", type=float, default=1.0)
    parser.add_argument("--min-count", type=int, default=8)
    parser.add_argument(
        "--validate-probabilities",
        action="store_true",
        help="Enable probability checks during evaluation (slower).",
    )
    parser.add_argument("--json-out", type=str, default=None)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    train = load_sequence(data_dir, "train")
    seq = load_sequence(data_dir, args.split)
    metadata = _load_metadata(data_dir / "metadata.json")
    alphabet_size = _infer_alphabet_size(train, seq, metadata)

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

    eval_result = evaluate_sequence(
        predictor,
        seq,
        max_context_length=args.max_context_length,
        max_seconds=args.max_seconds,
        validate_probabilities=args.validate_probabilities,
    )

    full_sequence_evaluated = eval_result.num_tokens == len(seq)
    compression_summary = summarize_compression(
        seq,
        alphabet_size=alphabet_size,
        model_bits=eval_result.total_bits if full_sequence_evaluated else None,
    )
    eval_summary = {
        "num_tokens": eval_result.num_tokens,
        "total_bits": eval_result.total_bits,
        "bits_per_symbol": eval_result.bits_per_symbol,
        "elapsed_seconds": eval_result.elapsed_seconds,
        "tokens_per_second": eval_result.tokens_per_second,
        "timed_out": eval_result.timed_out,
    }
    payload = {
        "tool": "compression_sanity_check",
        "leaderboard_metric_affected": False,
        "data_dir": str(data_dir),
        "split": args.split,
        "baseline": args.baseline,
        "params": {
            "max_context_length": args.max_context_length,
            "max_seconds": args.max_seconds,
            "ngram_n": args.ngram_n,
            "laplace": args.laplace,
            "min_count": args.min_count,
            "validate_probabilities": args.validate_probabilities,
        },
        "evaluation": eval_summary,
        "compression": compression_summary,
    }

    _print_summary(
        baseline=args.baseline,
        split=args.split,
        compression_summary=compression_summary,
        eval_summary=eval_summary,
        ngram_n=args.ngram_n if args.baseline in {"ngram", "ngram_threshold"} else None,
        laplace=args.laplace if args.baseline in {"ngram", "ngram_threshold"} else None,
        min_count=args.min_count if args.baseline == "ngram_threshold" else None,
    )

    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  json_written_to: {out_path}")


if __name__ == "__main__":
    main()
