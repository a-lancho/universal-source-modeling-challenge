"""Generate a reference benchmark table for provided baselines on a test set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.evaluation.harness import evaluate_sequence
from src.predictors.ngram import NGramPredictor
from src.predictors.ngram_threshold import CountThresholdNGramPredictor
from src.predictors.uniform import UniformPredictor


def _load_sequence(path: str | Path) -> np.ndarray:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Missing sequence file: {p}")
    arr = np.load(p)
    if arr.ndim != 1:
        raise ValueError(f"Expected 1D sequence at {p}, got shape {arr.shape}")
    return np.asarray(arr, dtype=np.int64)


def _infer_alphabet_size(train: np.ndarray, test: np.ndarray) -> int:
    max_symbol = -1
    if train.size:
        max_symbol = max(max_symbol, int(train.max()))
    if test.size:
        max_symbol = max(max_symbol, int(test.max()))
    if max_symbol < 0:
        raise ValueError("Cannot infer alphabet size from empty sequences.")
    return max_symbol + 1


def _default_specs() -> list[str]:
    return [
        "uniform",
        "ngram:n=4,laplace=1.0",
        "ngram_threshold:n=5,laplace=1.0,min_count=8",
        "ngram:n=3,laplace=1.0",
        "ngram:n=5,laplace=1.0",
    ]


def _split_include_specs(text: str) -> list[str]:
    # Supports comma-separated list of specs even when spec params also contain commas:
    # split on commas, then start a new spec when token looks like a baseline name/spec head.
    heads = ("uniform", "ngram:", "ngram_threshold:")
    parts = [p.strip() for p in text.split(",") if p.strip()]
    specs: list[str] = []
    for token in parts:
        if token.startswith(heads):
            specs.append(token)
        elif specs:
            specs[-1] = specs[-1] + "," + token
        else:
            raise ValueError(f"Invalid --include token: {token}")
    return specs


def _parse_spec(spec: str) -> tuple[str, dict[str, Any], str]:
    spec = spec.strip()
    if not spec:
        raise ValueError("Empty baseline spec.")

    if ":" in spec:
        name, param_str = spec.split(":", 1)
    else:
        name, param_str = spec, ""
    name = name.strip()
    params: dict[str, Any] = {}

    if param_str:
        for item in param_str.split(","):
            item = item.strip()
            if not item:
                continue
            if "=" not in item:
                raise ValueError(f"Invalid parameter in spec '{spec}': {item}")
            key, value = item.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key in {"n", "min_count"}:
                params[key] = int(value)
            elif key in {"laplace"}:
                params[key] = float(value)
            else:
                raise ValueError(f"Unsupported parameter '{key}' in spec '{spec}'.")

    if name not in {"uniform", "ngram", "ngram_threshold"}:
        raise ValueError(f"Unsupported baseline spec: {spec}")

    label = name
    if params:
        joined = ",".join(f"{k}={v}" for k, v in params.items())
        label = f"{name}:{joined}"
    return name, params, label


def _build_predictor(name: str, params: dict[str, Any], *, alphabet_size: int, max_context_length: int):
    if name == "uniform":
        return UniformPredictor(alphabet_size=alphabet_size, max_context_length=max_context_length)
    if name == "ngram":
        return NGramPredictor(
            alphabet_size=alphabet_size,
            n=int(params.get("n", 4)),
            laplace=float(params.get("laplace", 1.0)),
            max_context_length=max_context_length,
            adapt_online=True,
        )
    if name == "ngram_threshold":
        return CountThresholdNGramPredictor(
            alphabet_size=alphabet_size,
            n=int(params.get("n", 5)),
            laplace=float(params.get("laplace", 1.0)),
            min_count=int(params.get("min_count", 8)),
            max_context_length=max_context_length,
            adapt_online=True,
        )
    raise ValueError(f"Unsupported baseline name: {name}")


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    headers = [
        "baseline",
        "bits_per_symbol",
        "elapsed_seconds",
        "tokens_per_second",
        "timed_out",
        "evaluated_tokens",
    ]
    table_rows = []
    for row in rows:
        table_rows.append(
            [
                str(row["name"]),
                f"{row['bits_per_symbol']:.6f}",
                f"{row['elapsed_seconds']:.6f}",
                f"{row['tokens_per_second']:.2f}",
                str(row["timed_out"]),
                str(row["evaluated_tokens"]),
            ]
        )
    widths = [len(h) for h in headers]
    for r in table_rows:
        for i, cell in enumerate(r):
            widths[i] = max(widths[i], len(cell))

    def fmt(cells: list[str]) -> str:
        return "| " + " | ".join(cells[i].ljust(widths[i]) for i in range(len(cells))) + " |"

    lines = [fmt(headers), "|-" + "-|-".join("-" * w for w in widths) + "-|"]
    lines.extend(fmt(r) for r in table_rows)
    return "\n".join(lines)


def run_benchmark(
    *,
    test_path: str | Path,
    train_path: str | Path,
    max_context_length: int,
    time_limit_seconds: float,
    num_tokens: int,
    include_specs: list[str] | None = None,
) -> dict[str, Any]:
    train = _load_sequence(train_path)
    test_full = _load_sequence(test_path)
    if num_tokens <= 0:
        raise ValueError("num_tokens must be positive.")
    test = test_full[:num_tokens]
    if test.size == 0:
        raise ValueError("Selected test prefix is empty.")

    alphabet_size = _infer_alphabet_size(train, test)
    specs = include_specs if include_specs is not None else _default_specs()

    rows: list[dict[str, Any]] = []
    for spec in specs:
        name, params, label = _parse_spec(spec)
        predictor = _build_predictor(
            name, params, alphabet_size=alphabet_size, max_context_length=max_context_length
        )
        if isinstance(predictor, (NGramPredictor, CountThresholdNGramPredictor)):
            predictor.fit(train)
        result = evaluate_sequence(
            predictor,
            test,
            max_context_length=max_context_length,
            max_seconds=time_limit_seconds,
            validate_probabilities=False,
        )
        rows.append(
            {
                "name": label,
                "family": name,
                "params": params,
                "bits_per_symbol": result.bits_per_symbol,
                "elapsed_seconds": result.elapsed_seconds,
                "tokens_per_second": result.tokens_per_second,
                "timed_out": result.timed_out,
                "evaluated_tokens": result.num_tokens,
            }
        )

    return {
        "test_path": str(Path(test_path)),
        "train_path": str(Path(train_path)),
        "num_tokens": int(num_tokens),
        "max_context_length": int(max_context_length),
        "time_limit_seconds": float(time_limit_seconds),
        "alphabet_size": int(alphabet_size),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark provided baselines on a test set.")
    parser.add_argument("--test-path", type=str, default="data/public_practice/test.npy")
    parser.add_argument("--train-path", type=str, default="data/generator/train.npy")
    parser.add_argument("--max-context-length", type=int, default=256)
    parser.add_argument("--time-limit-seconds", type=float, default=600.0)
    parser.add_argument("--num-tokens", type=int, default=200_000)
    parser.add_argument("--json-out", type=str, default=None)
    parser.add_argument("--markdown-out", type=str, default=None)
    parser.add_argument(
        "--include",
        type=str,
        default=None,
        help=(
            "Comma-separated baseline specs, e.g. "
            "'uniform,ngram:n=4,laplace=1.0,ngram_threshold:n=5,laplace=1.0,min_count=8'"
        ),
    )
    args = parser.parse_args()

    include_specs = _split_include_specs(args.include) if args.include else None
    report = run_benchmark(
        test_path=args.test_path,
        train_path=args.train_path,
        max_context_length=args.max_context_length,
        time_limit_seconds=args.time_limit_seconds,
        num_tokens=args.num_tokens,
        include_specs=include_specs,
    )

    md = _markdown_table(report["rows"])
    print("# Baseline Benchmark Table")
    print()
    print(md)
    print()
    print("Recommended baselines:")
    print("- Official simple baseline: hard-backoff n-gram with n=4 (Laplace=1.0)")
    print("- Timeouts are disqualified on live day (per competition rules)")

    if args.markdown_out:
        path = Path(args.markdown_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(md + "\n", encoding="utf-8")

    if args.json_out:
        path = Path(args.json_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main()
