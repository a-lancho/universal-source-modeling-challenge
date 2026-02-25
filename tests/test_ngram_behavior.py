"""Diagnostics and regression tests for the Laplace backoff n-gram baseline.

Important note:
Pure "longest seen context + Laplace smoothing" n-gram models are not guaranteed
to improve monotonically with larger `n`. The xfail test below documents the
reported anomaly and guards against assuming monotonicity as a correctness
criterion.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pytest

from src.data.synthetic_source import HMMSourceConfig, build_random_hmm_source
from src.evaluation.harness import evaluate_sequence
from src.predictors.ngram import NGramPredictor


def _evaluate_ngram_sweep(
    *,
    train: np.ndarray,
    test: np.ndarray,
    alphabet_size: int,
    max_n: int = 5,
) -> list[float]:
    results: list[float] = []
    for n in range(1, max_n + 1):
        predictor = NGramPredictor(
            alphabet_size=alphabet_size,
            n=n,
            laplace=1.0,
            adapt_online=True,
        )
        predictor.fit(train)
        result = evaluate_sequence(
            predictor,
            test,
            max_context_length=256,
            validate_probabilities=False,
        )
        results.append(result.bits_per_symbol)
    return results


def _manual_tuple_evaluate(
    predictor: NGramPredictor,
    sequence: Iterable[int],
    *,
    max_context_length: int,
) -> float:
    predictor.initialize()
    history: list[int] = []
    total_bits = 0.0
    count = 0
    for raw_symbol in sequence:
        symbol = int(raw_symbol)
        context = tuple(history[-max_context_length:]) if max_context_length > 0 else ()
        log_probs = predictor.predict_next(context)
        total_bits += -float(log_probs[symbol])
        predictor.update(symbol)
        history.append(symbol)
        count += 1
    return total_bits / count


@pytest.fixture(scope="session")
def anomaly_dataset() -> tuple[np.ndarray, np.ndarray, int]:
    """Generate the fixed HMM dataset used in the user-reported anomaly."""

    config = HMMSourceConfig(
        num_states=8,
        alphabet_size=16,
        self_transition_bias=12.0,
        emission_concentration=0.20,
        transition_concentration=0.5,
        burn_in=4096,
        seed=0,
    )
    source = build_random_hmm_source(config)
    split = source.generate_train_test_split(
        train_length=100_000,
        test_length=100_000,
        seed=0,
        burn_in=4096,
    )
    return split.train, split.test, config.alphabet_size


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    reason=(
        "Monotonic improvement with larger n is not guaranteed for pure suffix-backoff "
        "+ Laplace-smoothed n-grams; this test documents the reported anomaly."
    ),
)
def test_ngram_monotonicity_assumption_reproduces_xfail(
    anomaly_dataset: tuple[np.ndarray, np.ndarray, int]
) -> None:
    train, test, alphabet_size = anomaly_dataset
    bps = _evaluate_ngram_sweep(train=train, test=test, alphabet_size=alphabet_size, max_n=5)

    eps = 1e-3
    for idx in range(1, len(bps)):
        assert bps[idx] <= bps[idx - 1] + eps, f"n={idx+1} worsened: {bps}"


def test_ngram_harness_context_view_matches_manual_tuple_evaluation() -> None:
    """Ensure ContextWindowView does not change n-gram predictions/evaluation."""

    config = HMMSourceConfig(
        num_states=6,
        alphabet_size=8,
        self_transition_bias=6.0,
        emission_concentration=0.25,
        transition_concentration=0.7,
        burn_in=512,
        seed=11,
    )
    source = build_random_hmm_source(config)
    split = source.generate_train_test_split(train_length=5000, test_length=5000, seed=7, burn_in=512)

    p_harness = NGramPredictor(alphabet_size=8, n=5, laplace=1.0, adapt_online=True)
    p_harness.fit(split.train)
    harness_result = evaluate_sequence(
        p_harness,
        split.test,
        max_context_length=256,
        validate_probabilities=True,
    )

    p_manual = NGramPredictor(alphabet_size=8, n=5, laplace=1.0, adapt_online=True)
    p_manual.fit(split.train)
    manual_bps = _manual_tuple_evaluate(p_manual, split.test, max_context_length=256)

    assert harness_result.bits_per_symbol == pytest.approx(manual_bps, abs=1e-12)


def test_ngram_lower_order_tables_are_consistent_across_model_orders() -> None:
    """Order-k tables embedded in n=5 must match those from n=4 for k<=3."""

    rng = np.random.default_rng(123)
    sequence = rng.integers(0, 16, size=5000, dtype=np.int64)

    n4 = NGramPredictor(alphabet_size=16, n=4, laplace=1.0, adapt_online=False).fit(sequence)
    n5 = NGramPredictor(alphabet_size=16, n=5, laplace=1.0, adapt_online=False).fit(sequence)

    for order in range(4):
        table4 = n4._counts_by_order[order]
        table5 = n5._counts_by_order[order]
        assert table4.keys() == table5.keys()
        for key in table4:
            stats4 = table4[key]
            stats5 = table5[key]
            assert stats4.total == stats5.total
            assert stats4.counts == stats5.counts
