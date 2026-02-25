"""Tests for the count-threshold n-gram predictor baseline."""

from __future__ import annotations

import numpy as np
import pytest

from src.data.synthetic_source import HMMSourceConfig, build_random_hmm_source
from src.evaluation.harness import evaluate_sequence
from src.predictors.ngram import NGramPredictor
from src.predictors.ngram_threshold import CountThresholdNGramPredictor


def test_threshold_ngram_probabilities_sum_to_one() -> None:
    sequence = np.array([0, 1, 2, 0, 1, 3, 0, 1, 2, 0, 1, 3], dtype=np.int64)
    predictor = CountThresholdNGramPredictor(
        alphabet_size=4,
        n=3,
        laplace=1.0,
        min_count=2,
        adapt_online=False,
    ).fit(sequence)

    log_probs = predictor.predict_next((0, 1))
    probs = np.exp2(log_probs)

    assert log_probs.shape == (4,)
    assert np.all(np.isfinite(log_probs))
    assert float(np.sum(probs)) == pytest.approx(1.0, abs=1e-12)
    assert np.all(probs >= 0.0)


def test_threshold_ngram_forces_backoff_when_context_count_too_small() -> None:
    # Context (0, 1) is seen once; suffix context (1,) is seen three times.
    train = np.array([2, 1, 0, 1, 2, 1, 3, 4], dtype=np.int64)

    hard = NGramPredictor(alphabet_size=5, n=3, laplace=1.0, adapt_online=False).fit(train)
    threshold = CountThresholdNGramPredictor(
        alphabet_size=5,
        n=3,
        laplace=1.0,
        min_count=2,
        adapt_online=False,
    ).fit(train)

    # Hard backoff uses the seen order-2 context (0,1); thresholded model backs off to (1,).
    hard_order, _ = hard._lookup_backoff_stats((0, 1))
    th_order, _ = threshold._lookup_backoff_stats((0, 1))

    assert hard_order == 2
    assert th_order == 1

    hard_probs = np.exp2(hard.predict_next((0, 1)))
    th_probs = np.exp2(threshold.predict_next((0, 1)))
    assert not np.allclose(hard_probs, th_probs)


def test_threshold_ngram_runs_on_small_synthetic_split() -> None:
    config = HMMSourceConfig(
        num_states=4,
        alphabet_size=8,
        transition_concentration=0.6,
        emission_concentration=0.25,
        self_transition_bias=4.0,
        burn_in=256,
        seed=21,
    )
    source = build_random_hmm_source(config)
    split = source.generate_train_test_split(
        train_length=4000,
        test_length=2000,
        seed=9,
        burn_in=config.burn_in,
    )

    predictor = CountThresholdNGramPredictor(
        alphabet_size=config.alphabet_size,
        n=5,
        laplace=1.0,
        min_count=5,
        adapt_online=True,
    )
    predictor.fit(split.train)

    result = evaluate_sequence(
        predictor,
        split.test,
        max_context_length=256,
        validate_probabilities=True,
    )

    assert result.num_tokens == len(split.test)
    assert np.isfinite(result.bits_per_symbol)
    assert result.bits_per_symbol > 0.0
    assert result.timed_out is False

