"""Synthetic stationary source generation using a configurable hidden Markov model.

This module is designed for instructor-side dataset generation. It supports:
- Reproducible random HMM construction from a seed
- Stationary initialization
- Efficient sequential sampling for large sequences
- Train/test split export
- Approximate entropy-rate estimation via true one-step predictive log-loss
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray


FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


@dataclass(frozen=True)
class HMMSourceConfig:
    """Configuration for random HMM source generation.

    Tuning notes:
    - Larger `self_transition_bias` -> longer latent persistence (more memory).
    - Smaller `emission_concentration` -> sharper state-specific emissions (lower entropy).
    - Larger `emission_concentration` -> flatter emissions (higher entropy).
    """

    num_states: int = 8
    alphabet_size: int = 16
    transition_concentration: float = 0.5
    emission_concentration: float = 0.3
    self_transition_bias: float = 4.0
    burn_in: int = 4096
    seed: int = 0

    def validate(self) -> None:
        if self.num_states <= 0:
            raise ValueError("num_states must be positive.")
        if self.alphabet_size <= 1:
            raise ValueError("alphabet_size must be at least 2.")
        if self.transition_concentration <= 0:
            raise ValueError("transition_concentration must be positive.")
        if self.emission_concentration <= 0:
            raise ValueError("emission_concentration must be positive.")
        if self.self_transition_bias < 0:
            raise ValueError("self_transition_bias must be non-negative.")
        if self.burn_in < 0:
            raise ValueError("burn_in must be non-negative.")


@dataclass(frozen=True)
class GeneratedSplit:
    """Generated train/test split and metadata."""

    train: IntArray
    test: IntArray
    metadata: dict[str, Any]


class HMMSource:
    """Stationary hidden Markov source over a finite alphabet."""

    def __init__(
        self,
        transition_matrix: FloatArray,
        emission_matrix: FloatArray,
        initial_distribution: FloatArray | None = None,
    ) -> None:
        transition_matrix = np.asarray(transition_matrix, dtype=np.float64)
        emission_matrix = np.asarray(emission_matrix, dtype=np.float64)

        if transition_matrix.ndim != 2 or transition_matrix.shape[0] != transition_matrix.shape[1]:
            raise ValueError("transition_matrix must be square with shape (S, S).")
        if emission_matrix.ndim != 2 or emission_matrix.shape[0] != transition_matrix.shape[0]:
            raise ValueError("emission_matrix must have shape (S, A).")
        if emission_matrix.shape[1] < 2:
            raise ValueError("Alphabet size must be at least 2.")
        if np.any(transition_matrix < 0) or np.any(emission_matrix < 0):
            raise ValueError("Matrices must be non-negative.")

        self.transition_matrix = self._normalize_rows(transition_matrix, "transition_matrix")
        self.emission_matrix = self._normalize_rows(emission_matrix, "emission_matrix")
        self.num_states = int(self.transition_matrix.shape[0])
        self.alphabet_size = int(self.emission_matrix.shape[1])

        if initial_distribution is None:
            self.initial_distribution = self.stationary_distribution(self.transition_matrix)
        else:
            initial_distribution = np.asarray(initial_distribution, dtype=np.float64)
            if initial_distribution.shape != (self.num_states,):
                raise ValueError(
                    f"initial_distribution must have shape ({self.num_states},)."
                )
            if np.any(initial_distribution < 0):
                raise ValueError("initial_distribution must be non-negative.")
            total = float(np.sum(initial_distribution))
            if total <= 0:
                raise ValueError("initial_distribution must sum to a positive value.")
            self.initial_distribution = initial_distribution / total

        self._transition_cdf = np.cumsum(self.transition_matrix, axis=1)
        self._emission_cdf = np.cumsum(self.emission_matrix, axis=1)
        self._initial_cdf = np.cumsum(self.initial_distribution)

        # Ensure the last CDF entry is exactly 1.0 to avoid edge-case search issues.
        self._transition_cdf[:, -1] = 1.0
        self._emission_cdf[:, -1] = 1.0
        self._initial_cdf[-1] = 1.0

    @staticmethod
    def _normalize_rows(matrix: FloatArray, name: str) -> FloatArray:
        row_sums = np.sum(matrix, axis=1, keepdims=True)
        if np.any(row_sums <= 0):
            raise ValueError(f"{name} has a row with non-positive sum.")
        return matrix / row_sums

    @staticmethod
    def stationary_distribution(
        transition_matrix: FloatArray,
        *,
        max_iter: int = 10000,
        tol: float = 1e-12,
    ) -> FloatArray:
        """Compute a stationary distribution via power iteration."""

        num_states = transition_matrix.shape[0]
        dist = np.full(num_states, 1.0 / num_states, dtype=np.float64)
        for _ in range(max_iter):
            next_dist = dist @ transition_matrix
            if np.max(np.abs(next_dist - dist)) < tol:
                break
            dist = next_dist
        dist = dist / np.sum(dist)
        return dist

    @classmethod
    def random(cls, config: HMMSourceConfig) -> "HMMSource":
        """Create a random ergodic HMM source from a config and seed."""

        config.validate()
        rng = np.random.default_rng(config.seed)

        trans_alpha = np.full(config.num_states, config.transition_concentration, dtype=np.float64)
        transition = rng.dirichlet(trans_alpha, size=config.num_states)
        if config.self_transition_bias > 0:
            transition = transition + config.self_transition_bias * np.eye(config.num_states)
            transition = transition / np.sum(transition, axis=1, keepdims=True)

        emit_alpha = np.full(config.alphabet_size, config.emission_concentration, dtype=np.float64)
        emission = rng.dirichlet(emit_alpha, size=config.num_states)

        return cls(transition_matrix=transition, emission_matrix=emission)

    @staticmethod
    def _sample_from_cdf(rng: np.random.Generator, cdf: FloatArray) -> int:
        return int(np.searchsorted(cdf, rng.random(), side="right"))

    def _generate_sequence_with_predictive_bits(
        self,
        length: int,
        *,
        seed: int,
        burn_in: int,
        return_states: bool = False,
    ) -> tuple[IntArray, FloatArray, IntArray | None]:
        """Generate a sequence and predictive bits under the true HMM.

        Predictive bits are computed as `-log2 p(x_t | x_<t)` using HMM filtering.
        Their empirical average is a useful entropy-rate estimate for the source.
        """

        if length < 0:
            raise ValueError("length must be non-negative.")
        if burn_in < 0:
            raise ValueError("burn_in must be non-negative.")

        rng = np.random.default_rng(seed)
        total_steps = burn_in + length

        symbols = np.empty(length, dtype=np.int64)
        states = np.empty(length, dtype=np.int64) if return_states else None
        predictive_bits = np.empty(length, dtype=np.float64)

        # Hidden state sampling starts from the stationary distribution.
        current_state = self._sample_from_cdf(rng, self._initial_cdf)
        posterior = self.initial_distribution.copy()

        out_idx = 0
        for t in range(total_steps):
            if t > 0:
                current_state = self._sample_from_cdf(rng, self._transition_cdf[current_state])
                predictive_state_dist = posterior @ self.transition_matrix
            else:
                predictive_state_dist = posterior

            symbol = self._sample_from_cdf(rng, self._emission_cdf[current_state])
            predictive_symbol_dist = predictive_state_dist @ self.emission_matrix
            symbol_prob = float(predictive_symbol_dist[symbol])

            if symbol_prob <= 0.0:
                raise RuntimeError("Generated an impossible symbol under the HMM predictive model.")

            # HMM filter update: p(z_t | x_<=t) proportional to p(z_t | x_<t) p(x_t | z_t).
            posterior_unnorm = predictive_state_dist * self.emission_matrix[:, symbol]
            norm = float(np.sum(posterior_unnorm))
            if norm <= 0.0:
                raise RuntimeError("Filtering posterior collapsed to zero.")
            posterior = posterior_unnorm / norm

            if t >= burn_in:
                symbols[out_idx] = symbol
                predictive_bits[out_idx] = -np.log2(symbol_prob)
                if states is not None:
                    states[out_idx] = current_state
                out_idx += 1

        return symbols, predictive_bits, states

    def sample(
        self,
        length: int,
        *,
        seed: int = 0,
        burn_in: int = 0,
        return_states: bool = False,
    ) -> tuple[IntArray, FloatArray] | tuple[IntArray, FloatArray, IntArray]:
        """Sample a sequence and return predictive bits for entropy estimation."""

        symbols, predictive_bits, states = self._generate_sequence_with_predictive_bits(
            length, seed=seed, burn_in=burn_in, return_states=return_states
        )
        if return_states:
            assert states is not None
            return symbols, predictive_bits, states
        return symbols, predictive_bits

    def generate_train_test_split(
        self,
        *,
        train_length: int,
        test_length: int,
        seed: int = 0,
        burn_in: int = 0,
    ) -> GeneratedSplit:
        """Generate a contiguous train/test split from one stationary sample path."""

        if train_length <= 0 or test_length <= 0:
            raise ValueError("train_length and test_length must be positive.")

        total_length = train_length + test_length
        sequence, predictive_bits = self.sample(
            total_length, seed=seed, burn_in=burn_in, return_states=False
        )
        train = sequence[:train_length].copy()
        test = sequence[train_length:].copy()

        train_bits = predictive_bits[:train_length]
        test_bits = predictive_bits[train_length:]

        metadata = {
            "train_length": train_length,
            "test_length": test_length,
            "seed": seed,
            "burn_in": burn_in,
            "num_states": self.num_states,
            "alphabet_size": self.alphabet_size,
            "approx_entropy_bps_full": float(np.mean(predictive_bits)),
            "approx_entropy_bps_train": float(np.mean(train_bits)),
            "approx_entropy_bps_test": float(np.mean(test_bits)),
            "empirical_zero_order_entropy_train_bps": float(
                empirical_zero_order_entropy(train, self.alphabet_size)
            ),
            "empirical_zero_order_entropy_test_bps": float(
                empirical_zero_order_entropy(test, self.alphabet_size)
            ),
        }
        return GeneratedSplit(train=train, test=test, metadata=metadata)

    def save_split(
        self,
        split: GeneratedSplit,
        output_dir: str | Path,
        *,
        train_filename: str = "train.npy",
        test_filename: str = "test.npy",
        metadata_filename: str = "metadata.json",
        include_private_details: bool = False,
    ) -> None:
        """Save a generated split to disk.

        By default, only public metadata is saved (lengths and alphabet size) so the
        hidden generative process is not exposed to students.
        """

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        np.save(out / train_filename, split.train)
        np.save(out / test_filename, split.test)

        metadata: dict[str, Any] = {
            "train_length": int(len(split.train)),
            "test_length": int(len(split.test)),
            "alphabet_size": int(self.alphabet_size),
        }
        if include_private_details:
            metadata.update(split.metadata)
            metadata["transition_matrix"] = self.transition_matrix.tolist()
            metadata["emission_matrix"] = self.emission_matrix.tolist()
            metadata["initial_distribution"] = self.initial_distribution.tolist()

        with (out / metadata_filename).open("w", encoding="utf-8") as fh:
            json.dump(metadata, fh, indent=2)


def empirical_zero_order_entropy(sequence: IntArray, alphabet_size: int) -> float:
    """Plug-in entropy estimate H(X) from symbol frequencies (not entropy rate)."""

    if len(sequence) == 0:
        return float("nan")
    counts = np.bincount(sequence, minlength=alphabet_size).astype(np.float64)
    probs = counts / np.sum(counts)
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def build_random_hmm_source(config: HMMSourceConfig) -> HMMSource:
    """Convenience factory used by scripts and tests."""

    return HMMSource.random(config)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic HMM train/test split.")
    parser.add_argument("--train-length", type=int, default=100_000)
    parser.add_argument("--test-length", type=int, default=100_000)
    parser.add_argument("--num-states", type=int, default=8)
    parser.add_argument("--alphabet-size", type=int, default=16)
    parser.add_argument("--transition-concentration", type=float, default=0.5)
    parser.add_argument("--emission-concentration", type=float, default=0.3)
    parser.add_argument("--self-transition-bias", type=float, default=4.0)
    parser.add_argument("--burn-in", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/generator",
        help="Where to write train/test .npy files and metadata.json",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Generate and report stats without writing files.",
    )
    parser.add_argument(
        "--include-private-details",
        action="store_true",
        help="Include hidden HMM parameters in metadata.json (instructor-only).",
    )
    args = parser.parse_args()

    config = HMMSourceConfig(
        num_states=args.num_states,
        alphabet_size=args.alphabet_size,
        transition_concentration=args.transition_concentration,
        emission_concentration=args.emission_concentration,
        self_transition_bias=args.self_transition_bias,
        burn_in=args.burn_in,
        seed=args.seed,
    )
    source = build_random_hmm_source(config)
    split = source.generate_train_test_split(
        train_length=args.train_length,
        test_length=args.test_length,
        seed=args.seed,
        burn_in=args.burn_in,
    )

    print("Synthetic HMM source split generated")
    print(f"  train length: {len(split.train)}")
    print(f"  test length:  {len(split.test)}")
    print(f"  alphabet size: {source.alphabet_size}")
    print(f"  hidden states: {source.num_states}")
    print(
        "  approx entropy rate (true predictive, bps): "
        f"{split.metadata['approx_entropy_bps_full']:.4f}"
    )
    print(
        "  train/test predictive entropy estimate (bps): "
        f"{split.metadata['approx_entropy_bps_train']:.4f} / "
        f"{split.metadata['approx_entropy_bps_test']:.4f}"
    )
    print(
        "  train/test zero-order entropy (bps): "
        f"{split.metadata['empirical_zero_order_entropy_train_bps']:.4f} / "
        f"{split.metadata['empirical_zero_order_entropy_test_bps']:.4f}"
    )

    if not args.no_save:
        source.save_split(
            split, args.output_dir, include_private_details=args.include_private_details
        )
        print(f"  saved to: {args.output_dir}")


if __name__ == "__main__":
    _cli()
