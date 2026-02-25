# Universal Source Modeling Challenge

Research-oriented framework for evaluating sequential probabilistic predictors under computational constraints.

## What The Score Means

Notation reference (shared across repo): `docs/notation.md`.
Official competition parameters and frozen public baseline reference numbers: `OFFICIAL_CONFIG.md` (single source of truth for competition configuration/policy values).

The primary score is empirical average log-loss / empirical cross-entropy on a hidden evaluated sequence `x_1^N` (all logs base 2):

`\widehat{H}_Q(x_1^N) = (1/N) * sum_{i=1}^N -log2 q_i(x_i | x_1^{i-1})`

- Units: `bits per symbol`
- Lower is better
- This is the leaderboard score (direct log-loss)
- It has an ideal codelength interpretation under predictor `Q`, but it is not an achieved compressed file size because scoring does not run arithmetic coding

## Generate Synthetic Data (Instructor Side)

Generate a reproducible HMM-based stationary source split:

```bash
python -m src.data.synthetic_source \
  --train-length 100000 \
  --test-length 100000 \
  --num-states 8 \
  --alphabet-size 16 \
  --seed 0 \
  --output-dir data/generator
```

Notes:
- The generator prints an approximate entropy-rate estimate (in bits/symbol).
- By default, exported `metadata.json` is public-safe and does not reveal hidden HMM parameters.
- Use `--include-private-details` only for instructor-side private artifacts.

## Run Baselines

Uniform baseline:

```bash
python -m baselines.run_baseline \
  --data-dir data/generator \
  --baseline uniform
```

Laplace-smoothed n-gram baseline (default `n=3`, online sequential updates enabled):

```bash
python -m baselines.run_baseline \
  --data-dir data/generator \
  --baseline ngram \
  --ngram-n 3 \
  --laplace 1.0 \
  --max-context-length 256
```

Optional runtime cap:

```bash
python -m baselines.run_baseline \
  --data-dir data/generator \
  --baseline ngram \
  --max-seconds 30
```

Note:
- For pure suffix-backoff + Laplace-smoothed n-grams, test performance is not guaranteed to improve monotonically as `n` increases (higher-order contexts can be too sparse).
- Recommended "official simple baseline": hard-backoff n-gram with `n=4`.

Count-threshold backoff n-gram (stronger classical reference baseline):

```bash
python -m baselines.run_baseline \
  --data-dir data/generator \
  --baseline ngram_threshold \
  --ngram-n 5 \
  --laplace 1.0 \
  --min-count 5 \
  --max-context-length 256
```

This variant only uses a context if its count is at least `min_count`; otherwise it backs off to a shorter suffix context, which often reduces high-order sparsity overfitting.

## Optional Sanity Check (Bonus)

This optional tool demonstrates the prediction-compression connection by comparing:
- model-based cross-entropy code length (bits/symbol)
- standard compressors (`zlib`, `lzma`, `bz2`) on the same symbol sequence bytes

This is a verification/teaching aid only. It does **not** affect leaderboard scoring.

Example:

```bash
python -m baselines.run_sanity_check \
  --data-dir data/generator \
  --split test \
  --baseline ngram_threshold \
  --ngram-n 5 \
  --min-count 8
```

## Run Tests

```bash
pytest -q
```

The n-gram test suite includes an `xfail` test documenting the non-monotonicity anomaly reported on the HMM benchmark config.
