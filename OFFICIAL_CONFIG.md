# Official Competition Configuration

This file is the single source of truth for the **official competition parameters** and the **official public baseline reference numbers** for this repo state.

No scoring, harness, generator, or live-eval logic is defined here; this document records the frozen values used for student-facing instructions and grading policy.

## Official Competition Parameters

- Alphabet size: `A = |\mathcal{X}| = 16`
- Default evaluated prefix length: `N = 200000` symbols
- `max_context_length = 256`
- `time_limit_seconds = 600`
- Leaderboard score: empirical average log-loss / empirical cross-entropy in base 2 (units: `bits/symbol`)
  - `\widehat{H}_Q(x_1^N) = (1/N)\sum_{i=1}^N -\log_2 q_i(x_i \mid x_1^{i-1})`

## Operational Ranking Policy

- Timeout disqualification policy: runs with `timed_out=True` are disqualified for official ranking.
- Ranking rule for valid non-timeout runs:
  1. `bits_per_symbol` (ascending; lower is better)
  2. `elapsed_seconds` (ascending) as tiebreaker

## Public Practice Dataset (Frozen Reference Context)

- Public practice seed: `123`
- Public practice test length: `300000` symbols
- Public practice file: `data/public_practice/test.npy`
- Public practice metadata: `data/public_practice/metadata.json`
- Live-day seed: **secret** (not published in advance)

## Official Classical Baseline (Public Practice Reference)

These values are **empirical on the public practice set** (`seed=123`) using the benchmark tool on the current repo state, evaluated on the official prefix length `N=200000`.

Benchmark command used:

```bash
python -m competition.benchmark_baselines \
  --test-path data/public_practice/test.npy \
  --train-path data/generator/train.npy \
  --num-tokens 200000
```

Frozen reference values (bits/symbol):

- Uniform baseline: `4.000000`
- Hard-backoff n-gram (`n=4`, `laplace=1.0`): `2.998625`
- Hard-backoff n-gram (`n=3`, `laplace=1.0`): `3.035208`

## Notes

- These baseline values are reference numbers for this repository state and public practice dataset only.
- They are not theoretical constants and may change if data, implementation, or evaluation settings change.
- This document does not alter evaluator defaults, live-eval output format, or scoring behavior.
