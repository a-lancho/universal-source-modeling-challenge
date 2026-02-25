# Student Submission Template

Students should submit a Python file (e.g. `predictor.py`) that defines:

```python
def build_predictor(alphabet_size: int, max_context_length: int) -> Predictor:
    ...
```

The live evaluator imports this function and runs the returned predictor sequentially on the released `test.npy`.

Score definition (shared notation): see `docs/notation.md`. In short, the evaluator computes the empirical average log-loss
`\widehat{H}_Q(x_1^N) = (1/N)\sum_{i=1}^N -\log_2 q_i(x_i \mid x_1^{i-1})`
in bits/symbol (lower is better).

## Starter Template

Use `submissions/template_predictor.py` as a starting point. It loads `data/generator/train.npy`, fits an `NGramPredictor` (`n=4`), and returns it.

## Example Commands

Smoke test (fast, evaluates 5000 symbols):

```bash
python -m competition.run_live_eval \
  --test-path data/live_release/test.npy \
  --predictor-path submissions/template_predictor.py \
  --smoke-test
```

Full default live eval (evaluates first `N=200000` symbols):

```bash
python -m competition.run_live_eval \
  --test-path data/live_release/test.npy \
  --predictor-path submissions/template_predictor.py
```

Notes:
- By default only the first `N=200000` symbols are evaluated (fixed prefix for fairness/runtime predictability).
- Runtime and context limits are enforced by the competition evaluator.
- No lookahead is allowed (sequential prediction only).

## Bonus (Course Grade Only): Arithmetic Coding Validation on Practice Set

This is optional and does **not** affect leaderboard ranking.

Recommended setup:
- Dataset: `data/public_practice/test.npy` only
- Short fixed prefix: `N_AC = 30000` symbols

What to compute/report (base-2 logs, units `bits/symbol`; see `docs/notation.md`):
- `Hhat_Q = (1/N_AC)\sum_{i=1}^{N_AC} -\log_2 q_i(x_i \mid x_1^{i-1})`
- `compressed_bits` from your arithmetic-coded bitstream
- `bps_AC = compressed_bits / N_AC`
- `Delta_AC = bps_AC - Hhat_Q`
- `DECODE_OK=True/False`

Requirements:
- Lossless decode is required (`DECODE_OK=True`).
- Your arithmetic coder may be your own implementation or a cited third-party reference implementation, but the integration with your sequential PMFs `q_i(\cdot \mid x_1^{i-1})` must be yours.
- Provide a short script/notebook cell that runs end-to-end and prints the values above.
- Practice set only; do not run this on the live-day secret test.
