# Score Notation (Source of Truth)

Use this notation consistently in student-facing materials.

## Sequence, Alphabet, and Predictor

- Alphabet: `\mathcal{X}` with size `A = |\mathcal{X}|` (fixed finite alphabet in this competition; default `A=16`)
- Random sequence: `X_1^N = (X_1, \dots, X_N)`
- Realized evaluated sequence: `x_1^N = (x_1, \dots, x_N)`
- Sequential predictor outputs a time-varying conditional PMF:
  `q_i(\cdot \mid x_1^{i-1})` for `i=1,\dots,N`
- No lookahead: `q_i` may depend only on past symbols `x_1^{i-1}`

## Leaderboard Score (bits/symbol)

All logs are base 2. The evaluated score is the empirical average log-loss (empirical cross-entropy of the realized sequence under predictor `Q`):

`\widehat{H}_Q(x_1^N) \triangleq \frac{1}{N}\sum_{i=1}^N -\log_2 q_i(x_i \mid x_1^{i-1})`

- Units: `bits/symbol`
- Lower is better
- This is the leaderboard score

Interpretation:
- Each term `-\log_2 q_i(x_i \mid x_1^{i-1})` is an idealized codelength contribution under the model probabilities.
- In this competition, scoring is the log-loss directly (we do not run arithmetic coding in the evaluator), so the score is not an achieved compressed file size.

## Distributional Definitions (base 2)

For PMFs `P,Q` on a common alphabet:

- Entropy: `H(P) = -\mathbb{E}_P[\log_2 P(X)]`
- Cross-entropy: `H(P,Q) \triangleq -\mathbb{E}_P[\log_2 Q(X)]`
- KL divergence: `D(P\|Q) = \mathbb{E}_P\!\left[\log_2 \frac{P(X)}{Q(X)}\right]`
- Identity: `H(P,Q)=H(P)+D(P\|Q)`

For stationary ergodic sources, the empirical average log-loss above is often discussed as an empirical estimate / upper-bound proxy for entropy-rate-related quantities, but finite-`N` effects and sampling noise matter.

## Connection to LLMZip (short)

- LLMZip uses the same numerator `\sum_i -\log_2 q_i(x_i)` (bits per token before normalization).
- It then converts to bits/character by dividing by average characters per token.
- Our competition uses a fixed finite symbol alphabet and reports `bits/symbol` directly.
- LLMZip discusses arithmetic coding as the near-optimal way to realize the codelength; our leaderboard does not implement arithmetic coding and scores log-loss directly.
