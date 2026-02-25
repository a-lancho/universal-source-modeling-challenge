# Grading Rubric: Universal Source Modeling Challenge

Audience: students, instructors, and TAs. This rubric is course-facing and aligned with the Information Theory framing used in `HANDOUT_INFO_THEORY.md` and `docs/notation.md`.

Primary references:
- `OFFICIAL_CONFIG.md` (official competition parameters + frozen public baseline references)
- `COMPETITION_RULES.md` (live-day rules, ranking validity, timeout policy)
- `HANDOUT_INFO_THEORY.md` (IT framing and interpretation)
- `docs/notation.md` (score notation and definitions)

Global timeout grading cap (official live-day run):
- If `timed_out=True` on the official live-day run:
  - the team is disqualified from ranking
  - `A2` (ranking bonus) = `0`
  - the total assignment grade is capped at `75/100`
  - bonus points do not override this cap

## Grading Structure (100 points total + optional bonus)

- A) Performance (bits/symbol): `40 pts`
- B) Efficiency & reliability: `20 pts`
- C) Information-theoretic understanding: `25 pts`
- D) Communication/presentation & experimental clarity: `15 pts`
- Bonus) Arithmetic coding validation (practice set only): `+5 pts` extra credit

## A) Performance (Bits/Symbol) — 40 pts

- A1) Absolute improvement vs baseline (`Δ`) — `35 pts`
- A2) Ranking position bonus — `5 pts`

Official baseline definition (method):
- Hard-backoff n-gram with `n=4`, `laplace=1.0` (see `OFFICIAL_CONFIG.md`)

Frozen public practice reference (for expectations only):
- Empirical score on public practice set (`seed=123`, official prefix): approximately `2.9986` bits/symbol (see `OFFICIAL_CONFIG.md`)

Grading basis:
- Performance points are based on the **live-day evaluation on the secret test set**.
- Instructors/TAs should run the official baseline method on the same live-day test set and official config.
- Let:
  - `b_base` = official baseline live-day score (bits/symbol)
  - `b_student` = student live-day score (bits/symbol)
  - `Δ = b_base - b_student` (absolute improvement in bits/symbol; larger is better)

A1 point mapping (use `Δ` on the secret live-day set, max `35 pts`):
- `Δ <= 0.000`: `5 pts` (floor; low credit if not beating baseline)
- `0.000 < Δ <= 0.020`: `16 pts`
- `0.020 < Δ <= 0.050`: `22 pts`
- `0.050 < Δ <= 0.100`: `28 pts`
- `0.100 < Δ <= 0.200`: `32 pts`
- `Δ > 0.200`: `35 pts`

A2 ranking bonus (`5 pts`, small-cohort bonus):
- Let `M` = number of valid teams (`evaluated_tokens` correct and `timed_out=False`)
- Let `R` = rank among valid teams (`R=1` is best)
- Compute `s = (M - R) / max(1, M-1)`
- Award `A2 = 5*s`
- Round `A2` to the nearest `0.5` point for simplicity
- Edge case: if `M=1`, that team receives full `5 pts`

Worked example (`M=6` valid teams):
- `R=1` → `A2 = 5.0`
- `R=2` → `A2 = 4.0`
- `R=3` → `A2 = 3.0`
- `R=4` → `A2 = 2.0`
- `R=5` → `A2 = 1.0`
- `R=6` → `A2 = 0.0`

Notes:
- Lower `bits/symbol` is better.
- If no valid live-day score is produced (e.g., invalid run), award `0 pts` in this section (`A1=0`, `A2=0`).
- `elapsed_seconds` remains a ranking tiebreaker per `COMPETITION_RULES.md`; it does not directly affect `A1`.

## B) Efficiency & Reliability — 20 pts

This section rewards valid, reproducible execution under the official constraints (`OFFICIAL_CONFIG.md`, `COMPETITION_RULES.md`).

### B1. Reliability / protocol compliance — 10 pts

Award `10 pts` only if all are true on the official run:
- Uses the provided predictor interface (`build_predictor(alphabet_size, max_context_length)`)
- Runs without manual intervention during evaluation
- Produces a valid `FINAL_SCORE ...` line
- `evaluated_tokens` matches the required official prefix length
- No protocol violations (e.g., lookahead, modified evaluation script)

Award `0 pts` if any of the above fail.

### B2. Runtime efficiency under constraints — 10 pts

Scoring rule (official live-day run):
- `timed_out=True`: `0 pts`
- Finishes within time limit and `elapsed_seconds <= 600`: `8 pts`
- Finishes within time limit and `elapsed_seconds <= 300`: `9 pts`
- Finishes within time limit and `elapsed_seconds <= 120`: `10 pts`

Policy note:
- Per `COMPETITION_RULES.md`, `timed_out=True` is disqualified for ranking.
- For grading, a timeout also receives `0 pts` in this runtime subsection.

## C) Information-Theoretic Understanding — 25 pts

Students must submit a short written report (`1–2 pages`) that explains their method using the course notation and IT concepts.

Checklist + points:
- Score definition using repo/course notation (`docs/notation.md`) — `5 pts`
  - Correctly defines `x_1^N`, `q_i(\cdot \mid x_1^{i-1})`, and
    `\widehat{H}_Q(x_1^N) = (1/N)\sum_{i=1}^N -\log_2 q_i(x_i \mid x_1^{i-1})`
  - States base-2 logs and units (`bits/symbol`)
- Coding interpretation (idealized codelength; arithmetic coding context) — `5 pts`
  - Explains `-\log_2 q_i(...)` as idealized codelength contribution
  - Notes that leaderboard scoring is direct log-loss (not achieved compressed size)
- Redundancy / mismatch explanation via `H(P,Q)=H(P)+D(P\|Q)` — `5 pts`
  - Correctly identifies the KL term as mismatch / excess cross-entropy
  - Avoids over-claiming finite-sample equality from one run
- Constraints and universal prediction discussion — `5 pts`
  - Connects context limit and time limit to sequential prediction under constraints
  - Explains why more context may not always help (sparsity / estimation tradeoff is enough)
- At least one method-specific IT insight — `5 pts`
  - Example: smoothing/backoff/interpolation as mismatch reduction
  - Example: online adaptation vs pre-fit tradeoff
  - Example: runtime-memory tradeoff and practical universality

## D) Communication / Presentation & Experimental Clarity — 15 pts

Evaluate the quality of the written/slides presentation and the clarity of empirical evidence.

Checklist + points:
- Clear method description (what was implemented, what data/updates are used online) — `4 pts`
- Ablations or controlled comparisons (at least 2) — `4 pts`
  - Examples: n-gram order, smoothing strength, thresholding, context length, caching
- Runtime reporting and operational details — `3 pts`
  - Include runtime, whether smoke/full runs were used, and any failure modes observed
- Clear final takeaway — `4 pts`
  - What worked, what did not, and why (supported by results)

## Submission Requirements

### Live day (required)

Submit the final evaluator line exactly as produced:
- `FINAL_SCORE bits_per_symbol=... elapsed_seconds=... timed_out=... evaluated_tokens=...`

Allowed prefix when reporting to instructor/TA:
- `TEAM=<name> FINAL_SCORE ...`

See `COMPETITION_RULES.md` and `competition/README.md` for the canonical output format and live-day protocol.

### After live day (course deliverables)

Submit:
- Code (`predictor.py` file or repo link with the predictor implementation used)
- Short report (`1–2 pages`) covering the items in Section C
- Slides / presentation materials (if required by instructor)

### Academic Integrity (must follow)

- No test-set peeking or manual inspection beyond normal evaluation usage
- No lookahead (no use of future test symbols when predicting current symbol)
- No external API calls during live evaluation
- Do not modify the provided evaluation script/protocol for official runs

Violations may result in invalidation of the live-day result and additional course penalties per instructor policy.

## TA Grading Notes (Suggested Workflow)

- Verify live-day validity first using the submitted `FINAL_SCORE` line and official logs.
- Apply Section B (reliability/runtime) before assigning performance points.
- Compute performance points using the secret-set baseline delta `Δ = b_base - b_student`.
- Grade report/slides using Sections C and D with this rubric as a checklist.
- Grade the optional arithmetic-coding validation bonus separately; it does not change leaderboard ranking.

## Bonus (+5 pts): Arithmetic Coding Validation (Practice Set Only)

This is an optional **course-grade bonus** only. It does **not** affect leaderboard ranking and is **not** run on the live-day secret test set.
The arithmetic-coding bonus may increase a student's score within the assignment, but the final assignment grade is capped at `100/100`.

Scope and fairness (fixed conditions):
- Dataset: `data/public_practice/test.npy` only
- Prefix length: use a fixed short prefix `N_AC = 30000` symbols
- Score/notation: use the notation in `docs/notation.md` (base-2 logs, units `bits/symbol`)

What the student must demonstrate:
1. Compute model log-loss on the chosen prefix:
   - `\widehat{H}_Q = (1/N_AC)\sum_{i=1}^{N_AC} -\log_2 q_i(x_i \mid x_1^{i-1})` (bits/symbol)
2. Implement (or use a cited reference implementation of) an arithmetic coder that sequentially uses `q_i(\cdot \mid x_1^{i-1})` to encode `x_1^{N_AC}`
3. Lossless decoding:
   - `decode(encoded_bits)` must reproduce exactly `x_1^{N_AC}`
4. Report realized compressed rate:
   - `bps_AC = compressed_bits / N_AC`
5. Report gap:
   - `Delta_AC = bps_AC - \widehat{H}_Q`
   - Explain expected small positive overhead (finite precision / renormalization / termination)

Reproducibility requirements:
- Provide a short script or notebook cell that runs end-to-end on the public practice set and prints:
  - `N_AC`, `Hhat_Q`, `compressed_bits`, `bps_AC`, `Delta_AC`, `DECODE_OK=True/False`
- If third-party arithmetic coding code is used, provide a citation.
- The integration from the student predictor PMFs `q_i` into the encoder/decoder workflow must be the student's own work.

Bonus scoring (`+5 pts` max):
- `+5`: lossless decode passes and `Delta_AC <= 0.05` bits/symbol
- `+3`: lossless decode passes and `Delta_AC <= 0.10` bits/symbol
- `+1`: lossless decode passes but `Delta_AC > 0.10` bits/symbol, with a reasonable explanation of overhead/precision issues
- `0`: decoding fails, or the comparison is not reproducible
