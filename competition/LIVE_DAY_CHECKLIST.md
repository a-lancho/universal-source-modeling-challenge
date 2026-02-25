# Live Day Checklist (Instructor)

Practical checklist for running the Universal Source Modeling Challenge live in class.

## A) Pre-Class Prep (Day Before)

- Verify repo state:

```bash
pytest -q
```

- Optional quick readiness scan (files/data/config, plus optional bundle/test checks):

```bash
python -m competition.live_day_quick_check
```

- Ensure training data exists:
  - `data/generator/train.npy`

- Ensure practice set exists (optional but recommended):
  - `data/public_practice/test.npy`
  - `data/public_practice/metadata.json`
  - `data/public_practice/sha256.txt`

- Build a fresh student bundle:

```bash
python -m competition.make_student_bundle --output-dir /tmp/student_bundle --overwrite
```

- Zip the bundle and upload/share it with students.

## B) Competition Day (Instructor Steps)

### 1. Final environment check

Optional full check:

```bash
pytest -q
```

Or quick import sanity check:

```bash
python -c "import competition.run_live_eval, competition.release_test; print('OK')"
```

### 2. Generate the secret live test

```bash
python -m competition.release_test \
  --test-length 300000 \
  --seed <SECRET_SEED> \
  --output-dir data/live_release
```

Optional helper script (wraps the command above):

```bash
bash competition/live_day_helper.sh <SECRET_SEED> [TEST_LENGTH] [OUT_DIR]
```

### 3. Confirm files exist and record checksum

```bash
ls -lh data/live_release
cat data/live_release/sha256.txt
```

### 4. Distribute only the released test artifacts

- Distribute `test.npy` to students.
- Optionally also share `metadata.json` and `sha256.txt`.
- Remind students: the required evaluated prefix is **200000** tokens.

### 5. Students run live evaluation

Student command template:

```bash
python -m competition.run_live_eval \
  --test-path <PATH_TO_TEST.NPY> \
  --predictor-path <THEIR_PREDICTOR.py>
```

Optional first check:

```bash
python -m competition.run_live_eval \
  --test-path <PATH_TO_TEST.NPY> \
  --predictor-path <THEIR_PREDICTOR.py> \
  --smoke-test
```

## C) Collecting and Ranking Results

- Ask each team to paste one line, optionally prefixed with a team label:
  - `TEAM=<name> FINAL_SCORE ...`
  - or just `FINAL_SCORE ...`

- Save all pasted lines into a text file (example: `scores_live.txt`).

- Rank with timeout disqualification and required-token enforcement:

```bash
python -m competition.collect_scores \
  --scores-path scores_live.txt \
  --required-tokens 200000 \
  --disqualify-timeouts
```

## D) Common Failure Modes and Fixes

- Wrong `evaluated_tokens`:
  - Rerun without overriding defaults; avoid incorrect `--num-tokens`.
- `timed_out=True`:
  - Simplify predictor, reduce overhead, cache expensive work, shrink model.
- Missing dependencies in Colab:
  - `pip install` required packages, then restart runtime if needed.
- Wrong test path:
  - Verify the file exists in Colab and the path is correct.
- Slow run:
  - Use `--smoke-test` first, then run full evaluation.
- Hash mismatch:
  - Re-download `test.npy` and re-check `sha256.txt`.

## E) Policy Reminders (from `COMPETITION_RULES.md`)

- Timeouts are disqualified.
- Runtime is only a tiebreaker among valid non-timeout runs.
- No external API calls during live evaluation.
- Do not modify evaluation scripts.
