#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: bash competition/live_day_helper.sh <SECRET_SEED> [TEST_LENGTH] [OUT_DIR]" >&2
  exit 1
fi

SECRET_SEED="$1"
TEST_LENGTH="${2:-300000}"
OUT_DIR="${3:-data/live_release}"

python -m competition.release_test \
  --test-length "${TEST_LENGTH}" \
  --seed "${SECRET_SEED}" \
  --output-dir "${OUT_DIR}"

echo
echo "Released files in: ${OUT_DIR}"
ls -lh "${OUT_DIR}"
echo
echo "sha256 checksum:"
cat "${OUT_DIR}/sha256.txt"
echo
echo "Student command template:"
echo "python -m competition.run_live_eval --test-path ${OUT_DIR}/test.npy --predictor-path <THEIR_PREDICTOR.py>"
echo "Optional smoke test:"
echo "python -m competition.run_live_eval --test-path ${OUT_DIR}/test.npy --predictor-path <THEIR_PREDICTOR.py> --smoke-test"
