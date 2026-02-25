"""Tiny instructor pre-class readiness checker for live competition day."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CheckResult:
    name: str
    status: str  # PASS | WARN | FAIL
    detail: str


def _file_exists_check(path: Path, *, name: str, warn: bool = False) -> CheckResult:
    if path.exists():
        return CheckResult(name=name, status="PASS", detail=str(path))
    return CheckResult(
        name=name,
        status="WARN" if warn else "FAIL",
        detail=f"Missing: {path}",
    )


def _parse_simple_yaml_kv(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    if not path.exists():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def _run_pytest(repo_root: Path) -> CheckResult:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q"],
        cwd=repo_root,
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        tail = proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "pytest passed"
        return CheckResult(name="pytest -q", status="PASS", detail=tail)
    detail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
    return CheckResult(
        name="pytest -q",
        status="FAIL",
        detail=detail[-1] if detail else f"pytest failed (code {proc.returncode})",
    )


def _check_bundle_build(repo_root: Path) -> CheckResult:
    with tempfile.TemporaryDirectory() as tmpdir:
        out_dir = Path(tmpdir) / "bundle"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "competition.make_student_bundle",
                "--output-dir",
                str(out_dir),
                "--overwrite",
            ],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if proc.returncode == 0:
            return CheckResult(
                name="bundle build",
                status="PASS",
                detail=f"Built temp bundle at {out_dir}",
            )
        detail = (proc.stdout + "\n" + proc.stderr).strip().splitlines()
        return CheckResult(
            name="bundle build",
            status="FAIL",
            detail=detail[-1] if detail else f"bundle build failed (code {proc.returncode})",
        )


def run_checks(*, run_tests: bool, check_bundle: bool) -> list[CheckResult]:
    repo_root = Path.cwd()
    results: list[CheckResult] = []

    required_files = [
        "COMPETITION_RULES.md",
        "competition/run_live_eval.py",
        "competition/release_test.py",
        "competition/release_practice.py",
        "competition/collect_scores.py",
        "competition/make_student_bundle.py",
        "competition/verify_hashes.py",
        "competition/LIVE_DAY_CHECKLIST.md",
        "notebooks/colab_starter.ipynb",
        "submissions/template_predictor.py",
    ]
    for rel in required_files:
        results.append(_file_exists_check(repo_root / rel, name=f"repo file: {rel}"))

    results.append(
        _file_exists_check(repo_root / "data/generator/train.npy", name="train data: data/generator/train.npy")
    )
    for rel in [
        "data/public_practice/test.npy",
        "data/public_practice/metadata.json",
        "data/public_practice/sha256.txt",
    ]:
        results.append(_file_exists_check(repo_root / rel, name=f"practice file: {rel}", warn=True))

    live_dir = repo_root / "data/live_release"
    if live_dir.exists():
        results.append(CheckResult(name="live release dir (optional)", status="PASS", detail=str(live_dir)))
    else:
        results.append(
            CheckResult(
                name="live release dir (optional)",
                status="WARN",
                detail="Not present yet (expected before secret test release).",
            )
        )

    config_path = repo_root / "competition/config.yaml"
    if config_path.exists():
        kv = _parse_simple_yaml_kv(config_path)
        default_tokens = kv.get("default_num_tokens", "200000")
        time_limit = kv.get("time_limit_seconds", "600")
        max_context = kv.get("max_context_length", "256")
        results.append(
            CheckResult(
                name="config sanity",
                status="PASS",
                detail=(
                    f"default_num_tokens={default_tokens}, "
                    f"time_limit_seconds={time_limit}, max_context_length={max_context}"
                ),
            )
        )
    else:
        results.append(
            CheckResult(
                name="config sanity",
                status="WARN",
                detail="competition/config.yaml missing; using defaults (200000, 600, 256).",
            )
        )

    if run_tests:
        results.append(_run_pytest(repo_root))
    if check_bundle:
        results.append(_check_bundle_build(repo_root))

    return results


def _print_results(results: list[CheckResult]) -> None:
    print("Live Day Quick Check")
    name_w = max(len("check"), *(len(r.name) for r in results))
    status_w = len("status")
    print(f"{'status'.ljust(status_w)} | {'check'.ljust(name_w)} | detail")
    print(f"{'-'*status_w}-+-{'-'*name_w}-+-{'-'*40}")
    for r in results:
        print(f"{r.status.ljust(status_w)} | {r.name.ljust(name_w)} | {r.detail}")

    fails = sum(r.status == "FAIL" for r in results)
    warns = sum(r.status == "WARN" for r in results)
    print()
    print(f"Summary: PASS={sum(r.status=='PASS' for r in results)} WARN={warns} FAIL={fails}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick readiness checklist for live competition day.")
    parser.add_argument("--run-tests", action="store_true", help="Run `pytest -q` and report result.")
    parser.add_argument(
        "--check-bundle",
        action="store_true",
        help="Attempt a temporary student bundle build and report result.",
    )
    args = parser.parse_args()

    results = run_checks(run_tests=args.run_tests, check_bundle=args.check_bundle)
    _print_results(results)
    has_fail = any(r.status == "FAIL" for r in results)
    raise SystemExit(1 if has_fail else 0)


if __name__ == "__main__":
    main()
