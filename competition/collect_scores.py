"""Instructor utility to parse and rank submitted FINAL_SCORE lines."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import re
import sys
from pathlib import Path


_SCORE_RE = re.compile(
    r"^(?:TEAM=(?P<team>\S+)\s+)?FINAL_SCORE\s+"
    r"bits_per_symbol=(?P<bits_per_symbol>[^\s]+)\s+"
    r"elapsed_seconds=(?P<elapsed_seconds>[^\s]+)\s+"
    r"timed_out=(?P<timed_out>True|False)"
    r"(?:\s+evaluated_tokens=(?P<evaluated_tokens>\d+))?\s*$"
)


@dataclass(frozen=True)
class ScoreEntry:
    """Parsed competition score entry."""

    team: str
    bits_per_symbol: float
    elapsed_seconds: float
    timed_out: bool
    evaluated_tokens: int | None
    source_line: str


@dataclass(frozen=True)
class ParseResult:
    """Result of parsing a score text block."""

    entries: list[ScoreEntry]
    ignored_lines: int


@dataclass(frozen=True)
class ClassifiedScores:
    """Entries split into ranked, disqualified, and invalid groups."""

    ranked: list[ScoreEntry]
    disqualified: list[ScoreEntry]
    invalid: list[ScoreEntry]
    ignored_lines: int


def parse_score_lines(text: str) -> ParseResult:
    """Parse score entries from text, ignoring non-matching lines."""

    entries: list[ScoreEntry] = []
    ignored_lines = 0
    unnamed_count = 0

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            ignored_lines += 1
            continue

        match = _SCORE_RE.match(line)
        if match is None:
            ignored_lines += 1
            continue

        team = match.group("team")
        if team is None:
            unnamed_count += 1
            team = f"Team{unnamed_count}"

        entries.append(
            ScoreEntry(
                team=team,
                bits_per_symbol=float(match.group("bits_per_symbol")),
                elapsed_seconds=float(match.group("elapsed_seconds")),
                timed_out=(match.group("timed_out") == "True"),
                evaluated_tokens=(
                    int(match.group("evaluated_tokens"))
                    if match.group("evaluated_tokens") is not None
                    else None
                ),
                source_line=line,
            )
        )

    return ParseResult(entries=entries, ignored_lines=ignored_lines)


def rank_entries(entries: list[ScoreEntry]) -> list[ScoreEntry]:
    """Sort score entries by competition rules."""

    return sorted(
        entries,
        key=lambda e: (
            e.bits_per_symbol,
            e.elapsed_seconds,
        ),
    )


def classify_entries(
    entries: list[ScoreEntry],
    *,
    disqualify_timeouts: bool = True,
    required_tokens: int = 200_000,
) -> ClassifiedScores:
    """Classify parsed entries for official ranking output."""

    if required_tokens <= 0:
        raise ValueError("required_tokens must be positive.")

    ranked: list[ScoreEntry] = []
    disqualified: list[ScoreEntry] = []
    invalid: list[ScoreEntry] = []

    for entry in entries:
        if entry.evaluated_tokens != required_tokens:
            invalid.append(entry)
            continue
        if disqualify_timeouts and entry.timed_out:
            disqualified.append(entry)
            continue
        ranked.append(entry)

    return ClassifiedScores(
        ranked=rank_entries(ranked),
        disqualified=rank_entries(disqualified),
        invalid=rank_entries(invalid),
        ignored_lines=0,
    )


def _read_input(scores_path: str | None) -> str:
    if scores_path is None:
        return sys.stdin.read()
    path = Path(scores_path)
    return path.read_text(encoding="utf-8")


def _format_rows(entries: list[ScoreEntry]) -> tuple[list[str], list[list[str]], list[int]]:
    headers = [
        "rank",
        "team",
        "bits_per_symbol",
        "elapsed_seconds",
        "timed_out",
        "evaluated_tokens",
    ]
    rows = [
        [
            str(idx),
            entry.team,
            f"{entry.bits_per_symbol:.10f}",
            f"{entry.elapsed_seconds:.6f}",
            str(entry.timed_out),
            str(entry.evaluated_tokens) if entry.evaluated_tokens is not None else "MISSING",
        ]
        for idx, entry in enumerate(entries, start=1)
    ]

    widths = [len(h) for h in headers]
    for row in rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))
    return headers, rows, widths


def _print_section(title: str, entries: list[ScoreEntry], *, empty_message: str) -> None:
    print(title)
    if not entries:
        print(f"  {empty_message}")
        return

    headers, rows, widths = _format_rows(entries)

    def fmt_row(cells: list[str]) -> str:
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(cells))

    print(fmt_row(headers))
    print("-+-".join("-" * w for w in widths))
    for row in rows:
        print(fmt_row(row))


def _print_report(
    classified: ClassifiedScores,
    *,
    parsed_entries_count: int,
    ignored_lines: int,
) -> None:
    _print_section(
        "Score Rankings",
        classified.ranked,
        empty_message="No valid ranked entries found.",
    )
    print()
    _print_section(
        "Disqualified (timed out)",
        classified.disqualified,
        empty_message="None.",
    )
    print()
    _print_section(
        "Invalid (wrong evaluated_tokens)",
        classified.invalid,
        empty_message="None.",
    )
    if classified.invalid:
        print("  Note: expected evaluated_tokens to match the required prefix length.")
    print()
    print(f"parsed_entries: {parsed_entries_count}")
    print(f"ignored_lines: {ignored_lines}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse and rank FINAL_SCORE lines.")
    parser.add_argument("--scores-path", type=str, default=None)
    parser.add_argument(
        "--disqualify-timeouts",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude timed_out=True entries from ranked results (default: True).",
    )
    parser.add_argument(
        "--required-tokens",
        type=int,
        default=200_000,
        help="Required evaluated_tokens for valid ranking (default: 200000).",
    )
    args = parser.parse_args()

    text = _read_input(args.scores_path)
    parsed = parse_score_lines(text)
    classified = classify_entries(
        parsed.entries,
        disqualify_timeouts=args.disqualify_timeouts,
        required_tokens=args.required_tokens,
    )
    _print_report(
        ClassifiedScores(
            ranked=classified.ranked,
            disqualified=classified.disqualified,
            invalid=classified.invalid,
            ignored_lines=parsed.ignored_lines,
        ),
        parsed_entries_count=len(parsed.entries),
        ignored_lines=parsed.ignored_lines,
    )


if __name__ == "__main__":
    main()
