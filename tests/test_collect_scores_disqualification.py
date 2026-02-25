"""Tests for timeout disqualification and token-count validation in score collection."""

from __future__ import annotations

from competition.collect_scores import classify_entries, parse_score_lines


def test_collect_scores_disqualification_and_invalid_groups() -> None:
    text = """
    TEAM=ok FINAL_SCORE bits_per_symbol=2.9000000000 elapsed_seconds=100.000000 timed_out=False evaluated_tokens=200000
    TEAM=slow FINAL_SCORE bits_per_symbol=2.8000000000 elapsed_seconds=600.000000 timed_out=True evaluated_tokens=200000
    TEAM=short FINAL_SCORE bits_per_symbol=2.7000000000 elapsed_seconds=50.000000 timed_out=False evaluated_tokens=5000
    TEAM=legacy FINAL_SCORE bits_per_symbol=3.1000000000 elapsed_seconds=90.000000 timed_out=False
    junk
    """

    parsed = parse_score_lines(text)
    classified = classify_entries(
        parsed.entries,
        disqualify_timeouts=True,
        required_tokens=200_000,
    )

    assert [e.team for e in classified.ranked] == ["ok"]
    assert [e.team for e in classified.disqualified] == ["slow"]
    assert [e.team for e in classified.invalid] == ["short", "legacy"]

