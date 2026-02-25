"""Lightweight tests for competition score collection/parsing."""

from __future__ import annotations

from competition.collect_scores import parse_score_lines, rank_entries


def test_collect_scores_parsing_and_sorting() -> None:
    text = """
    junk line that should be ignored
    TEAM=alice FINAL_SCORE bits_per_symbol=2.9500000000 elapsed_seconds=120.500000 timed_out=False evaluated_tokens=200000
    another ignored line
    FINAL_SCORE bits_per_symbol=2.9500000000 elapsed_seconds=110.000000 timed_out=False evaluated_tokens=200000
    FINAL_SCORE bits_per_symbol=3.1000000000 elapsed_seconds=90.000000 timed_out=False evaluated_tokens=200000
    """

    parsed = parse_score_lines(text)
    assert len(parsed.entries) == 3
    assert parsed.ignored_lines >= 2

    ranked = rank_entries(parsed.entries)
    assert len(ranked) == 3

    # Same bps + non-timeout for first two; lower elapsed time should rank first.
    assert ranked[0].team == "Team1"
    assert ranked[0].bits_per_symbol == 2.95
    assert ranked[1].team == "alice"
    assert ranked[1].bits_per_symbol == 2.95

    # Worse bps ranks later.
    assert ranked[2].bits_per_symbol == 3.1

