"""Count-threshold suffix-backoff n-gram predictor.

This predictor fits the same sparse counts as `NGramPredictor`, but during
prediction it only uses a context if that context has been seen at least
`min_count` times. Otherwise it backs off by dropping the oldest symbol
(suffix backoff) until a context meeting the threshold is found, ultimately
falling back to the empty context `()`.
"""

from __future__ import annotations

from src.predictors.ngram import NGramPredictor, _SparseNextSymbolCounts


class CountThresholdNGramPredictor(NGramPredictor):
    """Laplace-smoothed n-gram predictor with count-thresholded backoff.

    Parameters:
        alphabet_size: Number of symbols in the alphabet.
        n: Maximum n-gram order (conditions on up to `n-1` previous symbols).
        laplace: Additive smoothing constant (>0).
        min_count: Minimum context count required before using that context.
            If a seen context has total count < `min_count`, the model backs off
            to a shorter suffix context.
    """

    def __init__(
        self,
        alphabet_size: int,
        *,
        n: int = 5,
        laplace: float = 1.0,
        min_count: int = 5,
        max_context_length: int | None = None,
        adapt_online: bool = True,
        debug: bool = False,
        debug_max_messages: int = 20,
    ) -> None:
        if min_count < 1:
            raise ValueError("min_count must be at least 1.")
        self.min_count = int(min_count)
        super().__init__(
            alphabet_size=alphabet_size,
            n=n,
            laplace=laplace,
            max_context_length=max_context_length,
            adapt_online=adapt_online,
            debug=debug,
            debug_max_messages=debug_max_messages,
        )

    def _lookup_backoff_stats(
        self, context_tail: tuple[int, ...]
    ) -> tuple[int, _SparseNextSymbolCounts | None]:
        max_order = min(len(context_tail), self.n - 1)
        for order in range(max_order, -1, -1):
            key = context_tail[-order:] if order > 0 else ()
            stats = self._counts_by_order[order].get(key)
            if stats is not None and stats.total >= self.min_count:
                return order, stats
        return 0, None

