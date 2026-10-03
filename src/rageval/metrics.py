"""Retrieval and answer metrics.

Retrieval: Recall@k (k = 1, 3, 5) and MRR.
Answer quality: token-F1 and exact match against the reference answer.

Text normalisation is Devanagari aware: the danda (``।``), double danda,
Latin punctuation and case differences are all stripped before comparison,
so ``"क्यूआर कोड स्कैन करके भुगतान।"`` and ``"क्यूआर कोड स्कैन करके भुगतान"``
score as an exact match.
"""

from __future__ import annotations

import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .bm25 import is_word_char

__all__ = [
    "QueryResult",
    "StrategyMetrics",
    "exact_match",
    "mean",
    "mrr",
    "normalize",
    "recall_at_k",
    "reciprocal_rank",
    "token_f1",
]


def normalize(text: str) -> str:
    """NFC-fold, lowercase, drop punctuation, collapse whitespace (unicode-safe).

    Uses the same word-character definition as the tokenizer (letters, digits
    and combining marks from any script), so Devanagari matras survive while
    the danda, quotes, dashes and commas do not:

    >>> normalize("  यूपीआई,  कैसे काम करता है। ")
    'यूपीआई कैसे काम करता है'
    >>> normalize("10,372 करोड़")
    '10 372 करोड़'
    """
    parts: list[str] = []
    buffer: list[str] = []
    # NFC first, exactly like rageval.bm25.normalize: a decomposed ("NFD")
    # prediction -- e.g. text copied from macOS -- must still compare equal to
    # its composed gold form, which is the form retrieval indexed.
    for char in unicodedata.normalize("NFC", text).lower():
        if is_word_char(char):
            buffer.append(char)
        elif buffer:
            parts.append("".join(buffer))
            buffer = []
    if buffer:
        parts.append("".join(buffer))
    return " ".join(parts)


def _tokens(text: str) -> list[str]:
    normalised = normalize(text)
    return normalised.split() if normalised else []


def token_f1(prediction: str, gold: str) -> float:
    """SQuAD-style bag-of-tokens F1 over normalised text (0.0 - 1.0)."""
    predicted = _tokens(prediction)
    reference = _tokens(gold)
    if not predicted or not reference:
        return 0.0
    common = Counter(predicted) & Counter(reference)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return 2.0 * precision * recall / (precision + recall)


def exact_match(prediction: str, gold: str) -> float:
    """1.0 when normalised prediction equals normalised gold, else 0.0."""
    return 1.0 if normalize(prediction) == normalize(gold) else 0.0


def recall_at_k(
    ranked_doc_ids: Sequence[str], relevant_doc_ids: Iterable[str], k: int
) -> float:
    """Fraction of *relevant_doc_ids* found in the first *k* ranked results."""
    relevant = list(dict.fromkeys(relevant_doc_ids))  # dedupe, keep order
    if not relevant:
        raise ValueError("relevant_doc_ids must not be empty")
    if k <= 0:
        raise ValueError("k must be positive")
    top = list(ranked_doc_ids[:k])
    hits = sum(1 for doc_id in relevant if doc_id in top)
    return hits / len(relevant)


def reciprocal_rank(
    ranked_doc_ids: Sequence[str], relevant_doc_ids: Iterable[str]
) -> float:
    """``1 / rank`` of the first relevant document, 0.0 if none is retrieved."""
    relevant = set(relevant_doc_ids)
    if not relevant:
        raise ValueError("relevant_doc_ids must not be empty")
    for position, doc_id in enumerate(ranked_doc_ids, start=1):
        if doc_id in relevant:
            return 1.0 / position
    return 0.0


def mrr(
    ranked_lists: Sequence[Sequence[str]], relevant_lists: Sequence[Iterable[str]]
) -> float:
    """Mean reciprocal rank over a batch of queries."""
    if len(ranked_lists) != len(relevant_lists):
        raise ValueError("ranked_lists and relevant_lists must align")
    if not ranked_lists:
        return 0.0
    return sum(
        reciprocal_rank(ranked, relevant)
        for ranked, relevant in zip(ranked_lists, relevant_lists)
    ) / len(ranked_lists)


def mean(values: Sequence[float]) -> float:
    """Arithmetic mean, 0.0 for an empty sequence."""
    return sum(values) / len(values) if values else 0.0


@dataclass
class QueryResult:
    """Per-query outcome, kept so the report can show a detail table."""

    query_id: str
    ranked_doc_ids: tuple[str, ...]
    relevant_doc_ids: tuple[str, ...]
    recall: dict[int, float] = field(default_factory=dict)
    rr: float = 0.0
    prediction: str = ""
    token_f1: float = 0.0
    exact: float = 0.0


@dataclass
class StrategyMetrics:
    """Aggregated metrics for one retrieval strategy."""

    strategy: str
    n_queries: int
    recall: dict[int, float]
    mrr: float
    token_f1: float | None
    exact_match: float | None
    per_query: list[QueryResult] = field(default_factory=list)

    def row(self, k_values: Sequence[int] = (1, 3, 5)) -> dict[str, object]:
        """Flat ``metric -> value`` mapping used for CSV/report rows."""
        row: dict[str, object] = {"strategy": self.strategy}
        for k in k_values:
            row[f"recall@{k}"] = round(self.recall.get(k, 0.0), 4)
        row["mrr"] = round(self.mrr, 4)
        row["token_f1"] = (
            round(self.token_f1, 4) if self.token_f1 is not None else ""
        )
        row["exact_match"] = (
            round(self.exact_match, 4) if self.exact_match is not None else ""
        )
        row["n_queries"] = self.n_queries
        return row


def evaluate_rankings(
    strategy: str,
    rankings: Sequence[Sequence[str]],
    qrels: Sequence[Iterable[str]],
    *,
    predictions: Sequence[str] | None = None,
    gold_answers: Sequence[str] | None = None,
    query_ids: Sequence[str] | None = None,
    k_values: Sequence[int] = (1, 3, 5),
) -> StrategyMetrics:
    """Aggregate per-query rankings (and optionally predictions) into metrics.

    ``predictions``/``gold_answers`` are optional but come as a pair: when
    both are omitted the answer columns stay ``None`` and the CSV writes an
    empty cell, which is how the harness reports ``--generator none`` runs.
    """
    if len(rankings) != len(qrels):
        raise ValueError("rankings and qrels must align")
    if query_ids is not None and len(query_ids) != len(rankings):
        raise ValueError("query_ids must align with rankings")
    if (predictions is None) != (gold_answers is None):
        # supplying only one side would silently skip answer scoring
        raise ValueError("predictions and gold_answers must be supplied together")
    ids = list(query_ids) if query_ids else [str(i + 1) for i in range(len(rankings))]

    per_query: list[QueryResult] = []
    for query_id, ranked, relevant in zip(ids, rankings, qrels):
        relevant = tuple(dict.fromkeys(relevant))
        result = QueryResult(
            query_id=query_id,
            ranked_doc_ids=tuple(ranked),
            relevant_doc_ids=relevant,
        )
        for k in k_values:
            result.recall[k] = recall_at_k(ranked, relevant, k)
        result.rr = reciprocal_rank(ranked, relevant)
        per_query.append(result)

    answer_scores = predictions is not None and gold_answers is not None
    if answer_scores:
        preds = predictions or []
        golds = gold_answers or []
        if len(preds) != len(rankings) or len(golds) != len(rankings):
            raise ValueError("predictions, gold_answers and rankings must align")
        for result, prediction, gold in zip(per_query, preds, golds):
            result.prediction = prediction
            result.token_f1 = token_f1(prediction, gold)
            result.exact = exact_match(prediction, gold)

    # Macro-average: every query counts the same (the standard in IR eval),
    # so a query with several gold documents cannot outweigh a single-doc one.
    recall_agg = {
        k: mean([r.recall[k] for r in per_query]) if per_query else 0.0
        for k in k_values
    }
    return StrategyMetrics(
        strategy=strategy,
        n_queries=len(per_query),
        recall=recall_agg,
        mrr=mean([r.rr for r in per_query]),
        token_f1=mean([r.token_f1 for r in per_query]) if answer_scores else None,
        exact_match=mean([r.exact for r in per_query]) if answer_scores else None,
        per_query=per_query,
    )
