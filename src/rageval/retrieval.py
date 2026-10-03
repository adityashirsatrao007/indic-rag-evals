"""Retrieval strategies: BM25, character n-gram, and hybrid score fusion.

The default path is pure Python.  Dense (embedding) retrieval is optional and
is only imported when a strategy actually needs it, so
``pip install sentence-transformers`` is never required to run the harness.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from enum import Enum
from typing import Iterable, Mapping, Sequence

from .bm25 import (
    DEFAULT_GRAM_N,
    BM25Index,
    index_char_grams,
    index_word_tokens,
    tokenize,
    tokenize_char_grams,
)
from .chunking import Chunk, Document, iter_chunks

__all__ = [
    "DenseEncoder",
    "MissingDependencyError",
    "Query",
    "Retriever",
    "ScoredDoc",
    "Strategy",
    "cosine",
    "fuse_rrf",
    "fuse_weighted",
    "iter_document_languages",
    "load_qrels",
]


class MissingDependencyError(ImportError):
    """Raised when an optional dependency is used without being installed."""

    def __init__(self, package: str, purpose: str) -> None:
        super().__init__(
            f"{purpose} requires the optional dependency '{package}'. "
            f"Install it with: pip install {package} "
            f"(the core harness runs without it)"
        )
        self.package = package
        self.purpose = purpose


class Strategy(str, Enum):
    """Selectable retrieval strategies, as named in the report."""

    BM25 = "bm25"
    CHAR_GRAM = "char-gram"
    HYBRID = "hybrid"
    HYBRID_DENSE = "hybrid+dense"

    @classmethod
    def parse(cls, value: "str | Strategy") -> "Strategy":
        """Parse a user-supplied strategy name with a helpful error."""
        if isinstance(value, Strategy):
            return value
        try:
            return cls(str(value).strip().lower())
        except ValueError as exc:
            valid = ", ".join(s.value for s in cls)
            raise ValueError(
                f"unknown strategy {value!r} (choose from {valid})"
            ) from exc

    @property
    def components(self) -> tuple[str, ...]:
        """Names of the score components this strategy fuses."""
        if self is Strategy.BM25:
            return ("bm25",)
        if self is Strategy.CHAR_GRAM:
            return ("char-gram",)
        if self is Strategy.HYBRID:
            return ("bm25", "char-gram")
        return ("bm25", "char-gram", "dense")

    @property
    def uses_dense(self) -> bool:
        return self is Strategy.HYBRID_DENSE


@dataclass(frozen=True)
class ScoredDoc:
    """A document in a ranked result list (1-based ``rank``)."""

    doc_id: str
    score: float
    rank: int
    chunk_id: str | None = None
    title: str = ""


@dataclass(frozen=True)
class Query:
    """One labelled evaluation item from ``qa.jsonl``."""

    query_id: str
    question: str
    relevant_doc_ids: tuple[str, ...]
    answer: str


def load_qrels(path: str) -> list[Query]:
    """Load a JSONL eval file with id/question/relevant_doc_ids/answer."""
    import json

    queries: list[Query] = []
    with open(path, encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            for key in ("id", "question", "relevant_doc_ids", "answer"):
                if key not in row:
                    raise ValueError(f"{path}:{line_no}: missing key {key!r}")
            queries.append(
                Query(
                    query_id=str(row["id"]),
                    question=str(row["question"]),
                    relevant_doc_ids=tuple(str(x) for x in row["relevant_doc_ids"]),
                    answer=str(row["answer"]),
                )
            )
    if not queries:
        raise ValueError(f"{path}: no queries found")
    return queries


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    """Cosine similarity of two equal-length vectors (0.0 if either is empty)."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


# --------------------------------------------------------------------------
# score fusion
# --------------------------------------------------------------------------
def _check_weights(
    score_maps: Sequence[Mapping[str, float]], weights: Sequence[float] | None
) -> list[float]:
    if not score_maps:
        raise ValueError("nothing to fuse")
    resolved = list(weights) if weights is not None else [1.0] * len(score_maps)
    if len(resolved) != len(score_maps):
        raise ValueError("weights must match the number of score maps")
    return resolved


def fuse_rrf(
    score_maps: Sequence[Mapping[str, float]],
    *,
    weights: Sequence[float] | None = None,
    constant: float = 60.0,
) -> dict[str, float]:
    """Reciprocal-Rank Fusion of several ``doc_id -> score`` maps.

    Only documents with a strictly positive score in a component take part in
    that component's ranking, so a component that found nothing cannot promote
    documents it never matched.

    ``constant`` is the RRF *k*: it flattens the rank axis, so being #1 rather
    than #2 in one list is worth only a little -- one noisy ranker cannot
    outvote the others on its own.  60 is the value Cormack et al. (SIGIR 2009)
    tuned and that has been the default ever since.
    """
    resolved = _check_weights(score_maps, weights)
    fused: dict[str, float] = {}
    for weight, scores in zip(resolved, score_maps):
        ranked = sorted(
            ((doc_id, value) for doc_id, value in scores.items() if value > 0.0),
            key=lambda kv: (-kv[1], kv[0]),
        )
        for position, (doc_id, _) in enumerate(ranked, start=1):
            fused[doc_id] = fused.get(doc_id, 0.0) + weight / (constant + position)
    return fused


def fuse_weighted(
    score_maps: Sequence[Mapping[str, float]],
    *,
    weights: Sequence[float] | None = None,
) -> dict[str, float]:
    """Per-component max scaling, then a weighted sum over score maps.

    Each component is scaled by its own maximum score, which puts BM25
    (unbounded) and cosine similarity (``[-1, 1]``) on one scale.
    """
    resolved = _check_weights(score_maps, weights)
    fused: dict[str, float] = {}
    for weight, scores in zip(resolved, score_maps):
        if not scores:
            continue
        peak = max(scores.values())
        if peak <= 0.0:
            continue
        for doc_id, value in scores.items():
            if value <= 0.0:
                continue
            fused[doc_id] = fused.get(doc_id, 0.0) + weight * (value / peak)
    return fused


# --------------------------------------------------------------------------
# optional dense encoder
# --------------------------------------------------------------------------
class DenseEncoder:
    """Thin wrapper around sentence-transformers, imported lazily.

    Raises :class:`MissingDependencyError` with an install hint when the
    package is absent, so callers fail with an actionable message.
    """

    def __init__(self, model_name: str | None = None) -> None:
        name = model_name or os.environ.get(
            "RAGEVAL_DENSE_MODEL", "paraphrase-multilingual-MiniLM-L12-v2"
        )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise MissingDependencyError(
                "sentence-transformers", "dense retrieval"
            ) from exc
        self.model_name = name
        self._model = SentenceTransformer(name)

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed *texts* as plain Python float lists."""
        vectors = self._model.encode(list(texts))
        return [[float(x) for x in row] for row in vectors]


# --------------------------------------------------------------------------
# retriever
# --------------------------------------------------------------------------
class Retriever:
    """Chunk-level lexical indexes rolled up to document-level rankings.

    Documents are split into overlapping chunks once, then indexed twice: a
    word-token BM25 index and a character 3-gram BM25 index.  Each component
    produces ``doc_id -> score`` by taking the **max** chunk score of that
    document, and the selected strategy decides how the component maps are
    combined (weighted sum or RRF for multi-component strategies).
    """

    def __init__(
        self,
        documents: Sequence[Document],
        *,
        strategy: "str | Strategy" = Strategy.BM25,
        fusion: str = "rrf",
        gram_n: int = DEFAULT_GRAM_N,
        max_chars: int = 480,
        overlap_chars: int = 120,
        weights: Sequence[float] | None = None,
        dense_encoder: "DenseEncoder | None" = None,
    ) -> None:
        if fusion not in ("rrf", "weighted"):
            raise ValueError("fusion must be 'rrf' or 'weighted'")
        if not documents:
            raise ValueError("cannot build a retriever over an empty corpus")

        self.documents = {doc.doc_id: doc for doc in documents}
        self.strategy = Strategy.parse(strategy)
        self.fusion = fusion
        self.gram_n = gram_n
        self.weights = list(weights) if weights is not None else None

        self.chunks: list[Chunk] = iter_chunks(
            documents, max_chars=max_chars, overlap_chars=overlap_chars
        )
        chunk_texts = {chunk.chunk_id: chunk.text for chunk in self.chunks}
        self._chunk_to_doc = {chunk.chunk_id: chunk.doc_id for chunk in self.chunks}
        self._doc_chunks: dict[str, list[str]] = {}
        for chunk in self.chunks:
            self._doc_chunks.setdefault(chunk.doc_id, []).append(chunk.chunk_id)

        self.word_index: BM25Index = index_word_tokens(chunk_texts)
        self.gram_index: BM25Index = index_char_grams(chunk_texts, n=gram_n)

        self._dense = dense_encoder
        self._dense_vectors: list[list[float]] | None = None
        if self._dense is not None:
            # encode in self.chunks order: _component() zips this list with
            # self.chunks, so the two must line up index for index.
            self._dense_vectors = self._dense.encode([c.text for c in self.chunks])

    # -- component scores ---------------------------------------------
    @staticmethod
    def _rollup(
        chunk_scores: Mapping[str, float], chunk_to_doc: Mapping[str, str]
    ) -> tuple[dict[str, float], dict[str, str]]:
        """Max-pool chunk scores into ``doc_id -> score`` plus best chunk id."""
        docs: dict[str, float] = {}
        best: dict[str, str] = {}
        for chunk_id, score in chunk_scores.items():
            doc_id = chunk_to_doc[chunk_id]
            if score > docs.get(doc_id, float("-inf")):
                docs[doc_id] = score
                best[doc_id] = chunk_id
        return docs, best

    def _component(self, name: str, query: str) -> tuple[dict[str, float], dict[str, str]]:
        """Score + best-chunk maps for a single named component."""
        if name == "bm25":
            return self._rollup(self.word_index.score(tokenize(query)), self._chunk_to_doc)
        if name == "char-gram":
            grams = tokenize_char_grams(query, self.gram_n)
            return self._rollup(self.gram_index.score(grams), self._chunk_to_doc)
        if name == "dense":
            if self._dense is None or self._dense_vectors is None:
                raise MissingDependencyError(
                    "sentence-transformers", "dense retrieval"
                )
            query_vec = self._dense.encode([query])[0]
            chunk_scores = {
                chunk.chunk_id: cosine(query_vec, vector)
                for chunk, vector in zip(self.chunks, self._dense_vectors)
            }
            return self._rollup(chunk_scores, self._chunk_to_doc)
        raise ValueError(f"unknown component {name!r}")  # pragma: no cover

    def _combine(self, maps: Sequence[Mapping[str, float]]) -> dict[str, float]:
        if len(maps) == 1:
            return dict(maps[0])
        if self.fusion == "rrf":
            return fuse_rrf(maps, weights=self.weights)
        return fuse_weighted(maps, weights=self.weights)

    # -- public API ----------------------------------------------------
    def component_scores(self, query: str) -> dict[str, dict[str, float]]:
        """``component -> (doc_id -> score)`` for the selected strategy."""
        return {
            name: self._component(name, query)[0]
            for name in self.strategy.components
        }

    def search(self, query: str, top_k: int = 5) -> list[ScoredDoc]:
        """Rank the corpus for *query* and return the top *top_k* documents.

        Zero-score documents are omitted: a strategy that found no evidence
        reports "no match" instead of inventing an ordering.
        """
        if top_k <= 0:
            return []

        maps: list[dict[str, float]] = []
        best: dict[str, str] = {}
        for name in self.strategy.components:
            scores, best_chunks = self._component(name, query)
            maps.append(scores)
            # _rollup returns the same keys in both maps; the first component
            # of the strategy owns the snippet that gets shown for a document.
            for doc_id, chunk_id in best_chunks.items():
                best.setdefault(doc_id, chunk_id)

        fused = self._combine(maps)
        ranked = sorted(
            ((doc_id, value) for doc_id, value in fused.items() if value > 0.0),
            key=lambda kv: (-kv[1], kv[0]),
        )
        results: list[ScoredDoc] = []
        for position, (doc_id, value) in enumerate(ranked[:top_k], start=1):
            document = self.documents.get(doc_id)
            results.append(
                ScoredDoc(
                    doc_id=doc_id,
                    score=value,
                    rank=position,
                    chunk_id=best.get(doc_id),
                    title=document.title if document else "",
                )
            )
        return results

    def explain(self, query: str, limit: int = 5) -> dict[str, list[tuple[str, float]]]:
        """Per-component top hits, useful for debugging a fusion decision."""
        return {
            name: sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:limit]
            for name, scores in self.component_scores(query).items()
        }

    def chunk_text(self, chunk_id: str | None) -> str:
        """Raw text of a chunk id (or ``""``), for showing snippets."""
        if not chunk_id:
            return ""
        for chunk in self.chunks:
            if chunk.chunk_id == chunk_id:
                return chunk.text
        return ""


def iter_document_languages(documents: Iterable[Document]) -> list[str]:
    """Sorted unique base language tags found in the corpus headers."""
    langs = set()
    for document in documents:
        if document.lang:
            langs.add(document.lang.split("-")[0].split("_")[0].lower())
    return sorted(langs)
