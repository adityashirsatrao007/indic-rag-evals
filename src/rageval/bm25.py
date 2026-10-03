"""Pure-Python BM25 (Okapi) over word tokens or character n-grams.

No third-party dependency: postings, idf and length normalisation are all
computed with the standard library.  Two tokenisation strategies are
supported and both flow through the same scorer:

* ``tokenize``            -- whitespace/punctuation aware word tokens that
                             keep Devanagari chunks intact (best precision
                             when query vocabulary matches the corpus).
* ``tokenize_char_grams`` -- padded character 3-grams, robust to typos,
                             missing matras and code-mixed transliteration.

BM25 uses k1 = 1.5 and b = 0.75, the defaults of the reference Okapi
implementations (rank_bm25, gensim); Lucene/Elasticsearch ship k1 = 1.2 with
the same b = 0.75.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter, defaultdict
from typing import Callable, Iterable, Mapping, Sequence

__all__ = [
    "BM25Index",
    "DEFAULT_B",
    "DEFAULT_GRAM_N",
    "DEFAULT_K1",
    "index_char_grams",
    "index_word_tokens",
    "is_word_char",
    "merge_texts",
    "normalize",
    "tokenize",
    "tokenize_char_grams",
]

DEFAULT_K1 = 1.5
DEFAULT_B = 0.75
DEFAULT_GRAM_N = 3


def normalize(text: str) -> str:
    """NFC-normalise and lowercase *text* so query/doc comparisons match."""
    return unicodedata.normalize("NFC", text).lower()


def is_word_char(ch: str) -> bool:
    """True for characters that may appear inside a word.

    Python's ``\\w`` (and ``str.isalnum``) cover letters and digits in every
    script but **not** combining marks, and Devanagari vowel signs/matras such
    as ``ा``/``ी``/``े`` are marks (category Mn).  Treating them as separators
    would shred ``मराठी`` into ``मर`` + ``ठ``; this predicate keeps clusters
    like ``मराठी`` and ``क्यूआर`` whole while still splitting on punctuation.
    """
    return ch.isalnum() or unicodedata.category(ch).startswith("M")


def tokenize(text: str) -> list[str]:
    """Split *text* into normalised word tokens (Devanagari safe).

    Runs of word characters (letters, digits and combining marks from any
    script) form one token; whitespace and punctuation (including the danda
    ``।``) separate tokens.

    >>> tokenize("यूपीआई (UPI) कैसे काम करता है?")
    ['यूपीआई', 'upi', 'कैसे', 'काम', 'करता', 'है']
    >>> tokenize("मराठी भाषा, देवनागरी।")
    ['मराठी', 'भाषा', 'देवनागरी']
    """
    tokens: list[str] = []
    current: list[str] = []
    for char in normalize(text):
        if is_word_char(char):
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens


def tokenize_char_grams(text: str, n: int = DEFAULT_GRAM_N) -> list[str]:
    """Padded character *n*-grams over every word token.

    Tokens are wrapped in ``#`` so prefixes/suffixes stay distinguishable
    (``#up`` != ``up#``).  Short tokens still yield grams thanks to padding.

    >>> tokenize_char_grams("UPI", 3)
    ['#up', 'upi', 'pi#']
    """
    grams: list[str] = []
    for token in tokenize(text):
        padded = f"#{token}#"
        if len(padded) < n:
            grams.append(padded)
            continue
        grams.extend(padded[i : i + n] for i in range(len(padded) - n + 1))
    return grams


class BM25Index:
    """An Okapi BM25 index over pre-tokenised documents.

    Parameters
    ----------
    documents:
        Mapping of ``doc_id -> term sequence``.  Build one index per
        tokenisation strategy (words or char n-grams).
    k1, b:
        Term-saturation and length-normalisation constants.
    """

    def __init__(
        self,
        documents: Mapping[str, Sequence[str]],
        *,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be positive")
        if not 0 <= b <= 1:
            raise ValueError("b must be within [0, 1]")
        self.k1 = float(k1)
        self.b = float(b)
        self._doc_len: dict[str, int] = {}
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)

        for doc_id, terms in documents.items():
            tokens = list(terms)
            self._doc_len[doc_id] = len(tokens)
            for term, count in Counter(tokens).items():
                self._postings[term][doc_id] = count

        self._n = len(self._doc_len)
        self._avgdl = sum(self._doc_len.values()) / self._n if self._n else 0.0
        self._idf_cache: dict[str, float] = {}

    # -- introspection -------------------------------------------------
    @property
    def avgdl(self) -> float:
        """Average document length in tokens."""
        return self._avgdl

    @property
    def size(self) -> int:
        """Number of indexed documents."""
        return self._n

    def doc_frequency(self, term: str) -> int:
        """Number of documents containing *term*."""
        return len(self._postings.get(term, ()))

    def idf(self, term: str) -> float:
        """Robertson/Sparck-Jones idf in Lucene's positive ``log(1 + ...)`` form.

        The textbook ``log((N - df + 0.5) / (df + 0.5))`` turns negative once a
        term occurs in more than half the corpus, which would flip the sign of
        every score containing it; the ``+ 1`` inside the log (what Lucene
        indexes) keeps idf strictly positive, so matching a near-universal
        term can only add to a document's score, never subtract from it.
        """
        cached = self._idf_cache.get(term)
        if cached is not None:
            return cached
        df = self.doc_frequency(term)
        value = math.log(1.0 + (self._n - df + 0.5) / (df + 0.5))
        self._idf_cache[term] = value
        return value

    # -- scoring -------------------------------------------------------
    def score(self, query: Sequence[str]) -> dict[str, float]:
        """Score every document that matches at least one *query* term.

        BM25 is strictly positive here (see :meth:`idf`), so documents that
        match nothing are simply absent from the result rather than present
        with 0.0.  Repeated query terms are summed per occurrence, which
        matches Lucene's behaviour for repeated query terms.
        """
        scores: dict[str, float] = {}
        if self._n == 0:
            return scores
        for term in query:
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc_id, tf in postings.items():
                doc_len = self._doc_len[doc_id]
                denom = tf + self.k1 * (
                    1.0 - self.b + self.b * doc_len / (self._avgdl or 1.0)
                )
                scores[doc_id] = scores.get(doc_id, 0.0) + idf * tf * (
                    self.k1 + 1.0
                ) / denom
        return scores

    def top_k(self, query: Sequence[str], k: int = 5) -> list[tuple[str, float]]:
        """Return the *k* best ``(doc_id, score)`` pairs, ties broken by id."""
        if k <= 0:
            return []
        ranked = sorted(self.score(query).items(), key=lambda kv: (-kv[1], kv[0]))
        return ranked[:k]

    # -- helpers -------------------------------------------------------
    @classmethod
    def from_texts(
        cls,
        texts: Mapping[str, str],
        *,
        tokenizer: Callable[[str], list[str]] = tokenize,
        k1: float = DEFAULT_K1,
        b: float = DEFAULT_B,
    ) -> "BM25Index":
        """Tokenise ``doc_id -> raw text`` with *tokenizer* and index it."""
        return cls(
            {doc_id: tokenizer(text) for doc_id, text in texts.items()},
            k1=k1,
            b=b,
        )

    def zero_scores(self) -> dict[str, float]:
        """A ``doc_id -> 0.0`` map for queries that match nothing."""
        return {doc_id: 0.0 for doc_id in self._doc_len}


def index_word_tokens(texts: Mapping[str, str], **kwargs: float) -> BM25Index:
    """Convenience: word-token BM25 over ``doc_id -> text``."""
    return BM25Index.from_texts(texts, tokenizer=tokenize, **kwargs)


def index_char_grams(
    texts: Mapping[str, str], *, n: int = DEFAULT_GRAM_N, **kwargs: float
) -> BM25Index:
    """Convenience: character *n*-gram BM25 over ``doc_id -> text``."""
    return BM25Index.from_texts(
        texts, tokenizer=lambda text: tokenize_char_grams(text, n), **kwargs
    )


def merge_texts(chunks: Iterable[tuple[str, str]]) -> dict[str, str]:
    """Group ``[(doc_id, chunk_text), ...]`` into ``doc_id -> joined text``."""
    merged: dict[str, list[str]] = defaultdict(list)
    for doc_id, text in chunks:
        merged[doc_id].append(text)
    return {doc_id: " ".join(parts) for doc_id, parts in merged.items()}
