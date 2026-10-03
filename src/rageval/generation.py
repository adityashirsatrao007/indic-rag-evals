"""Answer generation: an extractive baseline plus an optional LLM path.

The default (``extractive``) needs no network and no third-party package: it
scores every sentence of the top-ranked document with the query's BM25 idf
weights and returns the best-scoring sentence(s).

The optional ``llm`` path speaks the OpenAI-compatible chat-completions
protocol, which is what Sarvam's API and most local gateways expose.  It is
guarded: ``requests`` is imported inside the function and missing
configuration raises a clear error instead of a traceback.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING, Callable, Sequence

from .bm25 import tokenize
from .chunking import Document, split_sentences
from .retrieval import MissingDependencyError, ScoredDoc

if TYPE_CHECKING:  # pragma: no cover
    from .bm25 import BM25Index

__all__ = [
    "GENERATORS",
    "extractive_answer",
    "generate_answer",
    "llm_answer",
    "resolve_llm_config",
    "score_sentences",
]

GENERATORS = ("extractive", "llm", "none")

_DEFAULT_BASE_URL = "https://api.sarvam.ai/v1"
_DEFAULT_MODEL = "sarvam-m"


def score_sentences(
    query: str,
    sentences: Sequence[str],
    idf: Callable[[str], float] | None = None,
) -> list[tuple[int, float]]:
    """Score sentences for *query*; returns ``[(index, score)]`` descending.

    A sentence's score is the summed idf of the query tokens it contains,
    divided by the square root of the sentence's distinct-token count, so
    that long, rambling sentences do not win on length alone (distinct
    tokens, not raw counts, keep the denominator about content rather than
    repeated words).  Ties keep the original order (earlier sentence first),
    which makes the output deterministic.
    """
    query_terms = list(dict.fromkeys(tokenize(query)))
    weight = idf or (lambda _term: 1.0)
    scored: list[tuple[int, float]] = []
    for position, sentence in enumerate(sentences):
        tokens = set(tokenize(sentence))
        matched = [term for term in query_terms if term in tokens]
        if not matched:
            scored.append((position, 0.0))
            continue
        raw = sum(weight(term) for term in matched)
        scored.append((position, raw / math.sqrt(max(len(tokens), 1))))
    scored.sort(key=lambda item: (-item[1], item[0]))
    return scored


def extractive_answer(
    query: str,
    documents: Sequence[Document],
    top_docs: Sequence[ScoredDoc],
    *,
    index: "BM25Index | None" = None,
    max_sentences: int = 1,
) -> str:
    """Return the best-scoring sentence(s) from the top-ranked document.

    Falls back to the document's first sentence when nothing in it overlaps
    the query, so the caller always gets *some* extractive answer (this is a
    baseline, not a hallucination guard).
    """
    if not top_docs or not documents:
        return ""
    by_id = {doc.doc_id: doc for doc in documents}
    document = by_id.get(top_docs[0].doc_id)
    if document is None:
        return ""

    sentences = split_sentences(document.text)
    if not sentences:
        return document.text.strip()

    idf = index.idf if index is not None else None
    ranked = score_sentences(query, sentences, idf)
    chosen = [sentences[position] for position, score in ranked if score > 0.0]
    if not chosen:
        chosen = [sentences[0]]
    return " ".join(chosen[: max(1, max_sentences)])


def resolve_llm_config(
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
) -> tuple[str, str, str]:
    """Resolve ``(api_key, base_url, model)`` from arguments and environment.

    An explicit argument always wins.  Of the environment variables the API
    key prefers ``SARVAM_API_KEY`` over ``LLM_API_KEY``, while the base URL
    and model prefer their ``LLM_*`` name over ``SARVAM_API_BASE``/
    ``SARVAM_MODEL``; the base URL and model then fall back to documented
    defaults.  Raises ``ValueError`` naming every variable when the API key
    is missing.
    """
    resolved_key = (
        api_key
        or os.environ.get("SARVAM_API_KEY")
        or os.environ.get("LLM_API_KEY")
        or ""
    )
    if not resolved_key:
        raise ValueError(
            "no API key found: set SARVAM_API_KEY (or LLM_API_KEY) in your "
            "environment or .env, see .env.example"
        )
    resolved_base = (
        base_url
        or os.environ.get("LLM_BASE_URL")
        or os.environ.get("SARVAM_API_BASE")
        or _DEFAULT_BASE_URL
    )
    resolved_model = (
        model or os.environ.get("LLM_MODEL") or os.environ.get("SARVAM_MODEL") or _DEFAULT_MODEL
    )
    return resolved_key, resolved_base.rstrip("/"), resolved_model


def llm_answer(
    query: str,
    context: str,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    timeout: float = 60.0,
) -> str:
    """Call an OpenAI-compatible chat endpoint and return the reply text.

    ``requests`` is imported lazily so the core harness never needs it; if it
    is missing, :class:`MissingDependencyError` explains how to install it.

    The prompt is explicit about grounding: answer only from *context*, reply
    in the language of the question.  That keeps the comparison against the
    extractive baseline fair and auditable.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - depends on environment
        raise MissingDependencyError("requests", "LLM generation") from exc

    key, base, model_name = resolve_llm_config(
        api_key=api_key, base_url=base_url, model=model
    )
    url = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    payload = {
        "model": model_name,
        "temperature": temperature,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You answer questions using ONLY the supplied context. "
                    "If the context does not contain the answer, say so. "
                    "Reply in the language of the question (Hindi, Marathi "
                    "or the code-mixed form used)."
                ),
            },
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {query}"},
        ],
    }
    response = requests.post(
        url,
        json=payload,
        headers={"Authorization": f"Bearer {key}"},
        timeout=timeout,
    )
    if response.status_code >= 400:
        raise RuntimeError(
            f"LLM request failed: HTTP {response.status_code} from {url}: "
            f"{response.text[:300]}"
        )
    data = response.json()
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError(f"unexpected LLM response shape: {data}") from exc
    return str(content).strip()


def generate_answer(
    mode: str,
    query: str,
    documents: Sequence[Document],
    top_docs: Sequence[ScoredDoc],
    *,
    index: "BM25Index | None" = None,
    max_sentences: int = 1,
    context_chars: int = 1500,
    **llm_kwargs: object,
) -> str:
    """Dispatch to the requested generator.

    ``mode='none'`` returns an empty string (retrieval-only evaluation),
    ``mode='extractive'`` is the offline default, ``mode='llm'`` needs
    ``requests`` plus ``SARVAM_API_KEY``.
    """
    if mode == "none":
        return ""
    if mode == "extractive":
        return extractive_answer(
            query, documents, top_docs, index=index, max_sentences=max_sentences
        )
    if mode == "llm":
        context = _build_context(documents, top_docs, limit=context_chars)
        return llm_answer(query, context, **llm_kwargs)  # type: ignore[arg-type]
    raise ValueError(
        f"unknown generator {mode!r} (choose from {', '.join(GENERATORS)})"
    )


def _build_context(
    documents: Sequence[Document],
    top_docs: Sequence[ScoredDoc],
    *,
    limit: int = 1500,
) -> str:
    """Join the top documents' text into one context string under *limit*.

    The budget is soft on purpose: the first hit is always taken whole (a
    truncated first document is worse than slightly overshooting), then
    further documents are added only while they fit.
    """
    by_id = {doc.doc_id: doc for doc in documents}
    parts: list[str] = []
    used = 0
    for hit in top_docs:
        document = by_id.get(hit.doc_id)
        if document is None:
            continue
        block = f"[{document.doc_id}] {document.text}"
        if used + len(block) > limit and parts:
            break
        parts.append(block)
        used += len(block)
    return "\n\n".join(parts)
