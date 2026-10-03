"""Devanagari-safe sentence splitting and overlapping chunking.

Hindi and Marathi text is written in Devanagari, which uses the danda
(``।``, U+0964) as its full stop and does not put spaces around it in the
same way English does around ``.``.  Naive ``text.split(". ")`` therefore
breaks Indic text badly.  This module splits on whitespace runs that follow
a sentence terminator, keeps digits such as ``10,372`` or ``2024.5`` intact,
and produces overlapping chunks suitable for a retrieval index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable

__all__ = [
    "Chunk",
    "chunk_document",
    "chunk_text",
    "iter_chunks",
    "load_documents",
    "split_paragraphs",
    "split_sentences",
    "Document",
]

# Sentence terminators: danda, double danda, ASCII/full stop, !, ?, ellipsis.
# A full stop only terminates a sentence when it is not surrounded by digits
# (so "2024.5" and "10.5" stay whole) and when the next character starts a
# new clause (whitespace, closing quote/bracket, or end of text).
_TERMINATOR = re.compile(
    r"(?:[।॥!?…]+|(?<!\d)\.(?!\d))(?=[\s\"'”’)\]]|$)"
)
_CLOSERS = "\"'”’)]»"
_PARAGRAPH = re.compile(r"\n\s*\n+")


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit of text carved out of a document."""

    doc_id: str
    chunk_id: str
    text: str
    index: int
    sentences: tuple[str, ...]


@dataclass(frozen=True)
class Document:
    """A corpus file with its YAML-ish header parsed out."""

    doc_id: str
    title: str
    lang: str
    text: str
    path: str = ""


def split_paragraphs(text: str) -> list[str]:
    """Return non-empty paragraphs (blank-line separated), whitespace-trimmed."""
    return [p.strip() for p in _PARAGRAPH.split(text.strip()) if p.strip()]


def split_sentences(text: str) -> list[str]:
    """Split *text* into sentences, Devanagari danda aware.

    Terminators are kept at the end of the sentence they close, trailing
    closing quotes/brackets are pulled in with them, and whitespace runs are
    normalised away:

    >>> split_sentences("यूपीआई 2016 में शुरू हुआ। यह एनपीसीआई का प्रणाली है।")
    ['यूपीआई 2016 में शुरू हुआ।', 'यह एनपीसीआई का प्रणाली है।']
    """
    sentences: list[str] = []
    for para in split_paragraphs(text):
        start = 0
        for match in _TERMINATOR.finditer(para):
            end = match.end()
            while end < len(para) and para[end] in _CLOSERS:
                end += 1
            candidate = re.sub(r"\s+", " ", para[start:end]).strip()
            if candidate:
                sentences.append(candidate)
            start = end
        tail = re.sub(r"\s+", " ", para[start:]).strip()
        if tail:
            sentences.append(tail)
    return sentences


def chunk_text(
    text: str,
    *,
    max_chars: int = 480,
    overlap_chars: int = 120,
    doc_id: str = "doc",
) -> list[Chunk]:
    """Split *text* into sentence-aligned chunks of at most *max_chars*.

    Consecutive chunks share trailing sentences worth roughly *overlap_chars*
    characters so a fact that straddles a boundary is still retrievable.  A
    sentence longer than *max_chars* becomes a chunk of its own (never
    cut mid-word), and the routine always advances so it cannot loop.
    """
    if max_chars <= 0:
        raise ValueError("max_chars must be positive")
    if overlap_chars < 0:
        raise ValueError("overlap_chars must be >= 0")
    if overlap_chars >= max_chars:
        raise ValueError("overlap_chars must be smaller than max_chars")

    sentences = split_sentences(text)
    if not sentences:
        return []

    chunks: list[Chunk] = []
    start = 0
    total = len(sentences)
    while start < total:
        length = 0
        end = start
        while end < total:
            added = len(sentences[end]) + (1 if end > start else 0)
            if length + added > max_chars and end > start:
                break
            length += added
            end += 1

        body = tuple(sentences[start:end])
        chunks.append(
            Chunk(
                doc_id=doc_id,
                chunk_id=f"{doc_id}#{len(chunks)}",
                text=" ".join(body),
                index=len(chunks),
                sentences=body,
            )
        )

        if end >= total:
            break

        # Carry up to overlap_chars worth of trailing sentences into the next
        # chunk, always leaving at least one new sentence for it.
        overlap = 0
        carry_start = end
        while carry_start > start + 1:
            candidate = len(sentences[carry_start - 1]) + 1
            if overlap + candidate > overlap_chars:
                break
            overlap += candidate
            carry_start -= 1
        # The loop stops at start + 1 at the lowest, so carry_start is always
        # > start and every iteration strictly advances (no infinite loop).
        start = carry_start

    return chunks


def chunk_document(
    document: Document, *, max_chars: int = 480, overlap_chars: int = 120
) -> list[Chunk]:
    """Chunk a :class:`Document`, prefixing chunk ids with its ``doc_id``."""
    return chunk_text(
        document.text,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
        doc_id=document.doc_id,
    )


def _parse_header(raw: str, fallback_id: str) -> tuple[str, str, str, str]:
    """Parse the leading ``---`` / ``key: value`` / ``---`` block.

    Returns ``(doc_id, title, lang, body)``.  Any malformed or missing field
    falls back to the file name / empty string rather than raising, so a
    slightly untidy corpus file still loads.
    """
    doc_id, title, lang = fallback_id, fallback_id, ""
    body = raw
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return doc_id, title, lang, body

    header: dict[str, str] = {}
    end = None
    for i in range(1, len(lines)):
        if lines[i].strip() == "---":
            end = i
            break
        if ":" in lines[i]:
            key, _, value = lines[i].partition(":")
            header[key.strip().lower()] = value.strip().strip("\"'")
    if end is None:  # unterminated header block -> treat whole file as body
        return doc_id, title, lang, body

    doc_id = header.get("id") or fallback_id
    title = header.get("title") or doc_id
    lang = header.get("lang") or ""
    body = "\n".join(lines[end + 1 :]).strip()
    return doc_id, title, lang, body


def load_documents(path: str) -> list[Document]:
    """Load one file or every ``*.txt``/``*.md`` file under directory *path*.

    Files are sorted by name for deterministic indexing order.
    """
    from pathlib import Path

    root = Path(path)
    if root.is_file():
        files = [root]
    elif root.is_dir():
        files = sorted(
            p for p in list(root.glob("*.txt")) + list(root.glob("*.md")) if p.is_file()
        )
    else:
        raise FileNotFoundError(f"no such document path: {path}")

    documents: list[Document] = []
    seen: set[str] = set()
    for file in files:
        raw = file.read_text(encoding="utf-8")
        doc_id, title, lang, body = _parse_header(raw, file.stem)
        if doc_id in seen:
            raise ValueError(f"duplicate document id {doc_id!r} in {file}")
        seen.add(doc_id)
        documents.append(
            Document(doc_id=doc_id, title=title, lang=lang, text=body, path=str(file))
        )
    if not documents:
        raise FileNotFoundError(f"no .txt/.md documents found in {path}")
    return documents


def iter_chunks(
    documents: Iterable[Document],
    *,
    max_chars: int = 480,
    overlap_chars: int = 120,
) -> list[Chunk]:
    """Chunk every document in *documents* in order."""
    chunks: list[Chunk] = []
    for document in documents:
        chunks.extend(
            chunk_document(document, max_chars=max_chars, overlap_chars=overlap_chars)
        )
    return chunks
