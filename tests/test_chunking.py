"""Tests for Devanagari-safe sentence splitting, chunking and corpus loading."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rageval.chunking import (  # noqa: E402
    chunk_text,
    iter_chunks,
    load_documents,
    split_paragraphs,
    split_sentences,
)

REPO = Path(__file__).resolve().parents[1]
DOCS = REPO / "data" / "documents"


class SplitSentencesTests(unittest.TestCase):
    def test_splits_on_danda(self) -> None:
        text = "यूपीआई 2016 में शुरू हुआ। यह एनपीसीआई की प्रणाली है। इसे भीम ऐप से चलाया जाता है।"
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertTrue(sentences[0].endswith("।"))
        self.assertIn("2016", sentences[0])

    def test_marathi_danda_and_question_mark(self) -> None:
        text = "सोलापूर प्रसिद्ध आहे. कोण प्रसिद्ध आहे? पहिले वाक्य!"
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 3)
        self.assertTrue(sentences[1].endswith("?"))

    def test_decimal_numbers_are_not_split(self) -> None:
        text = "वृद्धि दर 10.5 प्रतिशत रही। अगला वाक्य यह है।"
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertIn("10.5", sentences[0])

    def test_trailing_closing_quote_joins_sentence(self) -> None:
        text = 'उन्होंने कहा "यह सही है।" फिर सभी चले गए।'
        sentences = split_sentences(text)
        self.assertEqual(len(sentences), 2)
        self.assertTrue(sentences[0].endswith('।"'))

    def test_no_terminal_punctuation_returns_single_sentence(self) -> None:
        text = "बिना विराम का पाठ"
        self.assertEqual(split_sentences(text), [text])

    def test_whitespace_is_collapsed(self) -> None:
        text = "पहला   वाक्य ।\n\n\n  दूसरा\tवाक्य ।"
        self.assertEqual(split_sentences(text), ["पहला वाक्य ।", "दूसरा वाक्य ।"])


class ParagraphTests(unittest.TestCase):
    def test_blank_lines_split_paragraphs(self) -> None:
        paras = split_paragraphs("एक\n\nदो\n\n\nतीन")
        self.assertEqual(paras, ["एक", "दो", "तीन"])

    def test_empty_text_yields_nothing(self) -> None:
        self.assertEqual(split_paragraphs("   \n  "), [])


class ChunkingTests(unittest.TestCase):
    LONG = (
        "पहला वाक्य यहाँ है। दूसरा वाक्य यहाँ है। तीसरा वाक्य यहाँ है। "
        "चौथा वाक्य यहाँ है। पाँचवाँ वाक्य यहाँ है। छठा वाक्य यहाँ है। "
        "सातवाँ वाक्य यहाँ है। आठवाँ वाक्य यहाँ है। नौवाँ वाक्य यहाँ है। "
        "दसवाँ वाक्य यहाँ है। ग्यारहवाँ वाक्य यहाँ है। बारहवाँ वाक्य यहाँ है।"
    )

    def test_chunks_respect_size_budget(self) -> None:
        chunks = chunk_text(self.LONG, max_chars=120, overlap_chars=30)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(len(chunk.text), 120)
            self.assertTrue(chunk.text.endswith("।"))
        self.assertEqual([c.index for c in chunks], list(range(len(chunks))))
        self.assertEqual(chunks[0].chunk_id, "doc#0")
        self.assertEqual(chunks[1].chunk_id, "doc#1")

    def test_consecutive_chunks_share_sentences(self) -> None:
        chunks = chunk_text(self.LONG, max_chars=120, overlap_chars=30)
        overlap = set(chunks[0].sentences) & set(chunks[1].sentences)
        self.assertTrue(overlap, "expected overlap between consecutive chunks")

    def test_every_source_sentence_is_retrievable(self) -> None:
        source = set(split_sentences(self.LONG))
        chunks = chunk_text(self.LONG, max_chars=100, overlap_chars=40)
        covered: set[str] = set()
        for chunk in chunks:
            covered.update(chunk.sentences)
        self.assertEqual(source, covered)

    def test_custom_doc_id_prefix(self) -> None:
        chunks = chunk_text("एक वाक्य।", doc_id="hi-x")
        self.assertEqual(chunks[0].chunk_id, "hi-x#0")
        self.assertEqual(chunks[0].doc_id, "hi-x")

    def test_empty_text_returns_no_chunks(self) -> None:
        self.assertEqual(chunk_text("   "), [])

    def test_invalid_budgets_raise(self) -> None:
        with self.assertRaises(ValueError):
            chunk_text(self.LONG, max_chars=0)
        with self.assertRaises(ValueError):
            chunk_text(self.LONG, max_chars=100, overlap_chars=200)

    def test_very_long_sentence_is_kept_whole(self) -> None:
        sentence = "बहुत लंबा वाक्य " + "शब्द " * 60 + "समाप्त।"
        chunks = chunk_text(sentence, max_chars=50, overlap_chars=10)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].text, split_sentences(sentence)[0])


class CorpusLoadingTests(unittest.TestCase):
    def test_loads_all_six_documents(self) -> None:
        documents = load_documents(str(DOCS))
        self.assertEqual(len(documents), 6)
        ids = {doc.doc_id for doc in documents}
        self.assertEqual(
            ids,
            {"hi-upi", "hi-monsoon", "hi-india-ai", "mr-solapur", "mr-marathi", "mr-railway"},
        )

    def test_header_is_parsed_and_stripped_from_body(self) -> None:
        documents = {d.doc_id: d for d in load_documents(str(DOCS))}
        upi = documents["hi-upi"]
        self.assertEqual(upi.lang, "hi")
        self.assertIn("UPI", upi.title)
        self.assertFalse(upi.text.lstrip().startswith("---"))
        self.assertNotIn("lang: hi", upi.text)
        self.assertEqual(documents["mr-solapur"].lang, "mr")

    def test_header_values_are_not_in_body(self) -> None:
        for document in load_documents(str(DOCS)):
            self.assertNotIn("title:", document.text)
            self.assertTrue(document.text.strip())

    def test_missing_path_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            load_documents(str(REPO / "data" / "nope"))

    def test_iter_chunks_covers_every_document(self) -> None:
        documents = load_documents(str(DOCS))
        chunks = iter_chunks(documents, max_chars=300, overlap_chars=80)
        self.assertEqual({c.doc_id for c in chunks}, {d.doc_id for d in documents})


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
