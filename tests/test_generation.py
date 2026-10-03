"""Tests for the offline extractive answer generator.

``generation.py`` is the generator behind every default ``evaluate`` run, so
its scoring arithmetic is pinned here with hand-computed values (the CLI test
only pins the aggregate it feeds into).
"""

from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rageval.chunking import Document  # noqa: E402
from rageval.generation import (  # noqa: E402
    extractive_answer,
    resolve_llm_config,
    score_sentences,
)
from rageval.retrieval import ScoredDoc  # noqa: E402


class ScoreSentencesTests(unittest.TestCase):
    QUERY = "मानसून कब आता है"  # -> tokens: मानसून, कब, आता, है

    SENTENCES = [
        "मानसून जून में आता है",  # 3 query matches, 5 distinct tokens
        "कब आता है यह नहीं पता",  # 3 query matches, 6 distinct tokens
        "सोलापूर जिल्हा प्रसिद्ध आहे",  # no match (Marathi "आहे", not "है")
    ]

    def test_weights_by_idf_and_length(self) -> None:
        ranked = score_sentences(self.QUERY, self.SENTENCES)
        self.assertEqual([position for position, _ in ranked], [0, 1, 2])
        # equally good matches, so the shorter sentence wins: 3 / sqrt(distinct)
        self.assertAlmostEqual(ranked[0][1], 3.0 / 5**0.5)
        self.assertAlmostEqual(ranked[1][1], 3.0 / 6**0.5)
        self.assertEqual(ranked[2][1], 0.0)

        def expensive_monsoon(term: str) -> float:
            return 10.0 if term == "मानसून" else 0.0

        weighted = score_sentences(self.QUERY, self.SENTENCES, expensive_monsoon)
        self.assertAlmostEqual(weighted[0][1], 10.0 / 5**0.5)


class ExtractiveAnswerTests(unittest.TestCase):
    TEXT = "पहला वाक्य निरर्थक है। मानसून जून में आता है। अंतिम वाक्य।"
    DOCUMENT = Document(doc_id="d1", title="", lang="hi", text=TEXT)
    HIT = ScoredDoc(doc_id="d1", score=1.0, rank=1)

    def test_returns_the_best_sentence_of_the_top_document(self) -> None:
        answer = extractive_answer("मानसून कब आता है", [self.DOCUMENT], [self.HIT])
        # 3 matches over 5 tokens beats 1 match over 4 (है alone)
        self.assertEqual(answer, "मानसून जून में आता है।")
        answer = extractive_answer(
            "मानसून कब आता है", [self.DOCUMENT], [self.HIT], max_sentences=2
        )
        self.assertEqual(answer, "मानसून जून में आता है। पहला वाक्य निरर्थक है।")

    def test_no_overlap_falls_back_to_the_first_sentence(self) -> None:
        answer = extractive_answer("कोई और बात ही नहीं", [self.DOCUMENT], [self.HIT])
        self.assertEqual(answer, "पहला वाक्य निरर्थक है।")
        # no documents or no retrieved hit -> nothing to quote
        self.assertEqual(extractive_answer("x", [], [self.HIT]), "")
        self.assertEqual(extractive_answer("x", [self.DOCUMENT], []), "")


class ResolveLlmConfigTests(unittest.TestCase):
    """Environment precedence for the optional ``--generator llm`` path."""

    def test_precedence_between_arguments_and_variables(self) -> None:
        env = {
            "SARVAM_API_KEY": "sarvam-key",
            "LLM_API_KEY": "generic-key",
            "SARVAM_API_BASE": "https://sarvam.example/v1",
            "LLM_BASE_URL": "https://generic.example/v1/",
            "SARVAM_MODEL": "sarvam-model",
            "LLM_MODEL": "generic-model",
        }
        with patch.dict(os.environ, env, clear=True):
            # the key prefers SARVAM_*; base URL and model prefer LLM_*
            self.assertEqual(
                resolve_llm_config(),
                ("sarvam-key", "https://generic.example/v1", "generic-model"),
            )
            # explicit arguments beat every variable
            self.assertEqual(
                resolve_llm_config(
                    api_key="arg-key",
                    base_url="https://arg.example/v1",
                    model="arg-model",
                ),
                ("arg-key", "https://arg.example/v1", "arg-model"),
            )
        # with only the generic variables absent, SARVAM_* still resolves
        with patch.dict(
            os.environ,
            {k: v for k, v in env.items() if k.startswith("SARVAM_")},
            clear=True,
        ):
            self.assertEqual(
                resolve_llm_config(),
                ("sarvam-key", "https://sarvam.example/v1", "sarvam-model"),
            )

    def test_missing_key_raises_naming_the_variables(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError) as caught:
                resolve_llm_config()
        message = str(caught.exception)
        self.assertIn("SARVAM_API_KEY", message)
        self.assertIn("LLM_API_KEY", message)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
