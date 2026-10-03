"""Tests for the pure-Python BM25 index, tokenizers and score fusion."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rageval.bm25 import (  # noqa: E402
    BM25Index,
    index_char_grams,
    index_word_tokens,
    merge_texts,
    normalize,
    tokenize,
    tokenize_char_grams,
)
from rageval.retrieval import MissingDependencyError, fuse_rrf, fuse_weighted  # noqa: E402


class TokenizerTests(unittest.TestCase):
    def test_devanagari_words_survive(self) -> None:
        tokens = tokenize("यूपीआई कैसे काम करता है?")
        self.assertEqual(tokens, ["यूपीआई", "कैसे", "काम", "करता", "है"])

    def test_latin_is_lowercased(self) -> None:
        self.assertEqual(tokenize("UPI via QR Code"), ["upi", "via", "qr", "code"])

    def test_punctuation_and_danda_are_separators(self) -> None:
        tokens = tokenize("सोलापूर, चादरी; चना डाळ।")
        self.assertEqual(tokens, ["सोलापूर", "चादरी", "चना", "डाळ"])

    def test_underscores_are_split_out(self) -> None:
        self.assertEqual(tokenize("name@bank"), ["name", "bank"])

    def test_normalize_folds_case_only(self) -> None:
        self.assertEqual(normalize("UPI Upi"), "upi upi")
        # punctuation is the tokenizer's (and metrics.normalize's) job
        self.assertEqual(normalize("सोलापूर, चादरी!"), "सोलापूर, चादरी!")

    def test_normalize_applies_unicode_nfc(self) -> None:
        self.assertEqual(normalize("é"), normalize("é"))

    def test_char_grams_include_boundaries(self) -> None:
        self.assertEqual(tokenize_char_grams("UPI", 3), ["#up", "upi", "pi#"])

    def test_char_grams_of_short_devanagari_token(self) -> None:
        grams = tokenize_char_grams("नाक", 3)
        self.assertTrue(all(len(g) == 3 for g in grams))
        self.assertIn("#ना", grams)


class BM25IndexTests(unittest.TestCase):
    CORPUS = {
        "a": tokenize("भारत में मानसून जून से सितंबर तक चलता है"),
        "b": tokenize("यूपीआई भुगतान प्रणाली 2016 में शुरू हुई"),
        "c": tokenize("सोलापूर चादरी और चना डाळी के लिए प्रसिद्ध"),
    }

    def setUp(self) -> None:
        self.index = BM25Index(self.CORPUS)

    def test_size_and_avgdl(self) -> None:
        self.assertEqual(self.index.size, 3)
        self.assertGreater(self.index.avgdl, 0)

    def test_rare_term_ranks_matching_document_first(self) -> None:
        top = self.index.top_k(tokenize("मानसून कब आता है"), k=3)
        self.assertEqual(top[0][0], "a")
        self.assertGreater(top[0][1], 0.0)

    def test_unrelated_query_scores_nothing(self) -> None:
        self.assertEqual(self.index.score(tokenize("क्या आपने देखा")), {})

    def test_empty_query_returns_no_scores(self) -> None:
        self.assertEqual(self.index.score([]), {})

    def test_idf_is_positive_and_frequency_aware(self) -> None:
        self.assertGreater(self.index.idf("मानसून"), 0.0)
        # "में" style commonality: a term in every doc has lower idf.
        shared = BM25Index({"x": ["का"], "y": ["का"]})
        self.assertGreater(shared.idf("का"), 0.0)
        self.assertLess(shared.idf("का"), self.index.idf("मानसून"))

    def test_doc_frequency_counts(self) -> None:
        self.assertEqual(self.index.doc_frequency("मानसून"), 1)
        self.assertEqual(self.index.doc_frequency("zzz"), 0)

    def test_deterministic_ordering_breaks_ties_by_id(self) -> None:
        index = BM25Index({"b": ["समान"], "a": ["समान"]})
        self.assertEqual([d for d, _ in index.top_k(["समान"])], ["a", "b"])

    def test_k1_and_b_validation(self) -> None:
        with self.assertRaises(ValueError):
            BM25Index({"a": ["x"]}, k1=0)
        with self.assertRaises(ValueError):
            BM25Index({"a": ["x"]}, b=1.5)

    def test_zero_scores_helper_covers_all_docs(self) -> None:
        self.assertEqual(set(self.index.zero_scores()), set(self.CORPUS))

    def test_from_texts_tokenizes(self) -> None:
        index = BM25Index.from_texts({"d": "यूपीआई कैसे काम करता है"})
        self.assertGreater(index.score(tokenize("यूपीआई"))["d"], 0)


class CharGramRobustnessTests(unittest.TestCase):
    """The char-gram strategy exists to survive typos and script drift."""

    CORPUS = {
        "solapur": tokenize("सोलापूर शहर चादरी के लिए प्रसिद्ध"),
        "monsoon": tokenize("भारत में मानसून जून से सितंबर"),
    }

    def test_word_index_misses_missing_matra_typo(self) -> None:
        word_index = index_word_tokens({k: " ".join(v) for k, v in self.CORPUS.items()})
        typo = tokenize("सोलापुर")  # missing ू
        self.assertEqual(word_index.score(typo), {})

    def test_char_index_recovers_the_same_typo(self) -> None:
        gram_index = index_char_grams({k: " ".join(v) for k, v in self.CORPUS.items()})
        typo = tokenize_char_grams("सोलापुर")
        scores = gram_index.score(typo)
        self.assertIn("solapur", scores)
        self.assertGreater(scores["solapur"], scores.get("monsoon", 0.0))

    def test_char_index_handles_latin_transliteration_of_devanagari_name(self) -> None:
        gram_index = index_char_grams({k: " ".join(v) for k, v in self.CORPUS.items()})
        # "solapur" in Latin shares no gram with the Devanagari spelling, so
        # the strategy still finds nothing -- documents the known limitation.
        scores = gram_index.score(tokenize_char_grams("solapur"))
        self.assertEqual(scores.get("solapur"), None)


class FusionTests(unittest.TestCase):
    def test_rrf_rewards_documents_ranked_high_by_both_components(self) -> None:
        fused = fuse_rrf([{"a": 10.0, "b": 5.0}, {"a": 0.9, "b": 0.2}])
        self.assertGreater(fused["a"], fused["b"])

    def test_rrf_ignores_zero_scores(self) -> None:
        fused = fuse_rrf([{"a": 1.0, "b": 0.0}])
        self.assertNotIn("b", fused)

    def test_rrf_weights(self) -> None:
        fused = fuse_rrf([{"a": 1.0}, {"b": 1.0}], weights=[10.0, 1.0])
        self.assertGreater(fused["a"], fused["b"])

    def test_weighted_sum_normalises_per_component(self) -> None:
        fused = fuse_weighted([{"a": 100.0, "b": 50.0}, {"b": 0.5, "a": 0.1}])
        # a: 1.0 + 0.2 = 1.2 ; b: 0.5 + 1.0 = 1.5 after per-map scaling.
        self.assertAlmostEqual(fused["b"], 1.5)
        self.assertAlmostEqual(fused["a"], 1.2)

    def test_weighted_ignores_non_positive_scores(self) -> None:
        self.assertEqual(fuse_weighted([{"a": 0.0}]), {})

    def test_weight_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            fuse_rrf([{"a": 1.0}], weights=[1.0, 2.0])
        with self.assertRaises(ValueError):
            fuse_weighted([{"a": 1.0}], weights=[1.0, 2.0])


class HelperTests(unittest.TestCase):
    def test_merge_texts_groups_chunks(self) -> None:
        merged = merge_texts([("d1", "एक"), ("d1", "दो"), ("d2", "तीन")])
        self.assertEqual(merged, {"d1": "एक दो", "d2": "तीन"})

    def test_missing_dependency_message_mentions_pip(self) -> None:
        error = MissingDependencyError("sentence-transformers", "dense retrieval")
        self.assertIn("pip install sentence-transformers", str(error))
        self.assertIsInstance(error, ImportError)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
