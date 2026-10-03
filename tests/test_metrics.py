"""Tests for retrieval metrics (Recall@k, MRR) and answer metrics (F1, EM)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rageval.metrics import (  # noqa: E402
    exact_match,
    evaluate_rankings,
    mean,
    mrr,
    normalize,
    recall_at_k,
    reciprocal_rank,
    token_f1,
)


class NormalizationTests(unittest.TestCase):
    def test_danda_and_punctuation_are_dropped(self) -> None:
        self.assertEqual(normalize("क्यूआर कोड स्कैन करके भुगतान।"), "क्यूआर कोड स्कैन करके भुगतान")

    def test_case_and_whitespace_collapse(self) -> None:
        self.assertEqual(normalize("  UPI\tvia\nQR  code "), "upi via qr code")

    def test_devanagari_matras_survive(self) -> None:
        self.assertEqual(normalize("मराठी भाषा"), "मराठी भाषा")

    def test_commas_inside_numbers_become_spaces(self) -> None:
        self.assertEqual(normalize("10,372 करोड़ रुपये"), "10 372 करोड़ रुपये")

    def test_empty_string_stays_empty(self) -> None:
        self.assertEqual(normalize("।।।   !!!"), "")


class TokenF1Tests(unittest.TestCase):
    def test_identical_text_scores_one(self) -> None:
        text = "सोलापूर चादरी के लिए प्रसिद्ध आहे।"
        self.assertEqual(token_f1(text, text), 1.0)

    def test_punctuation_and_case_differences_still_score_one(self) -> None:
        self.assertEqual(token_f1("यूपीआई कैसे काम करता है?", "यूपीआई कैसे काम करता है।"), 1.0)
        self.assertEqual(token_f1("UPI भुगतान", "upi भुगतान"), 1.0)

    def test_partial_overlap_is_between_zero_and_one(self) -> None:
        score = token_f1("मानसून जून में आता है", "मानसून सितंबर तक चलता है")
        self.assertGreater(score, 0.0)
        self.assertLess(score, 1.0)

    def test_disjoint_texts_score_zero(self) -> None:
        self.assertEqual(token_f1("सोलापूर जिल्हा", "भारत में मानसून"), 0.0)

    def test_empty_side_scores_zero(self) -> None:
        self.assertEqual(token_f1("", "कुछ भी नहीं"), 0.0)
        self.assertEqual(token_f1("कुछ भी नहीं", ""), 0.0)


class ExactMatchTests(unittest.TestCase):
    def test_match_after_normalisation(self) -> None:
        self.assertEqual(exact_match("  उत्तर:  मराठी. ", "उत्तर मराठी"), 1.0)

    def test_mismatch(self) -> None:
        self.assertEqual(exact_match("देवनागरी", "लैटिन"), 0.0)

    def test_extra_token_breaks_match(self) -> None:
        self.assertEqual(exact_match("जून से सितंबर", "जून"), 0.0)


class RecallTests(unittest.TestCase):
    def test_relevant_document_at_rank_one(self) -> None:
        self.assertEqual(recall_at_k(["a", "b", "c"], ["a"], 1), 1.0)
        self.assertEqual(recall_at_k(["b", "c", "a"], ["a"], 1), 0.0)
        self.assertEqual(recall_at_k(["b", "c", "a"], ["a"], 3), 1.0)

    def test_two_relevant_documents_half_found(self) -> None:
        self.assertEqual(recall_at_k(["a", "x", "y"], ["a", "b"], 3), 0.5)
        self.assertEqual(recall_at_k(["a", "b", "y"], ["a", "b"], 3), 1.0)

    def test_ranking_shorter_than_k(self) -> None:
        self.assertEqual(recall_at_k(["b"], ["a"], 5), 0.0)

    def test_duplicate_relevant_ids_are_counted_once(self) -> None:
        self.assertEqual(recall_at_k(["a"], ["a", "a"], 1), 1.0)

    def test_invalid_arguments_raise(self) -> None:
        with self.assertRaises(ValueError):
            recall_at_k(["a"], [], 1)
        with self.assertRaises(ValueError):
            recall_at_k(["a"], ["a"], 0)


class ReciprocalRankTests(unittest.TestCase):
    def test_first_position(self) -> None:
        self.assertEqual(reciprocal_rank(["a", "b"], ["a"]), 1.0)

    def test_second_position(self) -> None:
        self.assertEqual(reciprocal_rank(["b", "a", "c"], ["a"]), 0.5)

    def test_absent_document_is_zero(self) -> None:
        self.assertEqual(reciprocal_rank(["b", "c"], ["a"]), 0.0)

    def test_uses_first_relevant_hit(self) -> None:
        self.assertEqual(reciprocal_rank(["a", "b"], ["a", "b"]), 1.0)

    def test_mrr_averages_over_queries(self) -> None:
        value = mrr([["a", "b"], ["x", "y", "z"], ["b", "a"]], [["b"], ["z"], ["a"]])
        # hits land at rank 2, rank 3 and rank 2 respectively
        self.assertAlmostEqual(value, (0.5 + 1 / 3 + 0.5) / 3)

    def test_mrr_rejects_mismatched_batches(self) -> None:
        with self.assertRaises(ValueError):
            mrr([["a"]], [["a"], ["b"]])


class AggregationTests(unittest.TestCase):
    def test_mean_of_empty_is_zero(self) -> None:
        self.assertEqual(mean([]), 0.0)

    def test_evaluate_rankings_without_predictions(self) -> None:
        result = evaluate_rankings(
            "bm25",
            [["a", "b"], ["x", "a"]],
            [["a"], ["a"]],
            query_ids=["q1", "q2"],
        )
        self.assertEqual(result.n_queries, 2)
        self.assertEqual(result.recall[1], 0.5)
        self.assertEqual(result.recall[5], 1.0)
        self.assertAlmostEqual(result.mrr, 0.75)
        self.assertIsNone(result.token_f1)
        self.assertIsNone(result.exact_match)
        row = result.row((1, 3, 5))
        self.assertEqual(row["strategy"], "bm25")
        self.assertEqual(row["token_f1"], "")
        self.assertEqual(row["n_queries"], 2)

    def test_evaluate_rankings_with_predictions(self) -> None:
        result = evaluate_rankings(
            "hybrid",
            [["a"]],
            [["a"]],
            predictions=["मानसून जून में"],
            gold_answers=["मानसून जून में आता है"],
            query_ids=["q1"],
        )
        self.assertIsNotNone(result.token_f1)
        self.assertGreater(result.token_f1, 0.0)
        self.assertEqual(result.exact_match, 0.0)
        self.assertEqual(result.per_query[0].query_id, "q1")
        self.assertEqual(result.per_query[0].ranked_doc_ids, ("a",))

    def test_length_mismatch_raises(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_rankings("bm25", [["a"]], [["a"], ["b"]])

    def test_prediction_alignment_is_checked(self) -> None:
        with self.assertRaises(ValueError):
            evaluate_rankings(
                "bm25",
                [["a"]],
                [["a"]],
                predictions=[],
                gold_answers=["x"],
            )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
