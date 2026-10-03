"""End-to-end CLI tests.

Each test runs ``python3 -m rageval`` as a real subprocess in a temporary
working directory, exactly the way a hiring manager would run it, and checks
the artefacts it writes.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
DOCS = REPO / "data" / "documents"
QRELS = REPO / "data" / "eval" / "qa.jsonl"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from rageval.cli import main  # noqa: E402


def run_cli(*args: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    """Run the CLI as a subprocess with PYTHONPATH=src, in *cwd*."""
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + os.pathsep + env.get("PYTHONPATH", "")
    return subprocess.run(
        [sys.executable, "-m", "rageval", *args],
        cwd=str(cwd),
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )


class EvaluateCliTests(unittest.TestCase):
    def test_evaluate_writes_metrics_csv_and_report(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            out = work / "results"
            proc = run_cli(
                "evaluate",
                "--docs", str(DOCS),
                "--qrels", str(QRELS),
                "--out", str(out),
                cwd=work,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("corpus: 6 documents, 12 queries", proc.stdout)
            self.assertIn("| strategy", proc.stdout)
            for strategy in ("bm25", "char-gram", "hybrid"):
                self.assertIn(strategy, proc.stdout)

            metrics_path = out / "metrics.csv"
            report_path = out / "report.md"
            self.assertTrue(metrics_path.is_file())
            self.assertTrue(report_path.is_file())

            with metrics_path.open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual([row["strategy"] for row in rows],
                             ["bm25", "char-gram", "hybrid"])

            # Regression baseline: these are the numbers committed in
            # results/metrics.csv.  Asserting them (rather than just "0 <= x
            # <= 1") is what makes a broken retriever or metric fail here --
            # an empty ranking would still satisfy a range check.
            golden = {
                #                        R@1     R@3  R@5  MRR     F1      EM
                "bm25":      (0.8333, 1.0, 1.0, 0.9444, 0.7965, 0.75),
                "char-gram": (0.9167, 1.0, 1.0, 1.0,    0.7746, 0.75),
                "hybrid":    (0.8333, 1.0, 1.0, 0.9583, 0.7965, 0.75),
            }
            columns = ("recall@1", "recall@3", "recall@5", "mrr",
                       "token_f1", "exact_match")
            for row in rows:
                for column, expected in zip(columns, golden[row["strategy"]]):
                    self.assertAlmostEqual(
                        float(row[column]), expected, places=4,
                        msg=f"{row['strategy']} {column}",
                    )
                self.assertEqual(int(row["n_queries"]), 12)

            report = report_path.read_text(encoding="utf-8")
            self.assertIn("# indic-rag-evals — evaluation report", report)
            self.assertIn("## Summary", report)
            for strategy in ("### bm25", "### char-gram", "### hybrid"):
                self.assertIn(strategy, report)
            self.assertIn("| q12", report)

    def test_evaluate_is_deterministic_across_runs(self) -> None:
        outputs = []
        for name in ("run-a", "run-b"):
            with tempfile.TemporaryDirectory() as tmp:
                work = Path(tmp)
                proc = run_cli(
                    "evaluate",
                    "--docs", str(DOCS),
                    "--qrels", str(QRELS),
                    "--out", str(work / name),
                    cwd=work,
                )
                self.assertEqual(proc.returncode, 0, proc.stderr)
                outputs.append((work / name / "metrics.csv").read_text(encoding="utf-8"))
        self.assertEqual(outputs[0], outputs[1])

    def test_evaluate_generator_none_leaves_answer_metrics_blank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            proc = run_cli(
                "evaluate",
                "--docs", str(DOCS),
                "--qrels", str(QRELS),
                "--out", str(work / "out"),
                "--generator", "none",
                cwd=work,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            text = (work / "out" / "metrics.csv").read_text(encoding="utf-8")
            lines = [line for line in text.splitlines() if line.startswith("bm25")]
            self.assertTrue(lines)
            self.assertIn(",,", lines[0])  # blank token_f1 + exact_match cells

    def test_evaluate_single_strategy_and_weighted_fusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            proc = run_cli(
                "evaluate",
                "--docs", str(DOCS),
                "--qrels", str(QRELS),
                "--out", str(work / "out"),
                "--strategies", "hybrid",
                "--fusion", "weighted",
                cwd=work,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            text = (work / "out" / "metrics.csv").read_text(encoding="utf-8")
            data_rows = [line for line in text.splitlines()[1:] if line.strip()]
            self.assertEqual(len(data_rows), 1)
            self.assertTrue(data_rows[0].startswith("hybrid,"))


class SearchCliTests(unittest.TestCase):
    def test_search_returns_ranked_results_with_scores(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            proc = run_cli(
                "search",
                "--query", "सोलापूर किस चीज़ के लिए प्रसिद्ध आहे?",
                "--docs", str(DOCS),
                "--top-k", "3",
                cwd=work,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("rank", proc.stdout)
            self.assertIn("score", proc.stdout)
            self.assertIn("mr-solapur", proc.stdout)
            self.assertIn("6 docs", proc.stdout)
            # the city must be rank 1, not merely somewhere in the top-k
            top_row = next(
                line for line in proc.stdout.splitlines() if line.startswith("| 1 ")
            )
            self.assertIn("mr-solapur", top_row)

    def test_search_with_explain_shows_components(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            proc = run_cli(
                "search",
                "--query", "UPI kaise kaam karta hai",
                "--docs", str(DOCS),
                "--strategy", "hybrid",
                "--explain",
                cwd=work,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("per-component scores:", proc.stdout)
            self.assertIn("bm25:", proc.stdout)
            self.assertIn("char-gram:", proc.stdout)

    def test_search_prints_chunk_text_on_request(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            proc = run_cli(
                "search",
                "--query", "मानसून कब आता है",
                "--docs", str(DOCS),
                "--show-chunk",
                cwd=work,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn("hi-monsoon#", proc.stdout)

    def test_unknown_strategy_fails_with_helpful_message(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            proc = run_cli(
                "search", "--query", "x", "--docs", str(DOCS),
                "--strategy", "nope", cwd=work,
            )
            self.assertEqual(proc.returncode, 2)
            self.assertIn("error:", proc.stderr)
            self.assertIn("bm25", proc.stderr)


class InProcessErrorTests(unittest.TestCase):
    """Error paths exercised through ``main()`` with stdout captured."""

    @staticmethod
    def _run_in_process(argv: list[str]) -> int:
        import contextlib
        import io

        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer), contextlib.redirect_stderr(buffer):
            return main(argv)

    def test_missing_documents_path_returns_exit_code_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = self._run_in_process([
                "evaluate",
                "--docs", str(Path(tmp) / "missing"),
                "--qrels", str(QRELS),
                "--out", str(Path(tmp) / "out"),
            ])
        self.assertEqual(code, 2)

    def test_missing_qrels_path_returns_exit_code_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            code = self._run_in_process([
                "evaluate",
                "--docs", str(DOCS),
                "--qrels", str(Path(tmp) / "missing.jsonl"),
                "--out", str(Path(tmp) / "out"),
            ])
        self.assertEqual(code, 2)

    def test_restricted_output_directory_returns_exit_code_two(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            blocker = Path(tmp) / "blocker"
            blocker.write_text("not a directory", encoding="utf-8")
            code = self._run_in_process([
                "evaluate",
                "--docs", str(DOCS),
                "--qrels", str(QRELS),
                "--out", str(blocker),
            ])
        self.assertEqual(code, 2)

    def test_blank_strategy_name_returns_exit_code_two(self) -> None:
        # "--strategies ''" would otherwise evaluate nothing and write an
        # empty metrics.csv without complaining
        with tempfile.TemporaryDirectory() as tmp:
            code = self._run_in_process([
                "evaluate",
                "--docs", str(DOCS),
                "--qrels", str(QRELS),
                "--out", str(Path(tmp) / "out"),
                "--strategies", " , ",
            ])
        self.assertEqual(code, 2)

    def test_unknown_generator_is_rejected_by_argparse(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(SystemExit):
                self._run_in_process([
                    "evaluate",
                    "--docs", str(DOCS),
                    "--qrels", str(QRELS),
                    "--out", str(Path(tmp) / "out"),
                    "--generator", "magic",
                ])


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
