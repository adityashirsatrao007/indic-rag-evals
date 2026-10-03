"""Command line interface: ``evaluate`` and ``search`` subcommands.

Run as ``python3 -m rageval ...`` from the repository root (or anywhere, with
``PYTHONPATH=src``).  Everything is offline; the metrics a run reports are
deterministic (only the report's "generated" timestamp varies between runs).
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .chunking import Document, load_documents
from .generation import GENERATORS, generate_answer
from .metrics import StrategyMetrics, evaluate_rankings
from .retrieval import (
    DenseEncoder,
    MissingDependencyError,
    Retriever,
    Strategy,
    iter_document_languages,
    load_qrels,
)

__all__ = ["main"]

DEFAULT_STRATEGIES = (Strategy.BM25, Strategy.CHAR_GRAM, Strategy.HYBRID)
DEFAULT_K_VALUES = (1, 3, 5)
CSV_FIELDS = (
    "strategy",
    "recall@1",
    "recall@3",
    "recall@5",
    "mrr",
    "token_f1",
    "exact_match",
    "n_queries",
)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    """Render a GitHub-flavoured markdown table with aligned cells."""
    widths = [len(h) for h in headers]
    body: list[list[str]] = []
    for row in rows:
        cells = ["" if cell is None else str(cell) for cell in row]
        body.append(cells)
        for i, cell in enumerate(cells):
            widths[i] = max(widths[i], len(cell))
    def fmt(cells: Sequence[str]) -> str:
        return "| " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(cells)) + " |"
    lines = [fmt(list(headers)), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(fmt(row) for row in body)
    return "\n".join(lines)


def _parse_strategies(values: Sequence[str] | None) -> list[Strategy]:
    if not values:
        return list(DEFAULT_STRATEGIES)
    parsed: list[Strategy] = []
    for raw in values:
        for item in str(raw).split(","):
            if item.strip():
                parsed.append(Strategy.parse(item))
    if not parsed:
        raise ValueError(
            "no strategy given (choose from "
            + ", ".join(s.value for s in Strategy)
            + ")"
        )
    return list(dict.fromkeys(parsed))  # de-duplicate, keep order


def _summaries(results: Sequence[StrategyMetrics]) -> tuple[list[str], list[list[object]]]:
    headers = [
        "strategy",
        "recall@1",
        "recall@3",
        "recall@5",
        "mrr",
        "token_f1",
        "exact_match",
    ]
    rows: list[list[object]] = []
    for result in results:
        row = result.row(DEFAULT_K_VALUES)
        rows.append([row.get(header, "") for header in headers])
    return headers, rows


# --------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------
def cmd_evaluate(args: argparse.Namespace) -> int:
    documents = load_documents(args.docs)
    queries = load_qrels(args.qrels)
    strategies = _parse_strategies(args.strategies)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    needs_dense = any(strategy.uses_dense for strategy in strategies)
    encoder = DenseEncoder() if needs_dense else None

    k_values = DEFAULT_K_VALUES
    # Recall@k needs k retrieved documents even when --top-k asks for fewer,
    # so the per-query depth is the largest k we report on.
    depth = max(args.top_k, max(k_values))
    results: list[StrategyMetrics] = []

    for strategy in strategies:
        retriever = Retriever(
            documents,
            strategy=strategy,
            fusion=args.fusion,
            max_chars=args.max_chars,
            overlap_chars=args.overlap,
            dense_encoder=encoder,
        )
        rankings: list[list[str]] = []
        predictions: list[str] = []
        for query in queries:
            hits = retriever.search(query.question, top_k=depth)
            rankings.append([hit.doc_id for hit in hits])
            predictions.append(
                generate_answer(
                    args.generator,
                    query.question,
                    documents,
                    hits,
                    index=retriever.word_index,
                    max_sentences=args.sentences,
                )
            )

        with_answers = args.generator != "none"
        results.append(
            evaluate_rankings(
                strategy.value,
                rankings,
                [query.relevant_doc_ids for query in queries],
                predictions=predictions if with_answers else None,
                gold_answers=[query.answer for query in queries] if with_answers else None,
                query_ids=[query.query_id for query in queries],
                k_values=k_values,
            )
        )

    headers, rows = _summaries(results)
    table = _markdown_table(headers, rows)

    _write_csv(out_dir / "metrics.csv", results)
    _write_report(
        out_dir / "report.md",
        results,
        documents=documents,
        n_queries=len(queries),
        fusion=args.fusion,
        generator=args.generator,
        depth=depth,
        max_chars=args.max_chars,
        overlap=args.overlap,
        table=table,
    )

    print(f"corpus: {len(documents)} documents, {len(queries)} queries")
    print(f"out:    {out_dir / 'metrics.csv'}, {out_dir / 'report.md'}")
    print()
    print(table)
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    documents = load_documents(args.docs)
    strategy = Strategy.parse(args.strategy)
    encoder = DenseEncoder() if strategy.uses_dense else None
    retriever = Retriever(
        documents,
        strategy=strategy,
        fusion=args.fusion,
        max_chars=args.max_chars,
        overlap_chars=args.overlap,
        dense_encoder=encoder,
    )
    hits = retriever.search(args.query, top_k=args.top_k)

    print(
        f"strategy: {strategy.value}   fusion: {args.fusion}   "
        f"corpus: {len(documents)} docs / {len(retriever.chunks)} chunks"
    )
    if not hits:
        print("no results (no indexed term matched the query)")
        return 0
    print(_markdown_table(("rank", "score", "doc_id", "title"),
                          [(h.rank, f"{h.score:.4f}", h.doc_id, h.title) for h in hits]))
    if args.show_chunk:
        for hit in hits:
            snippet = retriever.chunk_text(hit.chunk_id)
            print(f"\n[{hit.rank}] {hit.doc_id} ({hit.chunk_id}, score {hit.score:.4f})")
            print(f"    {snippet}")
    if args.explain:
        print("\nper-component scores:")
        for name, values in retriever.explain(args.query).items():
            rendered = ", ".join(f"{doc}={score:.4f}" for doc, score in values) or "-"
            print(f"  {name}: {rendered}")
    return 0


# --------------------------------------------------------------------------
# writers
# --------------------------------------------------------------------------
def _write_csv(path: Path, results: Sequence[StrategyMetrics]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS))
        writer.writeheader()
        for result in results:
            writer.writerow(result.row(DEFAULT_K_VALUES))


def _write_report(
    path: Path,
    results: Sequence[StrategyMetrics],
    *,
    documents: Sequence[Document],
    n_queries: int,
    fusion: str,
    generator: str,
    depth: int,
    max_chars: int,
    overlap: int,
    table: str,
) -> None:
    languages = ", ".join(iter_document_languages(documents)) or "n/a"
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    lines: list[str] = [
        "# indic-rag-evals — evaluation report",
        "",
        f"- generated: {generated}",
        f"- corpus: {len(documents)} documents, languages: {languages}",
        f"- queries: {n_queries} labelled items",
        f"- retrieval depth: {depth} (Recall@k for k = 1, 3, 5; MRR)",
        f"- fusion: {fusion}, generator: {generator}, "
        f"chunking: max_chars={max_chars}, overlap={overlap}",
        f"- strategies: {', '.join(r.strategy for r in results)}",
        "",
        "## Summary",
        "",
        table,
        "",
        "## Per-strategy detail",
        "",
    ]

    detail_headers = [
        "query_id",
        "relevant",
        "top-1",
        "recall@1",
        "recall@3",
        "recall@5",
        "mrr",
        "token_f1",
        "em",
    ]
    for result in results:
        lines.append(f"### {result.strategy}")
        lines.append("")
        detail_rows: list[list[object]] = []
        for item in result.per_query:
            detail_rows.append(
                [
                    item.query_id,
                    ", ".join(item.relevant_doc_ids),
                    item.ranked_doc_ids[0] if item.ranked_doc_ids else "-",
                    f"{item.recall[1]:.4f}",
                    f"{item.recall[3]:.4f}",
                    f"{item.recall[5]:.4f}",
                    f"{item.rr:.4f}",
                    "" if result.token_f1 is None else f"{item.token_f1:.4f}",
                    "" if result.exact_match is None else f"{item.exact:.4f}",
                ]
            )
        lines.append(_markdown_table(detail_headers, detail_rows))
        lines.append("")

    lines.extend(
        [
            "## How to read this",
            "",
            "- `recall@k` = share of gold documents found in the top *k* hits,",
            "  averaged over all queries; `mrr` = mean of 1/rank of the first hit.",
            "- `token_f1` / `exact_match` compare the generated answer with the",
            "  reference answer after normalising case, whitespace and Devanagari",
            "  punctuation (danda). They are `\"\"` when `--generator none`.",
            "- Caveats: the corpus and eval set above are small, so Recall@5",
            "  saturates easily and the numbers are a regression baseline,",
            "  not a benchmark.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# parser / entry point
# --------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rageval",
        description="Offline RAG retrieval evaluation for Hindi/Marathi corpora.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    evaluate = sub.add_parser(
        "evaluate", help="run the full harness and write metrics.csv + report.md"
    )
    evaluate.add_argument("--docs", required=True, help="corpus directory or file")
    evaluate.add_argument("--qrels", required=True, help="labelled qa.jsonl file")
    evaluate.add_argument("--out", required=True, help="output directory")
    evaluate.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="results retrieved per query (floored at 5 so Recall@5 is computable)",
    )
    evaluate.add_argument(
        "--strategies",
        action="append",
        default=None,
        help="strategy name(s), comma separated (default: bm25,char-gram,hybrid)",
    )
    evaluate.add_argument(
        "--fusion", choices=("rrf", "weighted"), default="rrf", help="hybrid fusion method"
    )
    evaluate.add_argument(
        "--generator", choices=GENERATORS, default="extractive", help="answer generator"
    )
    evaluate.add_argument(
        "--sentences", type=int, default=1, help="sentences per extractive answer"
    )
    evaluate.add_argument("--max-chars", type=int, default=480, help="chunk size budget")
    evaluate.add_argument("--overlap", type=int, default=120, help="chunk overlap budget")
    evaluate.set_defaults(func=cmd_evaluate)

    search = sub.add_parser("search", help="rank the corpus for a single query")
    search.add_argument("--query", required=True, help="natural language query")
    search.add_argument("--docs", required=True, help="corpus directory or file")
    search.add_argument("--top-k", type=int, default=5, help="number of hits to show")
    search.add_argument(
        "--strategy",
        default=Strategy.BM25.value,
        help="bm25 | char-gram | hybrid | hybrid+dense",
    )
    search.add_argument("--fusion", choices=("rrf", "weighted"), default="rrf")
    search.add_argument("--max-chars", type=int, default=480)
    search.add_argument("--overlap", type=int, default=120)
    search.add_argument("--show-chunk", action="store_true", help="print the best chunk text")
    search.add_argument("--explain", action="store_true", help="print per-component scores")
    search.set_defaults(func=cmd_search)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns a process exit code (0 = success)."""
    parser = build_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        return int(args.func(args))
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    except MissingDependencyError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 3


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
