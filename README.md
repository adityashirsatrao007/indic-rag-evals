# indic-rag-evals

**Retrieval-Augmented Generation over Hindi + Marathi documents, with a real,
reproducible evaluation harness.**

Offline, deterministic, and dependency-free: the core path runs on the Python 3
standard library only (developed and tested on Python 3.14.7). No API keys, no
network, no `pip install` required to reproduce every number on this page.

---

## Why this exists

I already have a RAG project that reports a **"92% hit-rate."** That number was
eyeballed from a demo: no labelled query set, no stated metric, no definition of
"hit," no way for anyone else to re-run it. For a job where the work is *measuring*
retrieval and generation quality, an unverifiable headline number is worse than no
number.

This repo replaces the vibe with an artifact:

- a **labelled eval set** (`data/eval/qa.jsonl`, 12 queries, gold document ids +
  gold answers),
- **standard retrieval metrics** computed by hand-written, tested code
  (Recall@1/3/5, MRR) and **answer metrics** (token-F1, exact match),
- an **offline pipeline** that writes `results/metrics.csv` and
  `results/report.md` on any machine in under a second,
- an **error analysis** you can read query by query, not just an average.

For what it is worth, the old "92%" lands in a plausible place: on this set the
char-gram index reaches **Recall@1 = 0.9167**. The difference is that 0.9167 here
is defined, computed by `metrics.py`, asserted by `tests/`, and traceable to the
individual queries that failed.

---

## What it measures

| Group | Metric | Definition used |
| --- | --- | --- |
| Retrieval | `recall@1`, `recall@3`, `recall@5` | share of gold doc ids present in the top *k* hits, averaged over queries |
| Retrieval | `mrr` | mean of `1 / rank` of the first gold hit (0 if none) |
| Generation | `token_f1` | bag-of-tokens F1 between generated and reference answer, after normalisation |
| Generation | `exact_match` | 1.0 iff normalised generated answer == normalised reference answer |

Normalisation is Devanagari-aware: case, whitespace runs and punctuation
(including the danda `।` and `॥`) are stripped, while matras/vowel signs are
**kept** — so `क्यूआर कोड स्कैन करके भुगतान।` and `क्यूआर कोड स्कैन करके भुगतान`
compare as equal.

Each strategy is evaluated end-to-end: retrieval *and* the answer generator
attached to its top hit.

---

## Reproduce it (3 commands)

```bash
# 1. the test suite (stdlib unittest, 90 tests)
cd indic-rag-evals && PYTHONPATH=src python3 -m unittest discover -s tests -v

# 2. the full evaluation -> results/metrics.csv + results/report.md
PYTHONPATH=src python3 -m rageval evaluate --docs data/documents --qrels data/eval/qa.jsonl --out results/

# 3. ad-hoc search
PYTHONPATH=src python3 -m rageval search --query "UPI se railway ticket ka bhugtan online ho sakta hai kya?" --docs data/documents --top-k 3 --explain
```

Requirements: Python 3.10+ (tested on 3.14.7), nothing else. `requirements.txt`
contains only commented-out optional/dev lines.

---

## Real output

### `rageval evaluate` (stdout)

Produced by command 2 above, pasted verbatim:

```
corpus: 6 documents, 12 queries
out:    results/metrics.csv, results/report.md

| strategy  | recall@1 | recall@3 | recall@5 | mrr    | token_f1 | exact_match |
| --------- | -------- | -------- | -------- | ------ | -------- | ----------- |
| bm25      | 0.8333   | 1.0      | 1.0      | 0.9444 | 0.7965   | 0.75        |
| char-gram | 0.9167   | 1.0      | 1.0      | 1.0    | 0.7746   | 0.75        |
| hybrid    | 0.8333   | 1.0      | 1.0      | 0.9583 | 0.7965   | 0.75        |
```

### `results/metrics.csv`

```csv
strategy,recall@1,recall@3,recall@5,mrr,token_f1,exact_match,n_queries
bm25,0.8333,1.0,1.0,0.9444,0.7965,0.75,12
char-gram,0.9167,1.0,1.0,1.0,0.7746,0.75,12
hybrid,0.8333,1.0,1.0,0.9583,0.7965,0.75,12
```

### `rageval search --explain`

```
strategy: bm25   fusion: rrf   corpus: 6 docs / 12 chunks
| rank | score  | doc_id     | title                           |
| ---- | ------ | ---------- | ------------------------------- |
| 1    | 2.3103 | hi-upi     | यूपीआई (UPI) कैसे काम करता है   |
| 2    | 2.2270 | mr-railway | भारतीय रेल्वेत तिकीट आणि प्रवास |

per-component scores:
  bm25: hi-upi=2.3103, mr-railway=2.2270
```

### Test suite tail

```
$ PYTHONPATH=src python3 -m unittest discover -s tests -v
...
----------------------------------------------------------------------
Ran 90 tests in 0.672s

OK
```

(The test count is fixed; the timing obviously varies by machine.)

---

## Reading the numbers honestly

Per-query detail for the `hybrid` strategy (the same table is written for every
strategy in `results/report.md`):

| query_id | relevant                | top-1       | recall@1 | recall@3 | recall@5 | mrr    | token_f1 | em     |
| -------- | ----------------------- | ----------- | -------- | -------- | -------- | ------ | -------- | ------ |
| q01      | hi-upi                  | hi-india-ai | 0.0000   | 1.0000   | 1.0000   | 0.5000 | 0.1569   | 0.0000 |
| q02      | hi-monsoon              | hi-monsoon  | 1.0000   | 1.0000   | 1.0000   | 1.0000 | 1.0000   | 1.0000 |
| q03      | hi-upi                  | hi-upi      | 1.0000   | 1.0000   | 1.0000   | 1.0000 | 1.0000   | 1.0000 |
| q04      | hi-india-ai             | hi-india-ai | 1.0000   | 1.0000   | 1.0000   | 1.0000 | 1.0000   | 1.0000 |
| q05      | hi-india-ai             | hi-india-ai | 1.0000   | 1.0000   | 1.0000   | 1.0000 | 1.0000   | 1.0000 |
| q06      | hi-monsoon              | hi-monsoon  | 1.0000   | 1.0000   | 1.0000   | 1.0000 | 0.2222   | 0.0000 |
| q07      | mr-solapur              | mr-solapur  | 1.0000   | 1.0000   | 1.0000   | 1.0000 | 1.0000   | 1.0000 |
| q08      | mr-solapur              | mr-solapur  | 1.0000   | 1.0000   | 1.0000   | 1.0000 | 1.0000   | 1.0000 |
| q09      | mr-marathi              | mr-marathi  | 1.0000   | 1.0000   | 1.0000   | 1.0000 | 1.0000   | 1.0000 |
| q10      | hi-india-ai, mr-marathi | hi-india-ai | 0.5000   | 1.0000   | 1.0000   | 1.0000 | 1.0000   | 1.0000 |
| q11      | mr-railway              | mr-railway  | 1.0000   | 1.0000   | 1.0000   | 1.0000 | 1.0000   | 1.0000 |
| q12      | hi-upi, mr-railway      | hi-upi      | 0.5000   | 1.0000   | 1.0000   | 1.0000 | 0.1786   | 0.0000 |

What actually went wrong — this is the analysis, not a cleaned-up story:

1. **`q01` — paraphrase beats lexical matching.** The question asks
   "यूपीआई की शुरुआत कब हुई और किसने की?"; the gold sentence says "…2016 में…ने
   शुरू किया". Word-BM25 puts `hi-india-ai` at rank 1 (shared function words and
   "शुरू" also occur there), so the extractive answer comes from the wrong
   document entirely (F1 0.16). char-gram gets the document right but still
   selects the wrong sentence (F1 0.07): no query token overlaps the gold
   sentence. **Takeaway: rank-1 errors and sentence-selection errors are
   different failures and this harness reports them separately.**
2. **`q06` — the code-mixed/script-mismatch case.** The query is fully
   transliterated Latin Hinglish ("Monsoon kamzor padne par kisano par kya asar
   hota hai?") against a Devanagari document. Retrieval survives on the single
   Latin token `monsoon`; the sentence scorer then picks the sentence that
   contains the English gloss "south-west monsoon" instead of the answer about
   crops (F1 0.22, EM 0).
3. **`q12` — cross-document query.** Two gold documents; only one can be rank 1,
   so `recall@1` is capped at 0.5 by construction (same for `q10`). BM25 picks
   `hi-upi` but selects sentence 1 instead of the railway-payment sentence;
   char-gram picks `mr-railway` and gets F1 0 against a `hi-upi` reference.
4. **Fusion is not free.** `hybrid` (RRF) *kept* the bm25 miss on `q01` even
   though char-gram had the right document first — averaging two rankings can
   preserve an error rather than cancel it. `hybrid` therefore does not dominate
   the single indexes on every metric; it lands between them (MRR 0.9583).
5. **`exact_match = 0.75` is exactly 9/12**: the three misses are `q01`,
   `q06`, `q12` above. EM is high because reference answers are annotated as
   *verbatim spans* from the gold document (SQuAD-style), which is the only way
   an extractive baseline can be graded by EM at all.

---

## How it works

```
data/documents/*.txt ──header──► Document ──► chunking.py (sentence-aware,
                                                overlap, Devanagari danda)
                                                  │  chunk ids  hi-upi#0 …
                                                  ▼
                              bm25.py  ── word-token index   (bm25)
                                        └─ char 3-gram index  (char-gram)
                                                  │  doc-level max-pool
                                                  ▼
                              retrieval.py ── fusion: weighted sum or RRF
                                                (+ optional dense vectors)
                                                  │  ranked doc ids
                                                  ▼
                              generation.py ── extractive (default)
                                                or LLM (optional, guarded)
                                                  ▼
                              metrics.py ── Recall@k, MRR, token-F1, EM
                                                  ▼
                              cli.py ── results/metrics.csv + report.md
```

- **`chunking.py`** — splits on `।`, `॥`, `!`, `?`, `…` and `.` (never inside a
  number like `10,372` or `2024.5`), pulls trailing closing quotes into the
  sentence, then builds chunks of ≤ 480 chars with ~120 chars of sentence-level
  overlap so facts on a boundary stay retrievable.
- **`bm25.py`** — Okapi BM25, `k1 = 1.5`, `b = 0.75`, positive RSJ idf. Two
  tokenisation strategies share one scorer:
  - *word tokens*: runs of letters/digits/**combining marks**. This detail
    matters: Python's `\w` excludes Unicode marks, so a naive regex shreds
    `मराठी` into `मर` + `ठ`. That bug existed here and is now covered by
    `test_devanagari_matras_survive`.
  - *padded char 3-grams*: robust to typos and dropped matras —
    `सोलापुर` (missing `ू`) still matches `सोलापूर`; see
    `tests/test_bm25.py::CharGramRobustnessTests`.
- **`retrieval.py`** — chunk scores are max-pooled to document scores, then
  fused per strategy with **RRF** (`k = 60`, default) or a **min-max weighted
  sum** (`--fusion weighted`). Zero-score documents are never ranked: a strategy
  that found nothing reports "no match".
- **`generation.py`** — default `extractive` picks the highest-scoring sentence
  of the top document (score = summed idf of matched query tokens ÷ √length,
  deterministic tie-break by position). The optional `llm` mode calls an
  OpenAI-compatible `/chat/completions` endpoint with a grounded prompt.
- **`metrics.py`** — the metrics above, each with a unit test and hand-computed
  expected values.

### Strategies in the report

| Name | What it fuses |
| --- | --- |
| `bm25` | word-token BM25 only |
| `char-gram` | character 3-gram BM25 only |
| `hybrid` | the two lexical indexes, RRF (or weighted) |
| `hybrid+dense` | + sentence-transformers vectors — **requires** `pip install sentence-transformers` |

### Optional pieces (all guarded)

```bash
# dense retrieval -- raises: "pip install sentence-transformers" if missing
PYTHONPATH=src python3 -m rageval evaluate --docs data/documents \
  --qrels data/eval/qa.jsonl --out results/ --strategies hybrid+dense

# LLM answers -- needs requests + SARVAM_API_KEY; no key => clear error, no traceback
PYTHONPATH=src python3 -m rageval evaluate --docs data/documents \
  --qrels data/eval/qa.jsonl --out results/ --generator llm
```

Environment variables are documented in `.env.example`
(`SARVAM_API_KEY`, `SARVAM_API_BASE`, `LLM_BASE_URL`, `LLM_MODEL`,
`RAGEVAL_DENSE_MODEL`). Nothing is auto-loaded from `.env`, no key is stored in
this repository, and the default path never needs one.

---

## Repo layout

```
indic-rag-evals/
├── README.md  LICENSE  .gitignore  .env.example  requirements.txt
├── data/
│   ├── documents/          6 short corpus files, YAML-ish header (id/title/lang)
│   │                       3 Hindi + 3 Marathi, Devanagari
│   └── eval/qa.jsonl       12 labelled queries (Hindi, Marathi, code-mixed)
├── src/rageval/            chunking · bm25 · retrieval · generation · metrics · cli
├── tests/                  test_chunking · test_bm25 · test_metrics · test_cli
└── results/                metrics.csv + report.md (regenerated, git-ignored)
```

The corpus is short, hand-written and factual (UPI, monsoons, IndiaAI/Bhashini,
Solapur, Marathi script, rail travel) — a stand-in for a real document store, not
a claim of authoritative coverage.

---

## Limits (read before quoting any number above)

- **Tiny eval set.** 12 queries, 6 documents, one author. These are regression
  numbers, not a benchmark, and the confidence interval on a 12-sample mean is
  enormous.
- **Recall@5 saturates.** With 6 documents, a 5-deep ranking almost always
  contains the answer; `recall@1` and `mrr` carry the signal here.
- **Synthetic corpus, single annotator.** The same person wrote the documents
  and the gold answers — the queries are therefore "lexically kind" to the
  corpus in a way real user traffic is not. No second annotator, no
  inter-annotator agreement.
- **Answers are verbatim spans.** This makes EM meaningful for an extractive
  baseline, but real users expect synthesised answers; multi-document questions
  (`q10`, `q12`) have a single-document reference, so a one-sentence extractive
  system cannot fully match them.
- **No LLM and no dense retrieval in the default numbers.** Every figure on this
  page is lexical + extractive. The LLM path is implemented and guarded but was
  run without network access, so no LLM results are shown — deliberately,
  because an unreproducible number is what this repo exists to avoid.
- **Script mismatch is the real gap.** Latin-transliterated queries against
  Devanagari text survive only when an acronym happens to overlap (`q06`, `q12`).
  The known fixes — cross-lingual dense embeddings, transliteration
  normalisation, or an Indic LLM reranker — are exactly what the optional
  `hybrid+dense` and `--generator llm` hooks are for.
- **Sentence selection is the weakest component.** It is pure lexical overlap;
  three of twelve answers miss because of it, not because of retrieval.
- **Facts in the corpus are brief summaries of public information**, written by
  me for this exercise; they are not sourced or verified against primary
  references.

---

## How this maps to a "Data Scientist – Evaluations" role (Sarvam)

Sarvam's product surface is Indic speech, LLMs and vision — evaluation work there
is mostly *building the measurement*, not running a leaderboard. This repo is a
small but complete instance of that:

1. **Labelled sets over vibes.** Queries in Hindi, Marathi and code-mixed
   English/Hindi, gold document ids, and gold answer spans — including
   multi-document items, so metrics behave differently from the single-hop
   happy path.
2. **Metric definitions written down.** Recall@k / MRR / token-F1 / exact match
   are implemented explicitly (not imported from a black box) and each has
   hand-computed test vectors, because an evaluation team has to be able to
   argue about what a number *means*.
3. **Indic-specific normalisation.** Danda handling, matra-preserving
   tokenisation, case/whitespace folding — plus a real bug found and fixed by
   testing the tokenizer against Devanagari (`\w` drops combining marks).
4. **Failure taxonomy, not averages.** The report is per-query: you can see
   whether a regression is a ranking miss, a sentence-selection miss, or a
   script-mismatch miss, and only then decide what to change.
5. **Reproducibility as a feature.** Stdlib-only, deterministic (the test suite
   asserts two runs produce byte-identical `metrics.csv`), no secrets, no
   network — the sort of harness that can sit in CI and gate a model or
   prompt change.
6. **Honest optional scale-out.** Dense vectors and an LLM judge/generator are
   behind guarded imports with `pip install …` errors, so the harness degrades
   gracefully on a locked-down machine instead of pretending to be complete.

---

## Tests

```
tests/test_chunking.py   danda splitting, no split inside numbers, overlap
                         coverage guarantees, corpus header parsing
tests/test_bm25.py       Devanagari tokenisation, ranking sanity, idf, typo
                         robustness of char-grams, RRF/weighted fusion
tests/test_metrics.py    Recall@k, MRR, token-F1, EM with hand-computed values
tests/test_cli.py        end-to-end CLI into temp dirs, artifact checks,
                         determinism of two runs, error exit codes
```

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

---

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 Aditya Shirsatrao.
