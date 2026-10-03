"""rageval — offline retrieval evaluation harness for Indic (Hindi/Marathi) RAG.

Public surface re-exported for convenience:

>>> from rageval import Retriever, load_documents, recall_at_k
"""

from .bm25 import BM25Index, tokenize, tokenize_char_grams
from .chunking import Chunk, Document, chunk_text, load_documents, split_sentences
from .generation import extractive_answer, generate_answer, llm_answer
from .metrics import (
    StrategyMetrics,
    exact_match,
    mrr,
    normalize,
    recall_at_k,
    reciprocal_rank,
    token_f1,
)
from .retrieval import (
    MissingDependencyError,
    Query,
    Retriever,
    ScoredDoc,
    Strategy,
    fuse_rrf,
    fuse_weighted,
    load_qrels,
)

__version__ = "0.1.0"

__all__ = [
    "BM25Index",
    "Chunk",
    "Document",
    "MissingDependencyError",
    "Query",
    "Retriever",
    "ScoredDoc",
    "Strategy",
    "StrategyMetrics",
    "__version__",
    "chunk_text",
    "exact_match",
    "extractive_answer",
    "fuse_rrf",
    "fuse_weighted",
    "generate_answer",
    "load_documents",
    "load_qrels",
    "llm_answer",
    "mrr",
    "normalize",
    "recall_at_k",
    "reciprocal_rank",
    "split_sentences",
    "token_f1",
    "tokenize",
    "tokenize_char_grams",
]
