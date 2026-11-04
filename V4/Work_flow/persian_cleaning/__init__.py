"""Rule-based Persian text cleaning pipeline for NLP fine-tuning corpora."""

from .config import CleaningConfig, RejectionStats
from .dataset_io import (
    DatasetError,
    SourceDocument,
    dump_documents,
    dump_sentences,
    iter_documents,
    load_records,
)
from .filters import RULES, reject_reason
from .normalize import normalize_text
from .pipeline import (
    OUTPUT_FORMATS,
    CleanedDocument,
    CleaningReport,
    clean_documents,
    run,
)
from .sentence_splitter import SentenceSplitter

__all__ = [
    "OUTPUT_FORMATS",
    "CleanedDocument",
    "CleaningConfig",
    "CleaningReport",
    "DatasetError",
    "RejectionStats",
    "RULES",
    "SentenceSplitter",
    "SourceDocument",
    "clean_documents",
    "dump_documents",
    "dump_sentences",
    "iter_documents",
    "load_records",
    "normalize_text",
    "reject_reason",
    "run",
]
