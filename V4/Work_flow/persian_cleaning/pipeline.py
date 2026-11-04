"""End-to-end orchestration: load -> normalise -> split -> filter -> export."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .config import CleaningConfig, RejectionStats
from .dataset_io import (
    SourceDocument,
    dump_documents,
    dump_sentences,
    iter_documents,
    load_records,
)
from .filters import reject_reason
from .normalize import normalize_text
from .sentence_splitter import SentenceSplitter

_DUP_KEY_RE = re.compile(r"\s+")
OUTPUT_FORMATS = ("docs", "flat")


@dataclass
class CleanedDocument:
    """The surviving sentences of one source record."""

    doc_id: str
    title: str
    url: str
    sentences: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.doc_id,
            "url": self.url,
            "title": self.title,
            "sentences": self.sentences,
        }


@dataclass
class CleaningReport:
    """Summary of one run."""

    records_read: int = 0
    records_without_text: int = 0
    documents_kept: int = 0
    documents_dropped_empty: int = 0
    sentences_seen: int = 0
    sentences_kept: int = 0
    duplicates_removed: int = 0
    backend: str = ""
    output_format: str = "docs"
    rejections: RejectionStats = field(default_factory=RejectionStats)

    @property
    def keep_rate(self) -> float:
        return self.sentences_kept / self.sentences_seen if self.sentences_seen else 0.0

    def as_dict(self) -> dict[str, object]:
        return {
            "records_read": self.records_read,
            "records_without_text": self.records_without_text,
            "documents_kept": self.documents_kept,
            "documents_dropped_empty": self.documents_dropped_empty,
            "sentences_seen": self.sentences_seen,
            "sentences_kept": self.sentences_kept,
            "duplicates_removed": self.duplicates_removed,
            "keep_rate": round(self.keep_rate, 4),
            "splitter_backend": self.backend,
            "output_format": self.output_format,
            "rejections_by_rule": dict(self.rejections.sorted_items()),
        }

    def render(self) -> str:
        lines = [
            f"splitter backend     : {self.backend}",
            f"output format        : {self.output_format}",
            f"records read         : {self.records_read}",
            f"records without text : {self.records_without_text}",
            f"documents kept       : {self.documents_kept}",
            f"documents emptied    : {self.documents_dropped_empty}",
            f"sentences produced   : {self.sentences_seen}",
            f"duplicates removed   : {self.duplicates_removed}",
            f"sentences kept       : {self.sentences_kept} ({self.keep_rate:.1%})",
            "rejected by rule     :",
        ]
        total = self.rejections.total or 1
        for reason, count in self.rejections.sorted_items():
            lines.append(f"  {reason:<20} {count:>9}  ({count / total:.1%})")
        return "\n".join(lines)


def _dedupe_key(sentence: str) -> str:
    return _DUP_KEY_RE.sub(" ", sentence).strip().lower()


def clean_documents(
    documents: Iterable[SourceDocument],
    config: CleaningConfig,
    splitter: SentenceSplitter,
    deduplicate: bool = True,
) -> tuple[list[CleanedDocument], CleaningReport]:
    """Clean each source record separately, keeping its sentences grouped."""
    report = CleaningReport(backend=splitter.backend)
    cleaned: list[CleanedDocument] = []

    for document in documents:
        report.records_read += 1
        text = normalize_text(document.text)
        if not text:
            report.records_without_text += 1
            continue

        try:
            sentences = splitter.split(text)
        except Exception as exc:  # one bad document must not kill the run
            report.records_without_text += 1
            report.rejections.add(f"split_error:{type(exc).__name__}")
            continue

        report.sentences_seen += len(sentences)
        seen: set[str] = set()
        kept: list[str] = []
        for sentence in sentences:
            reason = reject_reason(sentence, config)
            if reason is not None:
                report.rejections.add(reason)
                continue
            if deduplicate:
                key = _dedupe_key(sentence)
                if key in seen:
                    report.duplicates_removed += 1
                    continue
                seen.add(key)
            kept.append(sentence)

        if not kept:
            report.documents_dropped_empty += 1
            continue

        report.documents_kept += 1
        report.sentences_kept += len(kept)
        cleaned.append(
            CleanedDocument(
                doc_id=document.doc_id,
                title=document.title,
                url=document.url,
                sentences=kept,
            )
        )

    return cleaned, report


def run(
    input_path: str | Path,
    output_path: str | Path,
    config: CleaningConfig | None = None,
    use_hazm: bool = True,
    deduplicate: bool = True,
    limit: int | None = None,
    output_format: str = "docs",
) -> CleaningReport:
    """Clean ``input_path`` into ``output_path`` and return the run report.

    ``output_format`` selects the shape of the artefact: ``docs`` groups the
    sentences per source record, ``flat`` emits one array of all sentences.
    """
    if output_format not in OUTPUT_FORMATS:
        raise ValueError(f"output_format must be one of {OUTPUT_FORMATS}, got {output_format!r}")

    config = config or CleaningConfig()

    records = load_records(input_path)
    if limit is not None:
        records = records[:limit]

    splitter = SentenceSplitter(config, use_hazm=use_hazm)
    documents, report = clean_documents(
        iter_documents(records, config), config, splitter, deduplicate=deduplicate
    )
    report.output_format = output_format

    if output_format == "flat":
        dump_sentences(
            [sentence for document in documents for sentence in document.sentences],
            output_path,
            indent=config.indent,
        )
    else:
        dump_documents(
            [document.to_json() for document in documents], output_path, indent=config.indent
        )
    return report
