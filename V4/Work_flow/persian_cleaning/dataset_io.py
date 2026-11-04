"""Robust JSON reading and writing for the dataset."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

from .config import CleaningConfig


class DatasetError(RuntimeError):
    """Raised for unreadable, unparsable or unserialisable dataset files."""


_ENCODINGS = ("utf-8-sig", "utf-8", "cp1256")


@dataclass(frozen=True)
class SourceDocument:
    """One raw record, reduced to the fields the pipeline needs."""

    doc_id: str
    title: str
    url: str
    text: str


def _read_raw(path: Path) -> str:
    last_error: Exception | None = None
    for encoding in _ENCODINGS:
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError as exc:
            last_error = exc
        except OSError as exc:
            raise DatasetError(f"cannot read {path}: {exc}") from exc
    raise DatasetError(f"cannot decode {path}: {last_error}")


def _as_records(payload: Any, path: Path) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("docs", "data", "items", "records", "pages"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    raise DatasetError(f"{path}: expected a JSON array of records, got {type(payload).__name__}")


def load_records(path: str | Path) -> list[dict[str, Any]]:
    """Load a JSON array (or JSON Lines) of records from ``path``."""
    file_path = Path(path)
    if not file_path.exists():
        raise DatasetError(f"input file not found: {file_path}")

    raw = _read_raw(file_path)
    try:
        return _as_records(json.loads(raw), file_path)
    except json.JSONDecodeError:
        pass

    # JSON Lines fallback: one record per non-empty line.
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError as exc:
            raise DatasetError(f"{file_path}:{line_number} is not valid JSON: {exc}") from exc
        if isinstance(parsed, dict):
            records.append(parsed)
    if not records:
        raise DatasetError(f"{file_path} contains no usable JSON records")
    return records


def _field(record: dict[str, Any], name: str) -> str:
    value = record.get(name, "")
    return value if isinstance(value, str) else str(value)


def iter_documents(
    records: Iterable[dict[str, Any]], config: CleaningConfig
) -> Iterator[SourceDocument]:
    """Yield one :class:`SourceDocument` per record, picking the first populated text field."""
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        text = ""
        for field_name in config.text_fields:
            value = record.get(field_name)
            if isinstance(value, str) and value.strip():
                text = value
                break
        yield SourceDocument(
            doc_id=_field(record, "id") or str(index),
            title=_field(record, "title"),
            url=_field(record, "url"),
            text=text,
        )


def _write_json(payload: Any, path: str | Path, indent: int) -> None:
    file_path = Path(path)
    try:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=indent or None)
    except OSError as exc:
        raise DatasetError(f"cannot write {file_path}: {exc}") from exc
    except TypeError as exc:
        raise DatasetError(f"cannot serialise output: {exc}") from exc


def dump_sentences(sentences: list[str], path: str | Path, indent: int = 2) -> None:
    """Write the cleaned corpus as a flat JSON array of strings."""
    _write_json(sentences, path, indent)


def dump_documents(blocks: list[dict[str, Any]], path: str | Path, indent: int = 2) -> None:
    """Write per-document JSON blocks: ``[{"id", "url", "title", "sentences"}, ...]``."""
    _write_json(blocks, path, indent)
