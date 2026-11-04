"""Sentence segmentation for Persian text, with a dependency-free fallback."""

from __future__ import annotations

import re
from typing import Callable, Iterator, Sequence

from .config import CleaningConfig

_SENTENCE_END_RE = re.compile(r"(?<=[.!؟?])\s+")
_PARAGRAPH_RE = re.compile(r"\n\s*\n")


def _load_hazm() -> tuple[Callable[[str], Sequence[str]] | None, str]:
    try:
        import hazm
    except ImportError:
        return None, "hazm is not installed"

    factory = getattr(hazm, "SentenceTokenizer", None) or getattr(
        hazm, "SentenceSplitter", None
    )
    if factory is None:
        return None, "installed hazm exposes no sentence tokenizer"
    try:
        instance = factory()
    except Exception as exc:  # pragma: no cover - depends on bundled model files
        return None, f"hazm unavailable: {exc}"

    method = getattr(instance, "tokenize", None) or getattr(instance, "split", None)
    if method is None:  # pragma: no cover - defensive against API drift
        return None, "hazm tokenizer exposes neither tokenize() nor split()"
    return method, "hazm"


def _fallback_tokenize(text: str) -> list[str]:
    """Split on paragraph breaks then on sentence terminators."""
    sentences: list[str] = []
    for paragraph in _PARAGRAPH_RE.split(text):
        for candidate in _SENTENCE_END_RE.split(paragraph.strip()):
            cleaned = candidate.strip()
            if cleaned:
                sentences.append(cleaned)
    return sentences


def _guard_length(text: str, limit: int) -> Iterator[str]:
    """Keep CRF-free segmenters linear by never handing them a huge blob."""
    if len(text) <= limit:
        yield text
        return
    pieces = _PARAGRAPH_RE.split(text)
    buffer = ""
    for piece in pieces:
        if len(piece) > limit:
            if buffer:
                yield buffer
                buffer = ""
            for match in _SENTENCE_END_RE.split(piece):
                if len(buffer) + len(match) + 1 > limit and buffer:
                    yield buffer
                    buffer = match
                else:
                    buffer = f"{buffer} {match}".strip() if buffer else match
        elif len(buffer) + len(piece) + 2 > limit and buffer:
            yield buffer
            buffer = piece
        else:
            buffer = f"{buffer}\n\n{piece}" if buffer else piece
    if buffer:
        yield buffer


class SentenceSplitter:
    """Wraps the backend (hazm when present, regex otherwise)."""

    def __init__(self, config: CleaningConfig, use_hazm: bool = True) -> None:
        self.config = config
        self.backend = "regex"
        self._tokenize: Callable[[str], Sequence[str]] = _fallback_tokenize
        if use_hazm:
            loader, detail = _load_hazm()
            if loader is not None:
                self._tokenize, self.backend = loader, detail
            else:
                self.backend = f"regex ({detail})"

    def split(self, text: str) -> list[str]:
        if not text.strip():
            return []
        out: list[str] = []
        for chunk in _guard_length(text, self.config.max_chunk_chars):
            out.extend(str(sentence) for sentence in self._tokenize(chunk))
        return [sentence for sentence in (s.strip() for s in out) if sentence]
