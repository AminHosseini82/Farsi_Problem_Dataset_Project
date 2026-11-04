"""Tunable thresholds and compiled rules for the Persian cleaning pipeline."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Final, Pattern

#: Persian + Arabic letters + presentation forms.
PERSIAN_ARABIC_BLOCK: Final[str] = "\u0600-\u06ff\ufb50-\ufeff"

LETTER_RE: Final[Pattern[str]] = re.compile(rf"[{PERSIAN_ARABIC_BLOCK}A-Za-z]")
LATIN_RE: Final[Pattern[str]] = re.compile(r"[A-Za-z]")
DIGIT_CLASS: Final[str] = "0-9\u0660-\u0669\u06f0-\u06f9"
DIGIT_RE: Final[Pattern[str]] = re.compile(rf"[{DIGIT_CLASS}]")

#: A token counts as a word only if it carries at least one letter, so bare
#: numbers, dots or stray symbols never inflate the count. ZWNJ-joined
#: compounds (e.g. می‌شود) are kept as a single word.
_WORD_SHAPE_RE: Final[Pattern[str]] = re.compile(rf"[{PERSIAN_ARABIC_BLOCK}A-Za-z]")


def word_count(sentence: str) -> int:
    return sum(1 for token in sentence.split() if _WORD_SHAPE_RE.search(token))


HTML_TAG_RE: Final[Pattern[str]] = re.compile(r"</?\s*[a-zA-Z][^<>]{0,60}>")
WIKI_LINK_RE: Final[Pattern[str]] = re.compile(r"\[\[[^\[\]]{1,300}\]\]")
WIKI_TEMPLATE_RE: Final[Pattern[str]] = re.compile(r"\{\{[^{}]{1,300}\}\}")
CITATION_RE: Final[Pattern[str]] = re.compile(
    rf"\[\s*[{DIGIT_CLASS}]+(?:\s*[,،\-]\s*[{DIGIT_CLASS}]+)*\s*\]"
)
BRACKETED_RE: Final[Pattern[str]] = re.compile(r"\[[^\[\]]{1,200}\]")
#: ``[...]`` / ``[…]`` elision markers left behind by truncated quotations.
ELLIPSIS_BRACKET_RE: Final[Pattern[str]] = re.compile(r"\[\s*(?:\.{1,3}|…+)\s*\]")


def has_unbalanced_brackets(sentence: str) -> bool:
    """True when a ``[`` or ``]`` was severed by the sentence boundary."""
    return sentence.count("[") != sentence.count("]")

#: Any quote delimiter marks quoted material, whether balanced or severed.
#: Apostrophes inside Latin words are neutralised first so they do not count.
LATIN_APOSTROPHE_RE: Final[Pattern[str]] = re.compile(r"(?<=[A-Za-z])['‘](?=[A-Za-z])")
STRAY_QUOTE_RE: Final[Pattern[str]] = re.compile(r"[«»“”\"'‘’]")

#: Matches intact URLs and the broken "http: //www." spacing seen in some dumps.
URL_RE: Final[Pattern[str]] = re.compile(r"(?:https?|ftp)\s*:\s*/\s*/|www\.", re.IGNORECASE)
UNBALANCED_WIKI_RE: Final[Pattern[str]] = re.compile(r"\[\[|\]\]")

CONTROL_RE: Final[Pattern[str]] = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
REPLACEMENT_RE: Final[Pattern[str]] = re.compile(r"[\ufffd\ufffe\uffff\u2028\u2029]")
NON_PROSE_RE: Final[Pattern[str]] = re.compile(
    r"[\U0001f000-\U0001ffff\u202a-\u202e\u2066-\u2069\ue000-\uf8ff]"
)

#: Punctuation that is legitimate inside prose (includes the Persian/Arabic
#: decimal ٫ and thousands ٬ separators used with Persian digits).
PROSE_PUNCTUATION: Final[frozenset[str]] = frozenset(
    ".،،؛:؛!؟?..,«»()[]’'\"“”/%×±°+-–—٫٬"
)

#: Wiki markup residue; never legitimate inside a sentence.
MARKUP_CHARS: Final[frozenset[str]] = frozenset("{}<>|*=#_~^@&")


def latin_ratio(sentence: str) -> float:
    """Share of letter characters that are Latin."""
    letters = [c for c in sentence if LETTER_RE.fullmatch(c)]
    if not letters:
        return 1.0
    return sum(1 for c in letters if c.isascii() and c.isalpha()) / len(letters)


def symbol_ratio(sentence: str) -> float:
    """Share of punctuation/symbol characters outside the normal prose set.

    Only Unicode categories P (punctuation) and S (symbols) are scored, so
    letters, digits, ZWNJ (U+200C) and combining diacritics never count as
    noise.
    """
    scored = [c for c in sentence if unicodedata.category(c)[0] in ("P", "S")]
    if not scored:
        return 0.0
    return sum(1 for c in scored if c not in PROSE_PUNCTUATION) / len(scored)


def markup_ratio(sentence: str) -> float:
    """Share of non-space characters that are wiki/markup residue."""
    chars = [c for c in sentence if not c.isspace()]
    if not chars:
        return 1.0
    return sum(1 for c in chars if c in MARKUP_CHARS) / len(chars)


def bracket_ratio(sentence: str) -> float:
    """Share of characters that live inside ``[...]`` spans."""
    if not sentence:
        return 0.0
    outside = len(BRACKETED_RE.sub("", sentence))
    return (len(sentence) - outside) / len(sentence)


@dataclass(frozen=True)
class CleaningConfig:
    """Thresholds for every rule; ``0`` or ``0.0`` disables that rule."""

    min_words: int = 10
    max_words: int = 150
    max_english_ratio: float = 0.5
    max_symbol_ratio: float = 0.05
    max_markup_ratio: float = 0.03
    max_bracket_ratio: float = 0.20
    drop_quoted_sentences: bool = True
    drop_citations: bool = True
    max_chunk_chars: int = 20_000
    text_fields: tuple[str, ...] = ("text", "body", "content", "page")
    indent: int = 2

    def __post_init__(self) -> None:
        if self.min_words < 1:
            raise ValueError("min_words must be >= 1")
        if self.max_words and self.max_words <= self.min_words:
            raise ValueError("max_words must exceed min_words")
        for name in (
            "max_english_ratio",
            "max_symbol_ratio",
            "max_markup_ratio",
            "max_bracket_ratio",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")


@dataclass
class RejectionStats:
    """Per-rule rejection counters."""

    counts: dict[str, int] = field(default_factory=dict)

    def add(self, reason: str) -> None:
        self.counts[reason] = self.counts.get(reason, 0) + 1

    @property
    def total(self) -> int:
        return sum(self.counts.values())

    def sorted_items(self) -> list[tuple[str, int]]:
        return sorted(self.counts.items(), key=lambda item: (-item[1], item[0]))
