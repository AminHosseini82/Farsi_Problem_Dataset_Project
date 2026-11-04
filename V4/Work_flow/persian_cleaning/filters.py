"""Rule-based rejection logic; each rule answers 'should this sentence be dropped?'.

Rules are evaluated in declaration order and the first hit wins, so the
rejection counters stay mutually exclusive.
"""

from __future__ import annotations

from typing import Callable

from .config import (
    CITATION_RE,
    CONTROL_RE,
    ELLIPSIS_BRACKET_RE,
    HTML_TAG_RE,
    LATIN_APOSTROPHE_RE,
    NON_PROSE_RE,
    REPLACEMENT_RE,
    STRAY_QUOTE_RE,
    UNBALANCED_WIKI_RE,
    URL_RE,
    WIKI_LINK_RE,
    WIKI_TEMPLATE_RE,
    CleaningConfig,
    bracket_ratio,
    has_unbalanced_brackets,
    latin_ratio,
    markup_ratio,
    symbol_ratio,
    word_count,
)

Predicate = Callable[[str, CleaningConfig], bool]


def _no_letters(sentence: str, config: CleaningConfig) -> bool:
    return word_count(sentence) == 0


def _too_short(sentence: str, config: CleaningConfig) -> bool:
    return word_count(sentence) < config.min_words


def _too_long(sentence: str, config: CleaningConfig) -> bool:
    """Unsplit category/navigation dumps masquerading as one sentence."""
    return config.max_words > 0 and word_count(sentence) > config.max_words


def _has_html(sentence: str, config: CleaningConfig) -> bool:
    return HTML_TAG_RE.search(sentence) is not None


def _has_wiki_markup(sentence: str, config: CleaningConfig) -> bool:
    if WIKI_LINK_RE.search(sentence) or WIKI_TEMPLATE_RE.search(sentence):
        return True
    if UNBALANCED_WIKI_RE.search(sentence):  # severed [[link]] residue
        return True
    return config.max_markup_ratio > 0 and markup_ratio(sentence) > config.max_markup_ratio


def _contains_url(sentence: str, config: CleaningConfig) -> bool:
    return URL_RE.search(sentence) is not None


def _mostly_english(sentence: str, config: CleaningConfig) -> bool:
    return config.max_english_ratio > 0 and latin_ratio(sentence) > config.max_english_ratio


def _has_invalid_symbols(sentence: str, config: CleaningConfig) -> bool:
    if CONTROL_RE.search(sentence) or REPLACEMENT_RE.search(sentence):
        return True
    if NON_PROSE_RE.search(sentence):
        return True
    return config.max_symbol_ratio > 0 and symbol_ratio(sentence) > config.max_symbol_ratio


def _has_direct_quote(sentence: str, config: CleaningConfig) -> bool:
    """Any quote delimiter marks quoted material, balanced or severed.

    Apostrophes inside Latin words (don't, Cunningham's) are neutralised first
    so they do not count as quotation.
    """
    if not config.drop_quoted_sentences:
        return False
    return STRAY_QUOTE_RE.search(LATIN_APOSTROPHE_RE.sub("", sentence)) is not None


def _has_citation(sentence: str, config: CleaningConfig) -> bool:
    return config.drop_citations and CITATION_RE.search(sentence) is not None


def _bracket_heavy(sentence: str, config: CleaningConfig) -> bool:
    return (
        config.max_bracket_ratio > 0
        and bracket_ratio(sentence) > config.max_bracket_ratio
    )


def _bracket_residue(sentence: str, config: CleaningConfig) -> bool:
    """Elision markers and brackets severed by the sentence boundary."""
    return bool(ELLIPSIS_BRACKET_RE.search(sentence)) or has_unbalanced_brackets(sentence)


RULES: tuple[tuple[str, Predicate], ...] = (
    ("no_persian_words", _no_letters),
    ("too_short", _too_short),
    ("too_long", _too_long),
    ("html_markup", _has_html),
    ("wiki_markup", _has_wiki_markup),
    ("contains_url", _contains_url),
    ("mostly_english", _mostly_english),
    ("invalid_symbols", _has_invalid_symbols),
    ("direct_quote", _has_direct_quote),
    ("bracketed_citation", _has_citation),
    ("bracket_heavy", _bracket_heavy),
    ("bracket_residue", _bracket_residue),
)


def reject_reason(sentence: str, config: CleaningConfig) -> str | None:
    """Return the name of the first failing rule, or ``None`` to keep it."""
    for name, predicate in RULES:
        if predicate(sentence, config):
            return name
    return None
