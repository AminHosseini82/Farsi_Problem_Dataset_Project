"""Character-level normalisation applied before sentence splitting."""

from __future__ import annotations

import re
import unicodedata

from .config import CONTROL_RE, NON_PROSE_RE, REPLACEMENT_RE

# Arabic variants that Persian NLP models expect folded to their Persian forms.
_CHAR_FOLD: dict[str, str] = {
    "\u064a": "\u06cc",  # ي -> ی
    "\u0643": "\u06a9",  # ك -> ک
    "\u0624": "\u0648",  # ؤ
    "\u0625": "\u0627",  # إ
    "\u0623": "\u0627",  # أ
    "\u0649": "\u06cc",  # ى
    "\u0652": "",  # سكون
    "\u0651": "",  # تشدید
    "\u0640": "",  # tatweel
    "\u200b": "",  # zero-width space (ZWNJ \u200c is kept intentionally)
    "\u200d": "",  # zero-width joiner
    "\ufeff": "",  # BOM
}

_WHITESPACE_RE = re.compile(r"[ \t\u00a0\u2000-\u200a\u202f\u205f]+")
_NEWLINES_RE = re.compile(r"\n{3,}")
_TRAILING_SPACES_RE = re.compile(r"[ \t]*\n[ \t]*")

#: Wiki list markers, headings, table pipes and bold/italic quotes at line start.
_LINE_NOISE_RE = re.compile(r"^[ \t]*(?:[#*:;]+|[=]{2,}[^=\n]*[=]{2,}|__NOTOC__|__NOEDITSECTION__)", re.MULTILINE)
#: Wiki bold/italic marks only; single apostrophes are kept so the
#: ``'quoted'`` rule can still see them.
_QUOTE_MARKS_RE = re.compile(r"'{2,3}")

#: ``[[Link|shown text]]`` -> ``shown text``, ``[[Link]]`` -> ``Link``.
_WIKI_LINK_RE = re.compile(r"\[\[(?:[^\[\]|]*\|)?([^\[\]|]+)\]\]")
#: ``[http://target shown text]`` -> ``shown text``; the target itself is noise.
_WIKI_EXT_LINK_RE = re.compile(r"\[(?:https?|ftp|mailto)://[^\]\s]*[\s|]([^\]\n]+)\]")
_WIKI_ADDRESS_ONLY_RE = re.compile(r"\[(?:https?|ftp|mailto):[^\]\n]*\]")
#: ``{{ template ... }}`` carries no prose; innermost first so nesting resolves.
_WIKI_TEMPLATE_RE = re.compile(r"\{\{[^{}]*\}\}")
_EMPTY_PARENS_RE = re.compile(r"[（(]\s*[)）]")


def _strip_wiki_templates(text: str) -> str:
    for _ in range(5):
        stripped = _WIKI_TEMPLATE_RE.sub("", text)
        if stripped == text:
            break
        text = stripped
    return text


def normalize_text(text: str) -> str:
    """Fold letters, drop invisible/control noise and collapse whitespace.

    Paragraph structure is preserved (``\\n\\n``) because it is a strong
    sentence-boundary signal for the splitter.
    """
    if not text:
        return ""

    folded = text.translate({ord(k): v for k, v in _CHAR_FOLD.items()})
    folded = unicodedata.normalize("NFC", folded)

    folded = CONTROL_RE.sub(" ", folded)
    folded = REPLACEMENT_RE.sub(" ", folded)
    folded = NON_PROSE_RE.sub(" ", folded)

    folded = folded.replace("\r\n", "\n").replace("\r", "\n")
    folded = _LINE_NOISE_RE.sub("", folded)
    folded = _strip_wiki_templates(folded)
    folded = _WIKI_EXT_LINK_RE.sub(r"\1", folded)
    folded = _WIKI_ADDRESS_ONLY_RE.sub("", folded)
    folded = _WIKI_LINK_RE.sub(r"\1", folded)
    folded = _QUOTE_MARKS_RE.sub("", folded)
    folded = _EMPTY_PARENS_RE.sub("", folded)
    folded = _TRAILING_SPACES_RE.sub("\n", folded)
    folded = _WHITESPACE_RE.sub(" ", folded)
    folded = _NEWLINES_RE.sub("\n\n", folded)

    return folded.strip()
