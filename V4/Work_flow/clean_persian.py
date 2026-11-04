#!/usr/bin/env python
"""CLI entry point for the Persian dataset cleaning pipeline.

Example:
    python clean_persian.py --input ../uncleaned_wikipedia_dumps/part_0001.json
Use batch_clean.py instead when the whole folder must be processed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence

from persian_cleaning import OUTPUT_FORMATS, CleaningConfig, DatasetError, run

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE.parent / "uncleaned_wikipedia_dumps"
DEFAULT_OUTPUTS = {
    "docs": HERE / "output" / "cleaned_documents.json",
    "flat": HERE / "output" / "cleaned_sentences.json",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tokenise, filter and export clean Persian sentences from one dump file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, required=True, help=f"raw JSON dataset, e.g. {RAW_DIR / 'part_0001.json'}")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output JSON; defaults to output/cleaned_documents.json (docs) or cleaned_sentences.json (flat)",
    )
    parser.add_argument(
        "--format",
        choices=OUTPUT_FORMATS,
        default="docs",
        help="docs = one block per source record; flat = one array of all sentences",
    )
    parser.add_argument("--limit", type=int, default=None, help="process only the first N records")
    parser.add_argument("--min-words", type=int, default=10, help="drop sentences with fewer words than this")
    parser.add_argument("--max-words", type=int, default=150, help="drop unsplit run-on dumps (0 = disabled)")
    parser.add_argument("--max-english-ratio", type=float, default=0.5, help="max share of Latin letters")
    parser.add_argument("--max-symbol-ratio", type=float, default=0.05, help="max share of non-prose symbols")
    parser.add_argument("--max-bracket-ratio", type=float, default=0.20, help="max share of bracketed characters")
    parser.add_argument("--no-hazm", action="store_true", help="use the regex splitter instead of hazm")
    parser.add_argument("--no-dedupe", action="store_true", help="keep duplicate sentences")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent of the output file (0 = compact)")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    try:
        config = CleaningConfig(
            min_words=args.min_words,
            max_words=args.max_words,
            max_english_ratio=args.max_english_ratio,
            max_symbol_ratio=args.max_symbol_ratio,
            max_bracket_ratio=args.max_bracket_ratio,
            indent=args.indent,
        )
    except ValueError as exc:
        print(f"invalid configuration: {exc}", file=sys.stderr)
        return 2

    output_path = args.output or DEFAULT_OUTPUTS[args.format]

    try:
        report = run(
            input_path=args.input,
            output_path=output_path,
            config=config,
            use_hazm=not args.no_hazm,
            deduplicate=not args.no_dedupe,
            limit=args.limit,
            output_format=args.format,
        )
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 1

    print(report.render())
    print(f"output written  : {output_path}")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):  # Persian output on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
