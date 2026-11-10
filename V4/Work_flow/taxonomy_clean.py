#!/usr/bin/env python
"""Apply the mesale_khata error taxonomy to one stage-1 document file.

Example:
    python taxonomy_clean.py --input ../stage1_rule_cleaned_wikipedia_dumps/part_0001.json
Use batch_taxonomy.py when the whole folder must be processed.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Sequence

from persian_cleaning import taxonomy
from persian_cleaning.dataset_io import DatasetError, load_records

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT = HERE.parent / "stage1_rule_cleaned_wikipedia_dumps" / "part_0001.json"
DEFAULT_OUTPUT_DIR = HERE.parent / "stage2_error_taxonomy_cleaned_wikipedia_dumps"
INDENT = 2


def clean_record(record: dict[str, Any], stats: taxonomy.TaxonomyStats) -> dict[str, Any]:
    """Return the record with every taxonomy rule applied, keys and order kept."""
    title = record.get("title", "")
    sentences = record.get("sentences", [])
    return {
        "id": record.get("id", ""),
        "url": record.get("url", ""),
        "title": (
            taxonomy.apply(title, stats, is_title=True) if isinstance(title, str) else title
        ),
        "sentences": [
            taxonomy.apply(sentence, stats) if isinstance(sentence, str) else sentence
            for sentence in (sentences if isinstance(sentences, list) else [])
        ],
    }


def write_json(payload: Any, path: Path) -> None:
    """Write the same JSON shape the stage-1 pipeline produced, atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8", newline="\n") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=INDENT)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except OSError as exc:
        tmp.unlink(missing_ok=True)
        raise DatasetError(f"cannot write {path}: {exc}") from exc
    except TypeError as exc:
        tmp.unlink(missing_ok=True)
        raise DatasetError(f"cannot serialise {path}: {exc}") from exc


def process_file(input_path: Path, output_path: Path | None = None) -> dict[str, Any]:
    """Clean one file and return picklable per-category statistics."""
    started = time.perf_counter()
    stats = taxonomy.TaxonomyStats()
    blocks = [clean_record(r, stats) for r in load_records(input_path) if isinstance(r, dict)]
    result: dict[str, Any] = {
        "input": input_path.name,
        "records": len(blocks),
        "seconds": round(time.perf_counter() - started, 1),
        **stats.as_dict(),
    }
    if output_path is not None:
        write_json(blocks, output_path)
        result["output"] = str(output_path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Repair the Persian error taxonomy in one cleaned dump file.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="stage-1 JSON file")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="output JSON; defaults to <stage2 folder>/<input file name>",
    )
    parser.add_argument("--dry-run", action="store_true", help="count fixes but write nothing")
    parser.add_argument("--verbose", action="store_true", help="list the most frequent forms")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output_path = None if args.dry_run else (args.output or DEFAULT_OUTPUT_DIR / args.input.name)

    try:
        result = process_file(args.input, output_path)
    except DatasetError as exc:
        print(f"dataset error: {exc}", file=sys.stderr)
        return 1

    stats = taxonomy.TaxonomyStats.from_dict(result)
    print(f"{result['records']} record(s) from {args.input} in {result['seconds']}s")
    print(stats.render(verbose=args.verbose))
    print(f"output written  : {output_path}" if output_path else "output written  : (dry run)")
    return 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):  # Persian output on Windows consoles
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
