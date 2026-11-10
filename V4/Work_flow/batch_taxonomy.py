#!/usr/bin/env python
"""Run the error-taxonomy pass over a whole folder of stage-1 files.

Example:
    python batch_taxonomy.py --workers 4
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from persian_cleaning import taxonomy
from taxonomy_clean import process_file

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = HERE.parent / "stage1_rule_cleaned_wikipedia_dumps"
DEFAULT_OUTPUT_DIR = HERE.parent / "stage2_error_taxonomy_cleaned_wikipedia_dumps"


def clean_one(input_path: Path, output_path: Path) -> dict[str, Any]:
    """Worker body; runs in its own process, so it must return plain data."""
    try:
        return {"ok": True, **process_file(input_path, output_path)}
    except Exception as exc:  # one broken file must not abort the batch
        return {
            "ok": False,
            "input": input_path.name,
            "error": f"{type(exc).__name__}: {exc}",
            "sentences": 0,
            "changed": 0,
            "categories": {},
            "forms": {},
        }


def pending_jobs(
    input_dir: Path, output_dir: Path, pattern: str, overwrite: bool, dry_run: bool
) -> tuple[list[tuple[Path, Path]], int]:
    """Pair every input file with its output path, skipping finished work."""
    jobs: list[tuple[Path, Path]] = []
    skipped = 0
    for source in sorted(input_dir.glob(pattern)):
        if not source.is_file():
            continue
        target = output_dir / source.name
        if target.exists() and not overwrite and not dry_run:
            skipped += 1
            continue
        jobs.append((source, target))
    return jobs, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Apply the Persian error taxonomy to every stage-1 file in a folder.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="*.json", help="glob of input files inside --input-dir")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit-files", type=int, default=None, help="clean only the first N files")
    parser.add_argument("--overwrite", action="store_true", help="re-clean files that already have output")
    parser.add_argument("--dry-run", action="store_true", help="count fixes but write nothing")
    parser.add_argument("--verbose", action="store_true", help="list the most frequent forms")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"input folder not found: {args.input_dir}", file=sys.stderr)
        return 2
    if not args.dry_run:
        args.output_dir.mkdir(parents=True, exist_ok=True)

    jobs, skipped = pending_jobs(
        args.input_dir, args.output_dir, args.pattern, args.overwrite, args.dry_run
    )
    if args.limit_files is not None:
        jobs = jobs[: args.limit_files]

    if not jobs:
        print(f"nothing to do: {skipped} output file(s) already present in {args.output_dir}")
        return 0

    print(
        f"{len(jobs)} file(s) to clean, {skipped} already done, "
        f"{args.workers} worker(s){' [dry run]' if args.dry_run else ''}",
        flush=True,
    )
    started = time.perf_counter()
    total = taxonomy.TaxonomyStats()
    records = 0
    failures: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures: dict[Future[dict[str, Any]], str] = {
            pool.submit(clean_one, source, target): source.name for source, target in jobs
        }
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            if result["ok"]:
                records += result["records"]
                total.merge(taxonomy.TaxonomyStats.from_dict(result))
                print(
                    f"[{done}/{len(futures)}] {result['input']}: "
                    f"{result['changed']}/{result['sentences']} sentences touched, "
                    f"{result['records']} record(s) in {result['seconds']}s",
                    flush=True,
                )
            else:
                failures.append(result)
                print(f"[{done}/{len(futures)}] FAILED {result['input']}: {result['error']}", flush=True)

    elapsed = time.perf_counter() - started
    print(
        f"\ndone: {len(jobs) - len(failures)} ok, {len(failures)} failed, "
        f"{records} record(s) in {elapsed / 60:.1f} min"
    )
    print(total.render(verbose=args.verbose))
    if not args.dry_run:
        print(f"output folder   : {args.output_dir}")
    for failure in failures:
        print(f"  {failure['input']}: {failure['error']}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
