#!/usr/bin/env python
"""Clean every dump file in a folder, one output file per input file.

Example:
    python batch_clean.py --workers 4
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from persian_cleaning import CleaningConfig, run

HERE = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = HERE.parent / "uncleaned_wikipedia_dumps"
DEFAULT_OUTPUT_DIR = HERE / "output" / "cleaned"


def clean_one(input_path: Path, output_path: Path, config: CleaningConfig) -> dict[str, Any]:
    """Worker body; runs in its own process, so it must return plain data."""
    started = time.perf_counter()
    try:
        report = run(input_path, output_path, config=config, output_format="docs")
        return {
            "input": input_path.name,
            "ok": True,
            "seconds": round(time.perf_counter() - started, 1),
            **report.as_dict(),
        }
    except Exception as exc:  # one broken file must not abort the batch
        return {
            "input": input_path.name,
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "seconds": round(time.perf_counter() - started, 1),
        }


def pending_jobs(
    input_dir: Path, output_dir: Path, pattern: str, overwrite: bool
) -> tuple[list[tuple[Path, Path]], int]:
    """Pair every input file with its output path, skipping finished work."""
    jobs: list[tuple[Path, Path]] = []
    skipped = 0
    for source in sorted(input_dir.glob(pattern)):
        if not source.is_file():
            continue
        target = output_dir / source.name
        if target.exists() and not overwrite:
            skipped += 1
            continue
        jobs.append((source, target))
    return jobs, skipped


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the cleaning pipeline over a whole folder of raw dumps.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--pattern", default="*.json", help="glob of input files inside --input-dir")
    parser.add_argument("--workers", type=int, default=min(os.cpu_count() or 2, 4))
    parser.add_argument("--min-words", type=int, default=10)
    parser.add_argument("--indent", type=int, default=2)
    parser.add_argument("--limit-files", type=int, default=None, help="clean only the first N files")
    parser.add_argument("--overwrite", action="store_true", help="re-clean files that already have output")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not args.input_dir.is_dir():
        print(f"input folder not found: {args.input_dir}", file=sys.stderr)
        return 2
    args.output_dir.mkdir(parents=True, exist_ok=True)

    try:
        config = CleaningConfig(min_words=args.min_words, indent=args.indent)
    except ValueError as exc:
        print(f"invalid configuration: {exc}", file=sys.stderr)
        return 2

    jobs, skipped = pending_jobs(args.input_dir, args.output_dir, args.pattern, args.overwrite)
    if args.limit_files is not None:
        jobs = jobs[: args.limit_files]

    if not jobs:
        print(f"nothing to do: {skipped} output file(s) already present in {args.output_dir}")
        return 0

    print(f"{len(jobs)} file(s) to clean, {skipped} already done, {args.workers} worker(s)")
    started = time.perf_counter()
    results: list[dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures: dict[Future[dict[str, Any]], str] = {
            pool.submit(clean_one, source, target, config): source.name for source, target in jobs
        }
        for done, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results.append(result)
            if result["ok"]:
                print(
                    f"[{done}/{len(futures)}] {result['input']}: "
                    f"{result['sentences_kept']}/{result['sentences_seen']} kept "
                    f"({result['keep_rate']:.1%}) in {result['seconds']}s",
                    flush=True,
                )
            else:
                print(f"[{done}/{len(futures)}] FAILED {result['input']}: {result['error']}", flush=True)

    failures = [r for r in results if not r["ok"]]
    kept = sum(r.get("sentences_kept", 0) for r in results)
    seen = sum(r.get("sentences_seen", 0) for r in results)
    elapsed = time.perf_counter() - started
    print(
        f"\ndone: {len(results) - len(failures)} ok, {len(failures)} failed, "
        f"{kept} sentences kept of {seen} in {elapsed / 60:.1f} min"
    )
    for failure in failures:
        print(f"  {failure['input']}: {failure['error']}", file=sys.stderr)
    return 1 if failures else 0


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    raise SystemExit(main())
