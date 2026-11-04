#!/usr/bin/env python3
"""Rewrite the rule-cleaned Persian corpus into academic style with an LLM.

Reads every ``*.json`` file of the stage-1 output (arrays of
``{"id", "url", "title", "sentences"}`` blocks, or plain arrays of sentences),
sends the sentences to a chat-completions model in batches, and writes a file
with the identical name and shape to the output directory.

    .venv\\Scripts\\python.exe Work_flow\\academic_rewrite.py --limit-files 2
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from openai import (
    APIConnectionError,
    APIError,
    APITimeoutError,
    AsyncOpenAI,
    AuthenticationError,
    BadRequestError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)

HERE = Path(__file__).resolve().parent
PROJECT_ROOT = HERE.parent
DEFAULT_INPUT_DIR = HERE / "output" / "cleaned"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "academic_cleaned"
SENTENCES_KEY = "cleaned_sentences"
#: Used only by --dry-run to turn the request count into a wall-clock estimate.
AVG_REQUEST_SECONDS = 3.0

SYSTEM_PROMPT = """
You are a Senior Academic Copy Editor for the Persian language. I will give you a JSON array of Persian sentences. Rewrite each sentence to meet the strictest standards of Iranian Academic and ISI papers, and return ONLY a JSON array of the exact same length under the key 'cleaned_sentences'.

Rules:
1. Tone: Strictly formal, objective, and impersonal. Remove all first-person pronouns (I, we). Convert conversational words to formal equivalents.
2. Orthography: Correct spelling. Fix 'Ye' (ی) and 'Hamza' (ء) combinations (e.g., خانهٔ instead of خانه‌ی). Remove incorrect Tanween.
3. Spacing: Strictly enforce Zero-Width Non-Joiner (نیم‌فاصله) for prefixes (می, نمی‌), plural suffixes (ها, یان), and compound words. Fix spacing around punctuation marks.
4. Semantic & Stylistic: Remove semantic redundancies (حشو). Do not use foreign words if a standard Persian equivalent exists.
5. Fidelity: Do NOT alter facts, dates, proper nouns, or lengths. Do not merge or split sentences.
"""

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_HAS_LETTER_RE = re.compile(r"[\u0600-\u06ff\ufb50-\ufeffA-Za-z]")


class BatchError(RuntimeError):
    """The model could not produce a usable rewrite for a batch."""


class InvalidResponse(BatchError):
    """The reply was not JSON, or not an array of the expected length."""


@dataclass(frozen=True)
class Settings:
    model: str
    batch_sentences: int
    batch_chars: int
    file_concurrency: int
    max_inflight: int
    request_timeout: float
    max_retries: int
    backoff: float
    temperature: float
    on_batch_failure: str
    indent: int


@dataclass
class RunStats:
    requests: int = 0
    retries: int = 0
    suspicious: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)

    @property
    def failed(self) -> list[dict[str, Any]]:
        return [item for item in self.results if not item["ok"]]


@dataclass
class CorpusFile:
    """One input file, flattened so that a batch is always a contiguous slice."""

    documents: list[dict[str, Any]]
    sentences: list[str]
    flat: bool

    @classmethod
    def load(cls, path: Path) -> "CorpusFile":
        data = read_json(path)
        if not isinstance(data, list) or not data:
            raise ValueError(f"{path.name}: expected a non-empty JSON array")
        if all(isinstance(item, str) for item in data):
            return cls([], list(data), flat=True)
        documents: list[dict[str, Any]] = []
        sentences: list[str] = []
        for item in data:
            if not isinstance(item, dict) or not isinstance(item.get("sentences"), list):
                raise ValueError(f"{path.name}: items must be strings or carry a 'sentences' array")
            block = dict(item)
            start = len(sentences)
            sentences.extend(str(sentence) for sentence in block["sentences"])
            block["_range"] = (start, len(sentences))
            documents.append(block)
        return cls(documents, sentences, flat=False)

    def rebuilt(self) -> list[Any]:
        if self.flat:
            return self.sentences
        blocks: list[Any] = []
        for block in self.documents:
            start, stop = block.pop("_range")
            block["sentences"] = self.sentences[start:stop]
            blocks.append(block)
        return blocks


def read_json(path: Path) -> Any:
    for encoding in ("utf-8-sig", "utf-8", "cp1256"):
        try:
            with path.open(encoding=encoding) as handle:
                return json.load(handle)
        except UnicodeDecodeError:
            continue
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path.name}: invalid JSON ({exc})") from exc
    raise ValueError(f"{path.name}: not decodable as UTF-8 or CP1256")


def write_json(path: Path, payload: Any, indent: int) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=indent)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def extract_json(raw: str) -> Any:
    fenced = _JSON_FENCE_RE.search(raw)
    text = fenced.group(1) if fenced else raw.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        start, stop = text.find("{"), text.rfind("}")
        if start == -1 or stop <= start:
            raise InvalidResponse("reply contained no JSON object")
        return json.loads(text[start : stop + 1])


def parse_reply(raw: str, expected: int) -> list[str]:
    data = extract_json(raw)
    if isinstance(data, dict):
        items = data.get(SENTENCES_KEY)
        if not isinstance(items, list):
            items = next((value for value in data.values() if isinstance(value, list)), None)
    else:
        items = data
    if not isinstance(items, list) or len(items) != expected:
        got = len(items) if isinstance(items, list) else type(items).__name__
        raise InvalidResponse(f"expected {expected} entries under '{SENTENCES_KEY}', got {got}")
    cleaned: list[str] = []
    for item in items:
        if not isinstance(item, str):
            raise InvalidResponse(f"entry is {type(item).__name__}, not a sentence")
        text = item.strip()
        if not _HAS_LETTER_RE.search(text):
            raise InvalidResponse(f"entry carries no letters: {text[:40]!r}")
        cleaned.append(text)
    return cleaned


def suspicious_count(cleaned: Sequence[str], source: Sequence[str]) -> int:
    """Sentences the model shrank or inflated so much that the rewrite is doubtful."""
    flags = 0
    for new, old in zip(cleaned, source):
        if len(new) < len(old) * 0.25 or len(new) > len(old) * 4 + 20:
            flags += 1
    return flags


def batch_bounds(sentences: Sequence[str], settings: Settings) -> list[tuple[int, int]]:
    bounds: list[tuple[int, int]] = []
    start = 0
    chars = 0
    for index, sentence in enumerate(sentences):
        chars += len(sentence) + 1
        count = index - start + 1
        if count > settings.batch_sentences or (chars > settings.batch_chars and count > 1):
            bounds.append((start, index))
            start = index
            chars = len(sentence) + 1
    if start < len(sentences):
        bounds.append((start, len(sentences)))
    return bounds


def retry_after(exc: RateLimitError) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) or {}
    for name, scale in (("retry-after-ms", 1 / 1000), ("retry-after", 1.0)):
        try:
            return float(headers[name]) * scale
        except (KeyError, TypeError, ValueError):
            continue
    return None


_FATAL_ERRORS = (AuthenticationError, NotFoundError)
_TRANSIENT_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError, APIError)


async def request_rewrite(
    client: AsyncOpenAI,
    settings: Settings,
    semaphore: asyncio.Semaphore,
    sentences: Sequence[str],
    log: logging.Logger,
    stats: RunStats,
) -> list[str]:
    """Rewrite one batch, retrying transient failures and splitting oversized ones."""
    payload = json.dumps(list(sentences), ensure_ascii=False)
    for attempt in range(settings.max_retries + 1):
        try:
            async with semaphore:
                stats.requests += 1
                completion = await client.chat.completions.create(
                    model=settings.model,
                    temperature=settings.temperature,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": payload},
                    ],
                    timeout=settings.request_timeout,
                )
            usage = getattr(completion, "usage", None)
            if usage is not None:
                stats.prompt_tokens += usage.prompt_tokens or 0
                stats.completion_tokens += usage.completion_tokens or 0
            cleaned = parse_reply(completion.choices[0].message.content or "", len(sentences))
            stats.suspicious += suspicious_count(cleaned, sentences)
            return cleaned
        except _FATAL_ERRORS as exc:
            raise BatchError(f"fatal: {type(exc).__name__}: {exc}") from exc
        except BadRequestError as exc:
            if len(sentences) > 1:
                middle = len(sentences) // 2
                log.warning("batch rejected (%s); splitting %d sentences", exc, len(sentences))
                head = await request_rewrite(client, settings, semaphore, sentences[:middle], log, stats)
                tail = await request_rewrite(client, settings, semaphore, sentences[middle:], log, stats)
                return head + tail
            raise BatchError(f"model rejected the batch: {exc}") from exc
        except _TRANSIENT_ERRORS as exc:
            if attempt >= settings.max_retries:
                raise BatchError(f"{type(exc).__name__}: {exc}") from exc
            wait = retry_after(exc) if isinstance(exc, RateLimitError) else None
            if wait is None:
                wait = settings.backoff * 2**attempt
            stats.retries += 1
            log.warning("retry %d/%d in %.1fs: %s", attempt + 1, settings.max_retries, wait, exc)
            await asyncio.sleep(wait)
        except (json.JSONDecodeError, InvalidResponse) as exc:
            if attempt >= settings.max_retries:
                raise BatchError(f"unusable reply after {attempt + 1} tries: {exc}") from exc
            stats.retries += 1
            log.warning("invalid reply (%s); retry %d/%d", exc, attempt + 1, settings.max_retries)
            await asyncio.sleep(settings.backoff * 2**attempt)
    raise BatchError("unreachable: retry loop exhausted")


async def process_file(
    client: AsyncOpenAI,
    settings: Settings,
    input_path: Path,
    output_path: Path,
    file_semaphore: asyncio.Semaphore,
    api_semaphore: asyncio.Semaphore,
    log: logging.Logger,
    stats: RunStats,
) -> None:
    started = time.monotonic()
    record: dict[str, Any] = {"file": input_path.name, "ok": False, "error": None}
    async with file_semaphore:
        try:
            corpus = await asyncio.to_thread(CorpusFile.load, input_path)
            bounds = batch_bounds(corpus.sentences, settings)
            record.update(sentences=len(corpus.sentences), batches=len(bounds))
            rewrites = await asyncio.gather(
                *(
                    request_rewrite(client, settings, api_semaphore, corpus.sentences[start:stop], log, stats)
                    for start, stop in bounds
                ),
                return_exceptions=True,
            )
            fatal = [error for error in rewrites if str(error).startswith("fatal:")]
            if fatal:
                raise fatal[0]
            failed = [index for index, batch in enumerate(rewrites) if isinstance(batch, BaseException)]
            if failed and settings.on_batch_failure == "abort":
                raise BatchError(
                    f"{len(failed)} of {len(bounds)} batches failed, first: {rewrites[failed[0]]}"
                )
            for index, (start, stop) in enumerate(bounds):
                if isinstance(rewrites[index], BaseException):
                    log.error("%s: batch %d written unchanged (%s)", input_path.name, index, rewrites[index])
                    continue
                corpus.sentences[start:stop] = rewrites[index]
            await asyncio.to_thread(write_json, output_path, corpus.rebuilt(), settings.indent)
            record.update(ok=True, seconds=round(time.monotonic() - started, 1))
            log.info(
                "done %s: %d sentences, %d batches in %ss",
                input_path.name,
                record["sentences"],
                record["batches"],
                record["seconds"],
            )
        except Exception as exc:  # noqa: BLE001 - a broken file must not stop the run
            record.update(seconds=round(time.monotonic() - started, 1), error=str(exc))
            log.error("FAILED %s: %s", input_path.name, exc)
    stats.results.append(record)


def plan(files: Sequence[Path], output_dir: Path, overwrite: bool) -> tuple[list[tuple[Path, Path]], int]:
    jobs: list[tuple[Path, Path]] = []
    skipped = 0
    for path in files:
        destination = output_dir / path.name
        if destination.exists() and not overwrite:
            skipped += 1
            continue
        jobs.append((path, destination))
    return jobs, skipped


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Academic rewrite of the rule-cleaned Persian corpus via the OpenAI API.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR, help="Folder of stage-1 cleaned JSON files")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Folder for the rewritten JSON files")
    parser.add_argument("--pattern", default="*.json", help="Glob of input files")
    parser.add_argument("--model", default="gpt-4o-mini", help="Chat model used for the rewrite")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL"), help="Override the API base URL")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY", help="Environment variable holding the API key")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature")
    parser.add_argument("--batch-sentences", type=int, default=40, help="Sentences per request")
    parser.add_argument("--batch-chars", type=int, default=9000, help="Maximum source characters per request")
    parser.add_argument("--concurrency", type=int, default=5, help="Files processed at the same time")
    parser.add_argument("--max-inflight", type=int, default=20, help="API requests running at the same time")
    parser.add_argument("--request-timeout", type=float, default=120.0, help="Seconds before a request is abandoned")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries per batch after transient errors")
    parser.add_argument("--backoff", type=float, default=2.0, help="Initial retry delay in seconds")
    parser.add_argument(
        "--on-batch-failure",
        choices=("abort", "keep"),
        default="abort",
        help="abort: leave the file unwritten and retry later; keep: write the untouched batch",
    )
    parser.add_argument("--limit-files", type=int, default=0, help="Process only the first N pending files")
    parser.add_argument("--overwrite", action="store_true", help="Redo files that already exist in the output folder")
    parser.add_argument("--indent", type=int, default=2, help="JSON indent of the output files")
    parser.add_argument("--log-file", type=Path, default=None, help="Also write the log to this file")
    parser.add_argument("--dry-run", action="store_true", help="Show the work plan and token estimate, call nothing")
    parser.add_argument("--verbose", action="store_true", help="Debug-level logging")
    return parser.parse_args(argv)


def build_logger(args: argparse.Namespace) -> logging.Logger:
    logger = logging.getLogger("academic_rewrite")
    logger.setLevel(logging.DEBUG if args.verbose else logging.INFO)
    logger.handlers.clear()
    stream = logging.StreamHandler(sys.stderr)
    stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s", "%H:%M:%S"))
    logger.addHandler(stream)
    if args.log_file is not None:
        handler = logging.FileHandler(args.log_file, encoding="utf-8")
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(message)s"))
        logger.addHandler(handler)
    return logger


def estimate(sentences: int, chars: int, batches: int) -> str:
    # Persian is token-hungry; ~2.6 source characters per cl100k token is a working heuristic.
    tokens = (chars / 2.6) * 2
    return (
        f"{sentences:,} sentences | {chars:,} source chars | {batches:,} requests | "
        f"~{tokens / 1_000_000:.1f}M tokens in+out"
    )


def dry_run(jobs: Sequence[tuple[Path, Path]], settings: Settings, log: logging.Logger) -> int:
    total_sentences = 0
    total_batches = 0
    total_chars = 0
    for input_path, _ in jobs:
        try:
            corpus = CorpusFile.load(input_path)
        except Exception as exc:  # noqa: BLE001
            log.error("%s: %s", input_path.name, exc)
            continue
        bounds = batch_bounds(corpus.sentences, settings)
        total_sentences += len(corpus.sentences)
        total_batches += len(bounds)
        total_chars += sum(len(sentence) for sentence in corpus.sentences)
    log.info(
        "dry run: %d files | %s",
        len(jobs),
        estimate(total_sentences, total_chars, total_batches),
    )
    log.info(
        "dry run: ~%.1f h at %d in-flight requests and ~%.0f s per request",
        total_batches * AVG_REQUEST_SECONDS / settings.max_inflight / 3600,
        settings.max_inflight,
        AVG_REQUEST_SECONDS,
    )
    return 0


async def run(args: argparse.Namespace, settings: Settings, log: logging.Logger) -> int:
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    if not input_dir.is_dir():
        log.error("input directory not found: %s", input_dir)
        return 2
    files = sorted(input_dir.glob(args.pattern))
    if not files:
        log.error("no %s files in %s", args.pattern, input_dir)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    jobs, skipped = plan(files, output_dir, args.overwrite)
    if args.limit_files > 0:
        jobs = jobs[: args.limit_files]
    log.info("%d file(s) to rewrite, %d already done", len(jobs), skipped)
    if not jobs:
        return 0
    if args.dry_run:
        return dry_run(jobs, settings, log)

    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        log.error("missing API key: set %s (or pass --api-key-env)", args.api_key_env)
        return 2

    stats = RunStats()
    client = AsyncOpenAI(api_key=api_key, base_url=args.base_url, max_retries=0)
    file_semaphore = asyncio.Semaphore(settings.file_concurrency)
    api_semaphore = asyncio.Semaphore(settings.max_inflight)
    started = time.monotonic()
    try:
        await asyncio.gather(
            *(
                process_file(
                    client, settings, source, destination, file_semaphore, api_semaphore, log, stats
                )
                for source, destination in jobs
            )
        )
    finally:
        await client.close()

    minutes = (time.monotonic() - started) / 60
    kept = sum(item["sentences"] for item in stats.results if item["ok"])
    log.info(
        "done: %d ok, %d failed, %s sentences rewritten in %.1f min | %d requests, %d retries, "
        "%d suspicious rewrites | %d prompt + %d completion tokens",
        len(stats.results) - len(stats.failed),
        len(stats.failed),
        f"{kept:,}",
        minutes,
        stats.requests,
        stats.retries,
        stats.suspicious,
        stats.prompt_tokens,
        stats.completion_tokens,
    )
    for item in stats.failed:
        log.warning("failed file: %s (%s)", item["file"], item["error"])
    return 1 if stats.failed else 0


def main(argv: Sequence[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = parse_args(argv)
    log = build_logger(args)
    settings = Settings(
        model=args.model,
        batch_sentences=args.batch_sentences,
        batch_chars=args.batch_chars,
        file_concurrency=args.concurrency,
        max_inflight=args.max_inflight,
        request_timeout=args.request_timeout,
        max_retries=args.max_retries,
        backoff=args.backoff,
        temperature=args.temperature,
        on_batch_failure=args.on_batch_failure,
        indent=args.indent,
    )
    try:
        return asyncio.run(run(args, settings, log))
    except KeyboardInterrupt:
        log.error("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
