"""Thin, polite wrappers around yt-dlp for public metadata only.

Never follows or fetches URLs found in video descriptions (including Amazon
affiliate links). Description text is only used for length and is never crawled.
"""

from __future__ import annotations

import json
import subprocess
import time
from typing import Any, Iterable

YTDLP = "yt-dlp"
DEFAULT_SLEEP = 0.7


class YtDlpError(RuntimeError):
    pass


def run_ytdlp(args: list[str], timeout: int = 180) -> tuple[int, str, str]:
    cmd = [YTDLP, "--no-warnings", "--ignore-errors", *args]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    return proc.returncode, proc.stdout, proc.stderr


def iter_json_lines(
    args: list[str],
    timeout: int = 300,
    retries: int = 2,
) -> list[dict[str, Any]]:
    last_err = ""
    for attempt in range(retries + 1):
        try:
            code, out, err = run_ytdlp(args, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            last_err = f"timeout: {exc}"
            time.sleep(2 + attempt * 2)
            continue
        rows: list[dict[str, Any]] = []
        for line in out.splitlines():
            line = line.strip()
            if not line or not line.startswith("{"):
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        if rows:
            return rows
        last_err = err.strip() or f"exit {code}, no json"
        # Bot-check or empty result: brief backoff then retry.
        time.sleep(2 + attempt * 3)
    raise YtDlpError(last_err[:500] or "yt-dlp returned no data")


def dump_json(args: list[str], timeout: int = 180, retries: int = 2) -> dict[str, Any]:
    last_err = ""
    for attempt in range(retries + 1):
        try:
            code, out, err = run_ytdlp(args, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            last_err = f"timeout: {exc}"
            time.sleep(2 + attempt * 2)
            continue
        out = out.strip()
        if out.startswith("{") or out.startswith("["):
            try:
                data = json.loads(out)
            except json.JSONDecodeError:
                last_err = "invalid json"
                time.sleep(1 + attempt)
                continue
            if isinstance(data, list):
                return data[0] if data else {}
            return data
        last_err = err.strip() or f"exit {code}"
        time.sleep(2 + attempt * 3)
    raise YtDlpError(last_err[:500] or "yt-dlp returned no json")


def polite_sleep(seconds: float = DEFAULT_SLEEP) -> None:
    time.sleep(seconds)


def unique(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out
