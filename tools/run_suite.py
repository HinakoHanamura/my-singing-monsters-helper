"""Run the test suite and write a one-line verdict to reports/suite_result.txt.

The interactive terminal in this environment swallows or garbles pytest's own
output, so the result has to land in a file that can be read back reliably.
Keeping it as a committed tool means the same command works next time instead of
being reinvented.

Usage:
    python tools/run_suite.py            # whole suite
    python tools/run_suite.py tests/unit # a subset

How to wait for it, and how not to
----------------------------------
The report files are deleted on startup and written once at the end, so their
absence means "still running" and nothing more. Two ways to misread that, both of
which have cost real time:

* **Do not stop the background terminal while waiting.** Stopping it kills pytest
  mid-run, so the report is never written -- and the missing file then looks
  exactly like a hang, which invites stopping and retrying again. That loop was
  entered several times before the cause was spotted; the suite had been healthy
  throughout.
* **Do not run this in the foreground.** Commands lasting more than about a
  second are cut off here, and pytest needs longer than that just to import
  PySide6.

The full suite takes roughly 48 seconds. Wait for the file, do not interrogate
the terminal.

If it genuinely seems stuck, do not guess -- run pytest from a small python
wrapper with ``subprocess.run(..., timeout=N)`` so a hang is reported as a hang
instead of as silence, and check ``--collect-only`` first since that separates a
startup problem from a slow test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT_DIR = ROOT / "reports"
RESULT_FILE = REPORT_DIR / "suite_result.txt"
OUTPUT_FILE = REPORT_DIR / "suite_output.txt"


def main() -> int:
    REPORT_DIR.mkdir(exist_ok=True)
    # Stale files have caused wrong conclusions before; clear them up front.
    for path in (RESULT_FILE, OUTPUT_FILE):
        path.unlink(missing_ok=True)

    targets = sys.argv[1:] or ["tests"]
    command = [
        sys.executable,
        "-m",
        "pytest",
        *targets,
        "-q",
        "--no-header",
        "-p",
        "no:cacheprovider",
    ]

    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    OUTPUT_FILE.write_text(
        completed.stdout + "\n--- stderr ---\n" + completed.stderr,
        encoding="utf-8",
    )

    # pytest's summary line is the last non-empty line of stdout.
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    summary = lines[-1] if lines else "(no output)"
    RESULT_FILE.write_text(
        f"exit={completed.returncode}\nsummary={summary}\n", encoding="utf-8"
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
