"""Check that every comment and docstring in the codebase is English.

Why this exists
---------------
``project.md`` splits the languages: comments and docstrings are English because
the repository is meant to be read by strangers, while UI labels, log lines and
exception messages stay Chinese because the user is a native speaker. Both kinds
of text sit in the same files, so *counting non-ASCII bytes cannot tell them
apart* -- and a byte count was the only check available while the translation
milestone was in progress. It reported thousands of bytes in files whose comments
were already fully English.

This tool draws the distinction properly. It tokenises each file, marks which
string tokens are docstrings via the AST, and reports each non-ASCII run under
the heading it belongs to:

    COMMENT     must be English -- a finding
    DOCSTRING   must be English -- a finding
    STRING      user-facing text -- reported for information only

The one allowance
-----------------
A comment or docstring may contain Chinese when it is **quoting a string the
program actually produces**. ``grid.describe_grid`` returns a Chinese sentence and
``SlotMap.initial_shape`` stores one; documenting them in English while showing an
English example would describe output that does not exist. The same goes for
naming a UI control by the label the user sees.

So the rule is: Chinese in a comment or docstring must sit inside ASCII double
quotes on its own line. Prose in English, quotations verbatim. Anything else is a
finding.

Usage
-----
    python tools/audit_comment_language.py

Writes ``reports/comment_language.txt`` and prints one line. Exit code is non-zero
when there is at least one finding, so it can gate a milestone.
"""

from __future__ import annotations

import ast
import io
import sys
import tokenize
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORT = ROOT / "reports" / "comment_language.txt"

#: Directories that are not ours to police, plus generated trees.
SKIP_DIRS = ("__pycache__", ".git", ".pytest_cache", "reports", "captures", ".kiro")

STRING_PREFIXES = "rRbBuUfF"


def is_ascii(text: str) -> bool:
    return all(ord(ch) < 0x80 for ch in text)


def docstring_lines(tree: ast.AST) -> set:
    """Line numbers occupied by module, class and function docstrings."""
    lines = set()
    holders = [tree]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            holders.append(node)
    for holder in holders:
        body = getattr(holder, "body", None)
        if not body:
            continue
        first = body[0]
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant):
            if isinstance(first.value.value, str):
                end = first.value.end_lineno or first.value.lineno
                lines.update(range(first.value.lineno, end + 1))
    return lines


def strip_delimiters(token: str) -> str:
    """Remove a string token's prefix and quote delimiters, keeping the body.

    Needed so the quote-parity test below is not thrown off by the ``\"\"\"`` that
    opens a docstring.
    """
    body = token.lstrip(STRING_PREFIXES)
    for quote in ('"""', "'''", '"', "'"):
        if body.startswith(quote):
            body = body[len(quote) :]
            if body.endswith(quote):
                body = body[: -len(quote)]
            break
    return body


def quoted_violations(body: str) -> list:
    """Lines whose non-ASCII text is not enclosed in ASCII double quotes.

    Parity is evaluated per line, so an unpaired quote elsewhere in a long
    docstring cannot turn a compliant quotation into a false finding.
    """
    bad = []
    for line in body.split("\n"):
        if is_ascii(line):
            continue
        inside = False
        for ch in line:
            if ch == '"':
                inside = not inside
            elif ord(ch) >= 0x80 and not inside:
                bad.append(line.strip())
                break
    return bad


def audit_file(path: Path) -> tuple:
    """Returns (findings, info) for one file, each a list of report lines."""
    raw = path.read_bytes()
    if not any(byte >= 0x80 for byte in raw):
        return [], []

    text = raw.decode("utf-8")
    doc_lines = docstring_lines(ast.parse(text))

    findings = []
    info = []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if is_ascii(token.string):
            continue
        if token.type == tokenize.COMMENT:
            kind = "COMMENT"
            body = token.string
        elif token.type == tokenize.STRING:
            kind = "DOCSTRING" if token.start[0] in doc_lines else "STRING"
            body = strip_delimiters(token.string)
        else:
            # Non-ASCII outside a comment or a string would mean an identifier or
            # an operator, which is worth shouting about rather than ignoring.
            findings.append(
                "  L%-5d UNEXPECTED  %r" % (token.start[0], token.string[:80])
            )
            continue

        if kind == "STRING":
            info.append("  L%-5d %s" % (token.start[0], body.split("\n")[0][:90]))
            continue

        for line in quoted_violations(body):
            findings.append("  L%-5d %-9s %s" % (token.start[0], kind, line[:90]))

    return findings, info


def main() -> int:
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.unlink(missing_ok=True)

    lines = []
    finding_count = 0
    string_count = 0

    for path in sorted(ROOT.rglob("*.py")):
        rel = path.relative_to(ROOT).as_posix()
        if any(part in SKIP_DIRS for part in rel.split("/")):
            continue
        findings, info = audit_file(path)
        if not findings and not info:
            continue
        finding_count += len(findings)
        string_count += len(info)
        lines.append("%s  findings=%d  chinese-strings=%d" % (rel, len(findings), len(info)))
        lines.extend(findings)
        lines.append("")

    lines.append("=" * 72)
    lines.append("findings (comment/docstring not in English): %d" % finding_count)
    lines.append("chinese string literals (expected, user-facing): %d" % string_count)
    if finding_count == 0:
        lines.append("PASS")
    else:
        lines.append("FAIL")

    REPORT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote %s" % REPORT)
    return 1 if finding_count else 0


if __name__ == "__main__":
    raise SystemExit(main())
