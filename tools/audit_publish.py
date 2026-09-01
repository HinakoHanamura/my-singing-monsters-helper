"""Pre-publication audit: find machine-specific or personal data before a commit.

Run this before the first commit and again before making the repository public.
It walks everything that would actually be committed -- applying the same
exclusions as .gitignore -- and reports absolute paths, the Windows account
name, e-mail addresses and anything decoded as a fallback encoding.

Why it exists: git history cannot be scrubbed after the fact, so a personal
detail committed once is committed forever. Reviewing by eye does not scale and
had already missed things: the first run over 69 files found the Windows account
name in two rules documents, in passages nobody would have thought to check.
Both were paths written down as examples years of edits ago.

This file is deliberately pure ASCII, so the tool itself cannot be mangled by
whatever encoding it gets saved as (see workflow.md, trap one).

Writes reports/audit_publish.txt and prints one line, per the tooling shape in
workflow.md. Zero findings is the only acceptable result.
"""

import os
import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / "reports" / "audit_publish.txt"

# Mirrors .gitignore. Kept explicit because git is not initialised yet, so
# `git check-ignore` cannot be used to answer this.
SKIP_DIRS = {
    "captures",
    "reports",
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "env",
    ".pytest_cache",
    ".vscode",
    ".idea",
}
# Local-only documents. These are excluded in .gitignore, so they are not part
# of "everything that would be committed" and scanning them makes the zero
# findings pass condition unusable: a note written for local reference gets
# reported as if it were about to be published.
SKIP_FILES = {
    ".agents/rules/private.md",
    ".agents/rules/project.md",
    ".agents/rules/roadmap.md",
}
SKIP_SUFFIX = {".png", ".jpg", ".pt", ".pyc", ".log"}
SKIP_GLOB = ("junit",)

def build_patterns():
    """Build the patterns to search for.

    The account name and home directory come from the environment rather than
    being written into this file. That matters for three reasons: this tool
    carries no personal data of its own, it therefore does not flag its own
    source as a finding, and it works unchanged for anyone else who clones the
    repository.

    An earlier version hardcoded the name and immediately reported itself, which
    would have made "zero findings" impossible to use as a pass condition.
    """
    patterns = {
        "abs-path-drive": re.compile(r"[A-Za-z]:[\\/]{1,2}(Users|work|Program)"),
        "email": re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"),
    }
    account = os.environ.get("USERNAME") or os.environ.get("USER")
    if account:
        patterns["account-name"] = re.compile(re.escape(account))
    home = os.environ.get("USERPROFILE") or os.environ.get("HOME")
    if home:
        patterns["home-path"] = re.compile(re.escape(home))
    return patterns

TEXT_ENCODINGS = ("utf-8", "gb18030")


def read_text(path):
    data = path.read_bytes()
    for encoding in TEXT_ENCODINGS:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    return None, None


def tracked_files():
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if any(part in SKIP_DIRS for part in path.relative_to(ROOT).parts[:-1]):
            continue
        if rel in SKIP_FILES:
            continue
        if path.suffix.lower() in SKIP_SUFFIX:
            continue
        if path.name.startswith(SKIP_GLOB):
            continue
        yield rel, path


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.unlink(missing_ok=True)

    patterns = build_patterns()
    lines = []
    count = 0
    findings = 0
    undecodable = []

    for rel, path in tracked_files():
        count += 1
        text, encoding = read_text(path)
        if text is None:
            undecodable.append(rel)
            continue
        if encoding != "utf-8":
            lines.append("ENCODING %s decoded as %s" % (rel, encoding))
            findings += 1
        for number, line in enumerate(text.splitlines(), 1):
            for label, pattern in patterns.items():
                if pattern.search(line):
                    findings += 1
                    lines.append(
                        "%s  %s:%d  %s" % (label, rel, number, line.strip()[:120])
                    )

    header = [
        "patterns: %s" % ", ".join(sorted(patterns)),
        "files scanned (would be committed): %d" % count,
        "findings: %d" % findings,
        "undecodable: %s" % (undecodable or "none"),
        "",
    ]
    OUT.write_text("\n".join(header + lines) + "\n", encoding="utf-8")
    print("wrote %s" % OUT)


if __name__ == "__main__":
    main()
