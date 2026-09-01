"""Normalize project source files to UTF-8.

Why this exists
---------------
On this machine some editors / writers emit .py files using the system ANSI
code page (GBK on a Chinese Windows install) instead of UTF-8. Python 3
assumes UTF-8 for source files unless a PEP 263 coding declaration is present,
so a GBK-encoded file containing CJK characters fails to even parse:

    SyntaxError: (unicode error) 'utf-8' codec can't decode byte 0xbe ...

This script finds such files and rewrites them as UTF-8. It is intentionally
ASCII-only so that it can never corrupt itself.

Behaviour
---------
- A file that already decodes as UTF-8 is left completely untouched.
- A file that fails UTF-8 but decodes as GBK is rewritten as UTF-8.
- A file that decodes as neither is reported and left alone (never guessed at).
- Line endings and content are preserved byte-for-byte apart from the
  encoding change.

Usage
-----
    python tools/ensure_utf8.py           # fix files in place
    python tools/ensure_utf8.py --check   # report only, non-zero exit if dirty

Wired into .agents/hooks.json so it runs automatically after any
agent file write.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

# Extensions worth guarding. Source files are the ones that actually break.
SUFFIXES = {".py", ".md", ".txt", ".json", ".yaml", ".yml"}

# Directories that are never ours to rewrite, matched against any path part.
SKIP_DIRS = {
    "__pycache__",
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "assets",
}

# Sub-trees under .agents that must not be touched, as path prefixes relative to
# the project root.
#
# The legacy agent directory used to sit in SKIP_DIRS, which excluded the whole tree because the test
# matches any path *part*. That silently left .agents/rules/*.md unprotected --
# and those files are full of CJK, so they are exactly the ones that get written
# as GB18030. Only the machine-readable configuration needs shielding, so skip
# that and scan the documentation.
SKIP_PREFIXES = (
    (".agents", "hooks.json"),
)

# Fallback encodings tried in order when UTF-8 fails.
#
# UTF-16 is deliberately NOT in this list. Trial decoding cannot detect it:
# almost any even-length byte sequence decodes as UTF-16 without raising, so
# attempting it would happily reinterpret a GBK file as UTF-16 and rewrite it as
# mojibake. UTF-16 is only ever recognised from its byte-order mark, below.
#
# gb18030 sits after gbk on purpose. It is a superset of gbk, so trying the
# stricter one first keeps the reported encoding as specific as possible, and
# gb18030 then catches the four-byte extension sequences gbk rejects outright.
# Those are not hypothetical: a superscript character in a CJK document was
# written as b"\x81\x30\x85\x35", which made the file look corrupt (neither
# UTF-8 nor gbk) when it was in fact losslessly recoverable.
#
# gb18030 comes before big5 because the ANSI code page on this machine is a
# Chinese one; a Big5 file would be a genuine surprise here, whereas gb18030
# output is produced routinely.
FALLBACKS = ("gbk", "cp936", "gb18030", "big5")

#: Byte-order marks that identify UTF-16 unambiguously.
UTF16_BOMS = ((b"\xff\xfe", "utf-16-le"), (b"\xfe\xff", "utf-16-be"))

# --- mixed-encoding detection -------------------------------------------------
#
# A partial edit can leave one region of a file in UTF-8 and another in the
# Chinese ANSI code page. Such a file decodes cleanly under *no* single codec,
# and that is the one case where this script must refuse to act: gb18030 will
# happily decode the whole thing, so a conversion would "succeed" while turning
# every originally-correct UTF-8 character into mojibake. Making a damaged file
# worse is far more costly than leaving it for a human.
#
# The tell is that a substantial amount of CJK text decodes correctly as UTF-8
# even though the file as a whole does not. Measured on this project:
#
#   file                      replacements   valid CJK   ratio
#   partial edit (mixed)               51          44    0.86
#   whole file as gb18030            3193          10    0.003
#   whole file as gbk                2726          19    0.007
#
# Two orders of magnitude of separation, so the cutoff below is not delicate.
# The occasional accidental hit happens because some legacy byte pairs form a
# valid UTF-8 sequence by chance, which is what the ratio guards against.
MIXED_CJK_RATIO = 0.25

#: Below this many correctly decoded CJK characters, treat coincidence as the
#: more likely explanation and let the normal fallbacks run.
MIXED_MIN_CJK = 4

# The ratio above only separates documents that are *dense* in CJK. A source
# file is mostly ASCII with the odd Chinese comment, so its ratio stays low even
# when it really is mixed -- measured on a test module that had been appended to,
# the ratio was far below the cutoff and the damage went unnoticed.
#
# The sharper signal: decode up to the first error and look at what survived. In
# a whole-file legacy document the very first non-ASCII byte is already invalid
# UTF-8, so the valid prefix is pure ASCII no matter how long the header is. In
# a mixed file the prefix is the genuinely-UTF-8 region and is full of non-ASCII
# characters. A handful is required rather than one, because a legacy byte pair
# can form a valid UTF-8 sequence by chance.
MIXED_MIN_PREFIX_NON_ASCII = 4

#: Unified Han block. Deliberately narrow: it is a signal, not a full survey.
_CJK_FIRST = "\u4e00"
_CJK_LAST = "\u9fff"


def is_skipped(path: pathlib.Path, root: pathlib.Path) -> bool:
    """True when a file lies in a tree this script must not rewrite."""
    if any(part in SKIP_DIRS for part in path.parts):
        return True
    try:
        relative = path.relative_to(root).parts
    except ValueError:
        # Outside the scanned root; nothing to say about it.
        return False
    return any(relative[: len(prefix)] == prefix for prefix in SKIP_PREFIXES)


def iter_candidates(root: pathlib.Path):
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() not in SUFFIXES:
            continue
        if is_skipped(path, root):
            continue
        yield path


def utf8_damage(raw: bytes):
    """Return (valid_cjk_count, replacement_count) for a lossy UTF-8 read."""
    text = raw.decode("utf-8", errors="replace")
    replacements = text.count("\ufffd")
    cjk = sum(1 for char in text if _CJK_FIRST <= char <= _CJK_LAST)
    return cjk, replacements


def valid_prefix_non_ascii(raw: bytes) -> int:
    """Count non-ASCII characters in the UTF-8-valid prefix of ``raw``.

    Zero for a whole-file legacy encoding: its first non-ASCII byte is already
    an error, so nothing but ASCII precedes it.
    """
    try:
        raw.decode("utf-8")
    except UnicodeDecodeError as error:
        head = raw[: error.start].decode("utf-8", errors="ignore")
        return sum(1 for char in head if ord(char) > 127)
    return 0


def looks_mixed(raw: bytes) -> bool:
    """True when the file appears to hold both UTF-8 and legacy-encoded text.

    Two independent signals, either of which is enough. The prefix test catches
    mostly-ASCII source files; the ratio test catches documents whose UTF-8 and
    legacy regions are interleaved rather than split at one point.
    """
    cjk, replacements = utf8_damage(raw)
    if replacements == 0:
        return False

    if valid_prefix_non_ascii(raw) >= MIXED_MIN_PREFIX_NON_ASCII:
        return True

    if cjk < MIXED_MIN_CJK:
        return False
    return cjk >= MIXED_CJK_RATIO * replacements


def classify(raw: bytes):
    """Return (needs_fix, decoded_text_or_None, encoding_name_or_None).

    ``encoding`` is ``"mixed"`` with no text when the file must be left alone;
    see the MIXED_CJK_RATIO comment for why converting would be destructive.
    """
    try:
        raw.decode("utf-8")
        return False, None, "utf-8"
    except UnicodeDecodeError:
        pass

    # UTF-16 only when the BOM says so; never by trial decoding.
    for bom, encoding in UTF16_BOMS:
        if raw.startswith(bom):
            try:
                return True, raw.decode(encoding), encoding
            except UnicodeDecodeError:
                return True, None, None

    # Checked before the fallbacks, because gb18030 would otherwise swallow a
    # mixed file without complaint.
    if looks_mixed(raw):
        return True, None, "mixed"

    for encoding in FALLBACKS:
        try:
            return True, raw.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    return True, None, None


def main() -> int:
    parser = argparse.ArgumentParser(description="Normalize source files to UTF-8.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="report only, do not modify files (exit 1 if any file is not UTF-8)",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="project root to scan (defaults to the parent of this script)",
    )
    args = parser.parse_args()

    root = (
        pathlib.Path(args.root).resolve()
        if args.root
        else pathlib.Path(__file__).resolve().parent.parent
    )

    fixed = []
    undecodable = []

    for path in iter_candidates(root):
        raw = path.read_bytes()
        needs_fix, text, encoding = classify(raw)
        if not needs_fix:
            continue

        relative = path.relative_to(root).as_posix()

        if encoding == "mixed":
            undecodable.append(relative)
            print("MIXED-ENCODING    " + relative + "  (left untouched, needs a manual rewrite)")
            continue

        if text is None:
            undecodable.append(relative)
            print("UNKNOWN-ENCODING  " + relative)
            continue

        if args.check:
            fixed.append(relative)
            print("NOT-UTF8          " + relative + "  (looks like " + str(encoding) + ")")
            continue

        path.write_bytes(text.encode("utf-8"))
        fixed.append(relative)
        print("CONVERTED         " + relative + "  (" + str(encoding) + " -> utf-8)")

    if not fixed and not undecodable:
        # Silent on the happy path keeps the hook output clean.
        return 0

    print("")
    print("files needing conversion: %d, undecodable: %d" % (len(fixed), len(undecodable)))

    if args.check:
        return 1 if (fixed or undecodable) else 0
    return 1 if undecodable else 0


if __name__ == "__main__":
    sys.exit(main())
