"""Apply text replacements to a Chinese document, controlling the encoding.

Why this tool exists
--------------------
Two facts about this environment collide (both documented in
``.agents/rules/workflow.md``):

* A partial text edit on a file containing Chinese rewrites the whole file in an
  unpredictable encoding, and has repeatedly produced genuinely mixed-encoding
  files that no single codec can recover.
* Rewriting a whole file is therefore the only safe edit -- but on the 25KB
  rules documents that write has failed outright, repeatedly.

That leaves no safe way to make a small change to a large Chinese document, which
is exactly what keeping the rules docs current requires. So the rewrite happens
here, where the encoding is stated explicitly instead of guessed.

**This file is deliberately pure ASCII.** The Chinese lives in the data files it
reads, so nothing in the tool itself can be mangled by however it was saved. Keep
it that way.

Usage
-----
    python tools/apply_doc_edits.py <target> <edits.md> [<more-edits.md> ...]

Each edits file holds one or more blocks, delimited by three marker lines. Each
marker is three left angle brackets followed by OLD, NEW or END, and is only
recognised at the start of a line -- so an example indented by four spaces, like
the one below, is content rather than syntax:

    <<<OLD
    text to find, verbatim, including indentation
    <<<NEW
    text to put there
    <<<END

Every block must match **exactly once**. If any block matches zero times or more
than once, nothing at all is written and the report says which one -- a guessed
anchor would silently corrupt the document, which is the failure mode this whole
tool exists to avoid.

The target is read with UTF-8 first and the ANSI code pages as fallbacks, so it
works on a document that a previous write left as GB18030, and always writes back
UTF-8. The result is verified by decoding it again before the tool reports success.

Output goes to ``reports/apply_doc_edits.txt`` because terminal output on this
machine is unreliable; only one line is printed. Edits files are meant to be
throwaway -- write them under ``reports/`` and delete them afterwards.
"""

import pathlib
import sys

#: Read order. UTF-8 first, then the ANSI code pages a previous write may have
#: produced. UTF-16 is deliberately absent: trial-decoding it has misread Chinese
#: ANSI before and destroyed a file.
FALLBACKS = ("utf-8", "gb18030", "gbk")

# Markers are recognised only at the start of a line, so an edits file can quote
# the format in an indented example without the parser eating it. That is not a
# hypothetical: documenting this tool inside a rules doc broke the first
# attempt, and the "exactly once" guard is what caught it.
OLD_MARKER = "\n<<<OLD\n"
NEW_MARKER = "\n<<<NEW\n"
END_MARKER = "\n<<<END"


def load_text(path):
    """Decode a file, trying UTF-8 first. Returns (text, encoding used)."""
    data = path.read_bytes()
    for encoding in FALLBACKS:
        try:
            return data.decode(encoding), encoding
        except UnicodeDecodeError:
            continue
    raise SystemExit("cannot decode %s with any of %s" % (path, FALLBACKS))


def parse_blocks(text):
    """Split an edits file into (old, new) pairs."""
    blocks = []
    # Leading newline so a marker on the very first line still anchors.
    for chunk in ("\n" + text).split(OLD_MARKER)[1:]:
        body, _, _ = chunk.partition(END_MARKER)
        old, marker, new = body.partition(NEW_MARKER)
        if not marker:
            raise SystemExit("an OLD block has no NEW section")
        # The newline each marker consumed belongs to the text before it.
        blocks.append((old + "\n", new + "\n"))
    return blocks


def main():
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)

    target = pathlib.Path(sys.argv[1])
    sources = [pathlib.Path(name) for name in sys.argv[2:]]

    report = pathlib.Path("reports/apply_doc_edits.txt")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.unlink(missing_ok=True)
    lines = []

    text, encoding = load_text(target)
    lines.append("read %s as %s, %d chars" % (target, encoding, len(text)))

    pending = []
    for source in sources:
        source_text, source_encoding = load_text(source)
        lines.append("read %s as %s" % (source, source_encoding))
        for index, (old, new) in enumerate(parse_blocks(source_text)):
            count = text.count(old)
            lines.append("  block %d: %d match(es)" % (index, count))
            if count != 1:
                lines.append("    old begins: %r" % old[:70])
            pending.append((old, new, count))

    if not pending:
        lines.append("ABORTED: no blocks found")
    elif any(count != 1 for _, _, count in pending):
        # Partial application would be worse than none: the caller could not tell
        # which half of the intended change is in the file.
        lines.append("ABORTED: not every block matched exactly once, nothing written")
    else:
        for old, new, _ in pending:
            text = text.replace(old, new, 1)
        target.write_bytes(text.encode("utf-8"))
        rewritten = target.read_bytes()
        rewritten.decode("utf-8")  # proof, not decoration
        lines.append(
            "wrote %s as utf-8, %d bytes, %d block(s) applied"
            % (target, len(rewritten), len(pending))
        )

    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("wrote %s" % report)


if __name__ == "__main__":
    main()
