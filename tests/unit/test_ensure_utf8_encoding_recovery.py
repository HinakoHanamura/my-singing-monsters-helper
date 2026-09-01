"""Guard the encoding-repair tool that protects the CJK documents.

Why this file exists
--------------------
``tools/ensure_utf8.py`` is the only thing standing between the project's
Chinese-language documents and silent corruption: the editor writes them as the
Chinese ANSI code page often enough that manual checking is not viable. Three
real incidents motivated these tests.

* A superscript character was encoded as the GB18030 four-byte sequence
  b"\\x81\\x30\\x85\\x35". Because the fallback list only tried gbk, the file
  decoded as neither UTF-8 nor gbk and looked irrecoverably corrupt when it was
  in fact losslessly recoverable.
* The legacy agent directory sat in ``SKIP_DIRS``, and since that test matches any path *part*,
  the entire tree was excluded -- including the rules documents (now under
  ``.agents/rules/``), the files
  most likely to be mis-encoded. The exclusion was invisible: scanning with
  ``--root .agents`` printed nothing at all, which reads like "clean".
* A partial edit left one region of a file in UTF-8 and the rest in the ANSI
  code page. Such a file can still decode cleanly under gb18030, so the tool
  would "repair" it and mojibake every character that had been correct.
  Refusing is the only safe answer, so the refusal is asserted here.

All three failure modes are the quiet kind, so they get assertions rather than
trust.
"""

from __future__ import annotations

import importlib.util
import pathlib

import pytest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
TOOL_PATH = PROJECT_ROOT / "tools" / "ensure_utf8.py"


def load_tool():
    """Import the tool by path.

    ``tools/`` is a plain directory of scripts rather than a package, so it is
    loaded explicitly instead of relying on import machinery that would need an
    ``__init__.py`` added purely for the tests.
    """
    spec = importlib.util.spec_from_file_location("ensure_utf8_under_test", TOOL_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def tool():
    return load_tool()


# The exact text that triggered the original incident: CJK plus a superscript
# that gbk cannot represent, forcing gb18030's four-byte form.
GB18030_SAMPLE = "面积 ∈ [40², 200²]\n判定按路径分量做\n"


# --- recovering whole-file legacy encodings -----------------------------------


def test_gb18030_four_byte_sequence_is_recovered(tool):
    """A gb18030 file with four-byte sequences round-trips into UTF-8."""
    raw = GB18030_SAMPLE.encode("gb18030")

    # Precondition: this really is the shape that broke before, otherwise the
    # test would pass for the wrong reason.
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    with pytest.raises(UnicodeDecodeError):
        raw.decode("gbk")

    needs_fix, text, encoding = tool.classify(raw)

    assert needs_fix is True
    assert encoding == "gb18030"
    assert text == GB18030_SAMPLE


def test_plain_gbk_still_reports_the_stricter_encoding(tool):
    """gbk is tried before gb18030 so the reported encoding stays specific."""
    raw = "判定按路径分量做\n".encode("gbk")

    needs_fix, text, encoding = tool.classify(raw)

    assert needs_fix is True
    assert encoding == "gbk"
    assert text == "判定按路径分量做\n"


def test_utf8_file_is_left_alone(tool):
    """Already-correct files must not be rewritten, not even byte-identically."""
    raw = GB18030_SAMPLE.encode("utf-8")

    needs_fix, text, encoding = tool.classify(raw)

    assert needs_fix is False
    assert text is None
    assert encoding == "utf-8"


def test_utf16_is_recognised_only_from_its_bom(tool):
    """Trial decoding must never be used for UTF-16.

    Without a BOM a UTF-16 guess would reinterpret Chinese ANSI bytes and
    rewrite them as mojibake, which has happened once already.
    """
    with_bom = "\ufeff判定\n".encode("utf-16-le")
    assert with_bom.startswith(b"\xff\xfe")

    needs_fix, text, encoding = tool.classify(with_bom)

    assert needs_fix is True
    assert encoding == "utf-16-le"
    assert "判定" in text


# --- refusing mixed encodings -------------------------------------------------
#
# Not every mixture is dangerous: many happen to be invalid under every codec,
# and those are already safe because the tool reports UNKNOWN and moves on. The
# dangerous ones are those a fallback still accepts, and whether the bytes line
# up that way is not obvious by inspection. The two samples below were found by
# search and verified to decode under gbk, cp936 and gb18030, so they exercise
# the path the guard exists for.


def make_mixed(head_utf8: str, tail_legacy: str) -> bytes:
    """Reproduce a partial edit: UTF-8 head, ANSI code page tail."""
    return head_utf8.encode("utf-8") + tail_legacy.encode("gb18030")


#: CJK-dense document, the shape of the rules notes.
DENSE_HEAD = "面积判定按路径分量做\n"
DENSE_TAIL = "替换后内容：匹配后留在盘面，机会数恒为一点五倍对数。\n"

#: Mostly-ASCII source file with one Chinese comment. A ratio-based test alone
#: misses this, which is why detection also inspects the valid prefix.
SPARSE_HEAD = '"""doc"""\n\n# 卡背统一\nMIN_ASPECT = 0.85\n'
SPARSE_TAIL = "机会数恒为一点五倍对数\n"


@pytest.mark.parametrize(
    "head, tail",
    [(DENSE_HEAD, DENSE_TAIL), (SPARSE_HEAD, SPARSE_TAIL)],
    ids=["cjk_dense_document", "mostly_ascii_source"],
)
def test_fallback_decodable_mixture_is_refused(tool, head, tail):
    raw = make_mixed(head, tail)

    # Precondition: broken as UTF-8, yet a fallback accepts it. That gap is the
    # trap; if either half of this stopped holding the test would be vacuous.
    with pytest.raises(UnicodeDecodeError):
        raw.decode("utf-8")
    raw.decode("gb18030")

    assert tool.looks_mixed(raw) is True

    needs_fix, text, encoding = tool.classify(raw)

    assert needs_fix is True
    assert encoding == "mixed"
    # No text means main() reports and leaves the file alone.
    assert text is None


def test_whole_file_legacy_encoding_is_not_mistaken_for_mixed(tool):
    """The common case must still convert, or the guard would break the tool."""
    document = (
        "判定按路径分量做，机会数恒为一点五倍对数。\n"
        "卡背尺寸跨关卡变化约一点九倍，单一尺度模板盖不住。\n"
    ) * 20
    raw = document.encode("gb18030")

    assert tool.looks_mixed(raw) is False

    needs_fix, text, encoding = tool.classify(raw)

    assert needs_fix is True
    assert encoding in {"gbk", "cp936", "gb18030"}
    assert text == document


def test_legacy_file_with_a_long_ascii_header_is_not_mistaken_for_mixed(tool):
    """An ASCII prologue must not be read as evidence of a UTF-8 region.

    In a whole-file legacy document the first non-ASCII byte is already invalid
    UTF-8, so the valid prefix stays pure ASCII however long the header is.
    """
    document = (
        '"""Normalize project source files to UTF-8."""\n\n'
        "from __future__ import annotations\n\n"
        "SUFFIXES = {'.py', '.md'}\n\n"
        "# 判定按路径分量做，所以整棵树都会被跳过。\n"
    )
    raw = document.encode("gb18030")

    assert tool.valid_prefix_non_ascii(raw) == 0
    assert tool.looks_mixed(raw) is False

    needs_fix, text, encoding = tool.classify(raw)

    assert needs_fix is True
    assert text == document


def test_pure_utf8_is_never_called_mixed(tool):
    raw = "槽位下标不得重编号，绝不允许自动点 REPLAY。\n".encode("utf-8")

    assert tool.looks_mixed(raw) is False


def test_pure_ascii_is_never_called_mixed(tool):
    assert tool.looks_mixed(b"MIN_ASPECT = 0.85\n") is False


# --- which trees get scanned --------------------------------------------------


def test_rules_documents_are_scanned(tool, tmp_path):
    """.agents/rules must be visited; .agents/hooks.json must not."""
    rules = tmp_path / ".agents" / "rules"
    rules.mkdir(parents=True)

    (rules / "roadmap.md").write_text("x", encoding="utf-8")
    (tmp_path / ".agents" / "hooks.json").write_text("{}", encoding="utf-8")

    found = {
        path.relative_to(tmp_path).as_posix() for path in tool.iter_candidates(tmp_path)
    }

    assert ".agents/rules/roadmap.md" in found
    assert ".agents/hooks.json" not in found


def test_generated_and_vendored_trees_are_still_skipped(tool, tmp_path):
    """The original exclusions must survive the .agents change."""
    for part in ("__pycache__", ".git", "assets", "node_modules"):
        directory = tmp_path / part
        directory.mkdir()
        (directory / "note.md").write_text("x", encoding="utf-8")

    (tmp_path / "kept.md").write_text("x", encoding="utf-8")

    found = {
        path.relative_to(tmp_path).as_posix() for path in tool.iter_candidates(tmp_path)
    }

    assert found == {"kept.md"}
