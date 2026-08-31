"""Shared pytest configuration and fixtures.

Qt needs an offscreen platform before any QApplication is constructed, which is
why the environment variable is set at import time rather than inside a fixture.
"""

from __future__ import annotations

import os
import sys

# Must happen before PySide6 creates a QApplication anywhere.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from tests.synthetic import (  # noqa: E402
    REFERENCE_SIZE,
    make_coin_template,
    make_scene,
    write_template,
)


@pytest.fixture(scope="session")
def qapp():
    """One QApplication for the whole test session.

    Qt does not support creating and destroying multiple QApplication objects in
    a single process, so this is deliberately session scoped.
    """
    from PySide6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app
    app.processEvents()


@pytest.fixture(scope="session")
def coin_template() -> np.ndarray:
    return make_coin_template()


@pytest.fixture
def template_dir(tmp_path, coin_template) -> str:
    """A throwaway template directory containing a single coin template."""
    write_template(tmp_path, "coin", coin_template)
    return str(tmp_path)


@pytest.fixture
def reference_scene() -> np.ndarray:
    """Reference-resolution frame with two coins, one of them above a blob."""
    return make_scene(
        REFERENCE_SIZE[0],
        REFERENCE_SIZE[1],
        coin_centers=[(900, 400), (1300, 620)],
        monster_under=[(900, 400)],
    )
