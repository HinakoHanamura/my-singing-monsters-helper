"""Unit tests for BlacklistRow and BlacklistTableWidget."""

from __future__ import annotations

from PySide6.QtCore import QEvent
from PySide6.QtWidgets import QApplication
import pytest

from ui.main_window import BlacklistRow, BlacklistTableWidget


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_blacklist_initialization(qapp: QApplication) -> None:
    """Initial items should be created as confirmed rows, plus one trailing empty row."""
    table = BlacklistTableWidget(initial_items=["Gold Island", "Air Island"])
    # 2 confirmed + 1 trailing empty = 3 rows
    assert len(table._rows) == 3
    assert table._rows[0].confirmed_text == "Gold Island"
    assert table._rows[0].is_confirmed is True
    assert table._rows[0].btn.text() == "×"

    assert table._rows[1].confirmed_text == "Air Island"
    assert table._rows[1].is_confirmed is True
    assert table._rows[1].btn.text() == "×"

    assert table._rows[2].confirmed_text == ""
    assert table._rows[2].is_confirmed is False
    assert table._rows[2].btn.text() == "✔"

    assert table.get_blacklist() == ["Gold Island", "Air Island"]


def test_blacklist_confirm_new_row(qapp: QApplication) -> None:
    """Entering text on trailing row and clicking confirm button adds and appends next row."""
    table = BlacklistTableWidget(initial_items=[])
    assert len(table._rows) == 1

    last_row = table._rows[0]
    last_row.edit.setText("The Colossingum")
    last_row.btn.click()

    assert last_row.is_confirmed is True
    assert last_row.btn.text() == "×"
    assert table.get_blacklist() == ["The Colossingum"]

    # Next empty row automatically appended
    assert len(table._rows) == 2
    assert table._rows[1].is_confirmed is False
    assert table._rows[1].btn.text() == "✔"


def test_blacklist_modify_and_rollback_on_focus_loss(qapp: QApplication) -> None:
    """Modifying a confirmed row changes button to confirm; losing focus rolls back."""
    table = BlacklistTableWidget(initial_items=["Plant Island"])
    row = table._rows[0]
    assert row.btn.text() == "×"

    # User modifies text
    row.edit.setText("Plant Island Edited")
    row._on_text_edited("Plant Island Edited")
    assert row.is_confirmed is False
    assert row.btn.text() == "✔"

    # User loses focus without clicking confirm
    event = QEvent(QEvent.Type.FocusOut)
    row.eventFilter(row.edit, event)

    # Rolled back to original
    assert row.edit.text() == "Plant Island"
    assert row.is_confirmed is True
    assert row.btn.text() == "×"


def test_blacklist_modify_and_confirm(qapp: QApplication) -> None:
    """Modifying a confirmed row and clicking confirm updates the name and does not delete it."""
    table = BlacklistTableWidget(initial_items=["Plant Island"])
    row = table._rows[0]
    assert row.btn.text() == "×"

    # User modifies text
    row.edit.setText("Cold Island")
    row._on_text_edited("Cold Island")
    assert row.btn.text() == "✔"

    # User clicks confirm button
    row.btn.click()

    # Must be updated, confirmed, and NOT deleted
    assert row.is_confirmed is True
    assert row.btn.text() == "×"
    assert table.get_blacklist() == ["Cold Island"]


def test_blacklist_delete_row(qapp: QApplication) -> None:
    """Clicking delete button removes the row while preserving other rows order."""
    table = BlacklistTableWidget(initial_items=["First", "Second", "Third"])
    assert len(table._rows) == 4

    # Delete "Second" (index 1)
    row_to_del = table._rows[1]
    row_to_del.btn.click()

    assert table.get_blacklist() == ["First", "Third"]
    assert len(table._rows) == 3
