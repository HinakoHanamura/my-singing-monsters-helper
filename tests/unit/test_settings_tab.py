import pytest
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from config import DEFAULT_CONFIG
from ui.main_window import MainWindow


@pytest.fixture(scope="module")
def qapp():
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def test_settings_tab_initialization(qapp, monkeypatch, tmp_path) -> None:
    """Verify Settings tab exists and scan_first checkbox is present and functional."""
    monkeypatch.setattr("config.SETTINGS_FILE", str(tmp_path / "user_settings.json"))
    win = MainWindow(config=DEFAULT_CONFIG)

    # 4 tabs total
    assert win._tabs.count() == 4

    # Memory game scan_first default is False
    assert win._scan_first_box.isChecked() is False

    # Toggle scan_first
    win._scan_first_box.setChecked(True)
    assert win._scan_first_box.isChecked() is True

    # Map initialization auto-scroll to top defaults to True
    assert win._reset_map_box.isChecked() is True
    assert win._brake_dynamic_radio.isEnabled() is True
    assert win._brake_first_island_radio.isEnabled() is True
    assert win._brake_dynamic_radio.isChecked() is True

    # Switch to first island mode
    win._brake_first_island_radio.setChecked(True)
    assert win._brake_first_island_radio.isChecked() is True
    assert win._first_island_edit.isEnabled() is True
    assert win._first_island_btn.isEnabled() is False

    # Edit island name
    win._first_island_edit.setText("Cold Island")
    assert win._first_island_btn.isEnabled() is True

    # Save island name
    win._first_island_btn.click()
    assert win._saved_first_island_name == "Cold Island"
    assert win._first_island_btn.isEnabled() is False

    # Disabling reset map disables all brake controls
    win._reset_map_box.setChecked(False)
    assert win._reset_map_box.isChecked() is False
    assert win._brake_dynamic_radio.isEnabled() is False
    assert win._brake_first_island_radio.isEnabled() is False
    assert win._first_island_edit.isEnabled() is False

