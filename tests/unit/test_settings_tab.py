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


def test_settings_tab_initialization(qapp) -> None:
    """Verify Settings tab exists and scan_first checkbox is present and functional."""
    win = MainWindow(config=DEFAULT_CONFIG)

    # 4 tabs total
    assert win._tabs.count() == 4

    # Memory game scan_first default is False
    assert win._scan_first_box.isChecked() is False

    # Toggle scan_first
    win._scan_first_box.setChecked(True)
    assert win._scan_first_box.isChecked() is True
