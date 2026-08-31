"""Entry point.

    python main.py                 # normal run
    python main.py --diagnose      # also dump frames from rounds that miss
    python main.py --title "..."   # override the target window title
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import sys

# Anchored on this file, never the shell's cwd, so the app behaves the same no
# matter where it is launched from.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from PySide6.QtWidgets import QApplication  # noqa: E402

from config import DEFAULT_CONFIG, AppConfig  # noqa: E402
from ui.main_window import MainWindow  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="MSM idle helper")
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help=(
            "dump the frame whenever a round finds nothing clickable. Use this "
            "when the live run disagrees with the calibrated numbers: the saved "
            "frames show whether the zoom, the resolution or the scene changed"
        ),
    )
    parser.add_argument(
        "--title",
        default=None,
        help="override the target window title",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="debug-level console logging, including per-detection rejections",
    )
    return parser


def setup_logging(verbose: bool) -> None:
    """Console logging. The in-app log goes through Signals; this is for triage."""
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def build_config(args) -> AppConfig:
    config = DEFAULT_CONFIG
    if args.title:
        config = dataclasses.replace(
            config, window=dataclasses.replace(config.window, title=args.title)
        )
    if args.diagnose:
        config = dataclasses.replace(
            config,
            diagnostics=dataclasses.replace(
                config.diagnostics, dump_frames_on_miss=True
            ),
        )
    return config


def main() -> int:
    args = build_parser().parse_args()
    setup_logging(args.verbose)

    app = QApplication(sys.argv)
    app.setApplicationName("MSM Helper")
    app.setApplicationDisplayName("MSM Helper")

    window = MainWindow(config=build_config(args))
    window.show()

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
