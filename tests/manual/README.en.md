# tests/manual - scripts that need the real game

**English** | [简体中文](README.md)

These are **not automated tests**. pytest does not collect them (see
`--ignore=tests/manual` in `pytest.ini`). They require *My Singing Monsters* to
be running, they really do capture the screen and some of them really do click,
so they are run by hand and their results are read by a human.

| Script | Purpose | Does it click? |
| --- | --- | --- |
| `test_eyes.py` | proves `PrintWindow` can capture an unfocused, covered window | no |
| `test_hands.py` | proves `SendMessage` delivers a click without moving the physical mouse | **yes, with no protection whatsoever** |
| `test_memory_minigame.py` | live entry point for the memory minigame | **no by default**, add `--play` to click |

The first two were the project's original feasibility spikes, and they grew into
`core/game_window.py` and `core/action_agent.py`. They are kept for two reasons:
they record where the whole framework started, and when perception or clicking
breaks wholesale they answer -- with the fewest possible dependencies -- whether
the problem is inside the framework or outside it.

## Usage

```powershell
python tests/manual/test_eyes.py     # writes test_background.png next to this file
python tests/manual/test_hands.py    # clicks client-area (200, 200) once
```

For the memory minigame, **run observation mode first**:

```powershell
python tests/manual/test_memory_minigame.py             # observe only, no clicks
python tests/manual/test_memory_minigame.py --seconds 30
python tests/manual/test_memory_minigame.py --play --levels 1   # one level only
python tests/manual/test_memory_minigame.py --play              # all nine
```

**Before running**: open the minigame by hand and leave it on **a level's
starting board**, every card face down. A slot table can only be built from a
complete board, and a single frame cannot distinguish "an even number already
revealed" from "complete", so starting mid-level makes the script blind to the
cards it missed. It does detect that and stop, but starting from a fresh board
avoids the problem entirely.

The window may be covered and may sit in the background, but it **must not be
minimised** -- Windows stops rendering the client area of a minimised window, so
there is nothing to capture.

Annotated frames are written to `reports/manual_memory/`: `board.png` (with slot
numbers), `rejected.png` (what the screen looked like when the board gate refused
it), and `stopped.png` (what it looked like when a live run stopped).

## Notes

- `test_hands.py` **really clicks**, with no target validation and no
  anti-misclick protection at all: whatever sits at (200, 200) is what gets
  clicked. For a probe that keeps the safety rules, use `tools/probe_click.py`.
- `test_memory_minigame.py --play` clicks cards. After a failed level the game
  offers "replay for 2 diamonds", and the loop is *structurally* unable to click
  anything that is not a card: every target comes from a card-back detection in
  the current frame, and is re-confirmed to still fall inside a detected box
  before the click is sent, while the results screen produces no card-back boxes
  at all. **But that is a property asserted by unit tests, and a live run meets
  screens nobody anticipated, so run observation mode first.**
- `test_eyes.py` deliberately omits the DPI correction in `core/game_window.py`.
  On a scaled display it reports the magnified client size and captures only the
  top-left corner of the frame -- which is exactly how the original DPI bug
  revealed itself. That difference is kept on purpose.
- All three scripts keep their body in `main()` behind an
  `if __name__ == "__main__"` guard. The filenames match `test_*.py`, so if they
  are ever collected by accident, importing them still triggers no capture and no
  click.
- The `test_background.png` they produce is a frame of this machine's game
  window and is excluded in `.gitignore`, as is the whole `reports/` directory.
