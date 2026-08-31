# Test suite

```powershell
python -m pytest                 # everything
python -m pytest tests/unit      # fast, no Qt event loop
python -m pytest -m "not slow"   # skip the timed integration runs
python -m pytest -q --no-header  # terse
```

Qt runs on the `offscreen` platform (set in `conftest.py`), so no window ever
appears and the suite is CI-friendly.

## Layout

| Path | Covers |
|---|---|
| `unit/test_geometry_scale_conversion.py` | Reference-pixel to live-pixel arithmetic; the foundation of window-size adaptation |
| `unit/test_vision_multi_resolution_adaptation.py` | One template detected correctly at 854x480 through 2560x1440 |
| `unit/test_vision_template_variants_and_dedup.py` | Multi-frame animation templates, duplicate suppression, threshold behaviour |
| `unit/test_validators_anti_misclick_rules.py` | Exclusion zones, edge guard, confidence floor, texture probe, neighbour rules, chain ordering |
| `unit/test_click_guard_confirmation_cooldown_blacklist.py` | Cross-frame confirmation, per-position cooldown, self-correcting blacklist |
| `integration/test_engine_decision_pipeline.py` | Full detect -> validate -> guard -> click -> verify loop on a worker thread |
| `integration/test_ui_engine_lifecycle.py` | Button state machine, log rendering and escaping, clean shutdown |
| `real_frames/` | Reserved for regression tests driven by real game captures (see below) |

## What synthetic images do and do not prove

`synthetic.py` draws a high-contrast icon on a flat background. That is the
*correct* input for testing arithmetic and control flow: the expected output is
exactly knowable, and the tests cannot flake.

It proves nothing about detection quality. A drawn disc is not an animated,
semi-transparent game sprite on cluttered terrain. So these tests deliberately
**do not** assert that any particular threshold value is correct.

Perception parameters that still need real data:

| Parameter | Status |
|---|---|
| `VisionConfig.reference_size` | Measured from a real capture |
| `WindowConfig.print_window_flag` | Verified against the real game |
| `VisionConfig.match_threshold` | **Unvalidated placeholder** |
| `SafetyConfig.texture_min_std` | **Unvalidated placeholder**, and the riskiest one: real terrain is far busier than the synthetic background, so this rule may pass everything while appearing to work |
| `SafetyConfig.exclusion_zones` | Estimated from the island-select screen, not the in-game view |
| `VisionConfig.nms_distance` | Reasonable default, unvalidated |

Those belong in `real_frames/`, driven by captures from
`tools/grab_frame.py --watch`.

## Conventions

- Assert on stable codes (`Verdict.code`, `GuardBlock.code`), never on display
  strings, so rewording a log message cannot break a test.
- Inject time into `ClickGuard` instead of sleeping; the guard tests are instant.
- When building a frame for a non-reference window size, build it at the
  reference size and resize the **whole frame**. Pasting a full-size sprite onto
  a small canvas produces a scene that cannot occur in the real game, and the
  resulting test failure is meaningless.
