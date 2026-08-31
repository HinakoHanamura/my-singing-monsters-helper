# Real-frame regression tests

Empty on purpose. This is where perception gets validated against reality
instead of against drawings.

## What goes here

1. A small, curated set of real captures produced by
   `python tools/grab_frame.py --watch`, committed as fixtures:
   - frames containing collectible coins,
   - frames containing **no** coins (the negative baseline that makes a
     false-positive rate measurable),
   - frames from more than one island, so thresholds are not tuned to a single
     background,
   - if the game has them: coins in a non-collectible state.

2. Tests that assert measurable properties on those frames, for example:
   - every hand-labelled coin is detected,
   - no detection appears in a frame labelled as having none,
   - the `TextureBelow.measure()` distribution actually separates
     "coin above a monster" from "coin over empty ground" - if it does not, that
     rule provides no protection and must be replaced rather than retuned,
   - detections stay correct when the same scene is captured at a second window
     size, using real frames instead of a resized synthetic one.

## Why it is empty right now

No real frames have been captured yet, so any threshold committed today would be
a guess dressed up as a test. Keeping this directory visible and empty is the
honest state of the project.

Keep the fixture set small (a handful of frames, downscaled if needed). Bulk
recordings live in `captures/`, which is git-ignored.
