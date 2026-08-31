"""Self-contained minigame solvers.

Each minigame lives in its own module and is deliberately isolated from the
idle-farming engine: it borrows the same perception and action layers
(GameWindow, VisionAgent, ActionAgent) but owns its own decision loop, because
a minigame is a closed puzzle with a terminal state rather than an endless
patrol.

The modules here are split by how much they depend on pixels:

* ``grid``         - pure geometry, no image data at all.
* ``memory_game``  - pure game logic, works on opaque face keys.
* fingerprinting and card-back detection stay in the vision layer, where the
  thresholds can be calibrated against real frames.

That split is what makes the tricky part (the solver) testable without a
running game.
"""
