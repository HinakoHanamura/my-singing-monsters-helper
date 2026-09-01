# MSM Helper Design Document

**English** | [简体中文](DESIGN.md) | [Back to Main README.md](../README.en.md)

This document systematically details the architectural design, low-level Win32 system mechanics, computer vision perception contracts, empirical calibration methodology, and post-mortem analysis of critical bugs for the *My Singing Monsters* background automation helper (**MSM Helper**).

> **Development Model Statement**: This project was developed via a modern **Human-in-the-Loop (HITL)** development model.

| Role | Core Responsibilities & Contributions |
| :--- | :--- |
| **Human Developer** (Lead) | Defined engineering roadmap and milestones, established architectural invariants, formulated worst-case board construction and mathematical bound proofs, provided physical observations, and conducted live recordings and acceptance. |
| **AI Agent** (Copilot) | Engineered modern asynchronous GUI, translated state machine logic into code, constructed the 472-test headless suite, conducted runtime defect tracing and root-cause analysis, performed statistical calibration, and built diagnostic tooling. |

---

## 1. Design Principles

The primary design principle of this project is **zero intrusion on the host environment**, meaning that the automation loop must never interfere with the user's ongoing use of the computer. Four non-negotiable low-level constraints follow from this:

| Constraint | Implementation | Mechanism & Rationale |
| :--- | :--- | :--- |
| **Background Frame Capture** | `PrintWindow(hwnd, hdc, PW_CLIENTONLY \| PW_RENDERFULLCONTENT)` | Flag 3 (`PW_RENDERFULLCONTENT`) is required to read DirectX hardware-accelerated swap chains without receiving a blank black frame. |
| **Background Action Dispatch** | `SendMessage(hwnd, WM_LBUTTONDOWN / WM_LBUTTONUP, ...)` | Injects mouse messages directly into the game window's message queue without window activation. |
| **Never Move Physical Cursor** | Strictly avoid `SetCursorPos`, `mouse_event`, or `SendInput` | Never competes for physical mouse ownership; the user can freely interact with other windows concurrently. |
| **Never Steal Foreground Focus** | Strictly avoid `SetForegroundWindow` / `SetActiveWindow` | The game window can sit behind other windows without stealing focus. |

> **OS Constraint on Minimised Windows**: The game window must not be minimised. When a window is minimised, the Windows Desktop Window Manager (DWM) pauses its client-area rendering pipeline to conserve GPU resources. Consequently, no capture API can obtain valid frames. This is an operating system constraint, not a contradiction of running in the background.

Any approach that relies on maintaining the window in the foreground is considered an architectural flaw in this project.

---

## 2. Architecture & Interfaces

The system enforces a strict unidirectional layered architecture. Layers communicate solely through predefined data structures, decoupling high-level decisions from perception and OS-level primitives.

### Layer Invocations & Dependencies

| Caller Layer | Primary Module | Callee Layer | Target Module | Interaction Contract & Data Flow |
| :--- | :--- | :--- | :--- | :--- |
| **UI Layer** | `ui/main_window.py` | **Control Layer** | `core/bot_engine.py`<br>`core/minigames/memory_engine.py` | Task start/stop dispatch; asynchronously receives state updates and log signals |
| **Control Layer** | `core/bot_engine.py` | **Guard Layer** | `core/validators.py`<br>`core/click_guard.py` | Submits candidate targets for rule-chain validation and escalating backoff decisions |
| **Control Layer** | `core/bot_engine.py` | **Minigame Solvers** | `core/minigames/` | Dispatches dedicated minigame state machines and reading-order recovery algorithms |
| **Control Layer** | `core/bot_engine.py` | **Perception Layer** | `core/vision_agent.py` | Invocates `detect()` to obtain structured target candidate lists |
| **Control Layer** | `core/bot_engine.py` | **Action Layer** | `core/action_agent.py` | Dispatches validated target coordinates to trigger Win32 background message posting |
| **Perception Layer** | `core/vision_agent.py` | **Window Layer** | `core/game_window.py` | Acquires the latest valid background client-area rendered frame |
| **Perception Layer** | `core/vision_agent.py` | **Geometry Layer** | `core/geometry.py` | Maps baseline resolution (1024×768) coordinates against live window detection boxes |
| **Action Layer** | `core/action_agent.py` | **Window Layer** | `core/game_window.py` | Obtains target window handle and manages DPI thread context switching |
| **Action Layer** | `core/action_agent.py` | **Geometry Layer** | `core/geometry.py` | Transforms relative reference coordinates into live window physical click targets |

### Core Layers

| Layer | Primary Modules | Responsibility & Patterns |
| :--- | :--- | :--- |
| **UI** | `ui/main_window.py` | PySide6 dark interface: status display, live log rendering, global hotkeys. |
| **Control** | `core/bot_engine.py` | Asynchronous worker on `QThread`, running the main loop **without blocking the UI thread**. |
| **Perception** | `core/vision_agent.py` | Multi-scale template matching, HSV color segmentation; produces structured `Detection`. |
| **Action** | `core/action_agent.py` | Win32 message wrapper with Gaussian coordinate offsets and timing jitter. |
| **Geometry** | `core/geometry.py` | Scaling transformations between reference resolution (1024×768) and live window dimensions. |
| **Window** | `core/game_window.py` | Window handle lookup, GDI bitmap management, DPI thread context switching, black-padding stripping. |
| **Anti-misclick** | `core/validators.py`<br>`core/click_guard.py` | Environment rule chain, cross-frame tracking, escalating backoff cooldown, post-click verification. |
| **Minigames** | `core/minigames/` | State-machine-driven image-agnostic solver with reading-order recovery and card-face clustering. |

### Key Interface Contracts

| Interface Contract | Signature & Data Structure | Guarantees & Decoupling Value |
| :--- | :--- | :--- |
| **Perception Abstraction** | `BaseVisionAgent.detect(target_name, screenshot) -> list[Detection]` | Fully decouples state machine decisions from vision algorithms; swapping in deep-learning detectors (e.g. YOLO) requires zero upper-level changes. |
| **Structured Detection** | `Detection(box: Rect, score: float, ...)` | Mandates spatial bounding boxes and confidence scores, providing essential data for occlusion checks, priority peeling, and cross-frame verification. |

---

## 3. Perception Parameters

A central discipline of this project: **Perception parameters must never be set arbitrarily; all thresholds are derived from statistical measurements across real game recordings (`captures/`) with verified separation margins.**

### Calibrated Perception Parameters

| Parameter | Calibrated Value | Measured Evidence & Statistical Separation |
| :--- | :--- | :--- |
| `reference_size` | `(1024, 768)` | Base game resolution; anchor for all relative coordinate conversions. |
| `match_threshold` | `0.75` | Coin score distribution: fully visible icons score $0.80 \sim 0.98$ ($p_{50}=0.80, p_{95}=0.89$), dropping to $0.51$ under occlusion. $0.75$ optimizes the recall/precision tradeoff. |
| `nms_distance` | `30 px` | Spacing between adjacent plaques bottoms out at $38.6\text{px}$; cross-scale duplicate hits appear down to $26\text{px}$. $30\text{px}$ reliably suppresses duplicates without merging distinct plaques. |
| `match_downscale` | `0.5` | Reduces frame matching latency from $103\sim 108\text{ms}$ to $30\sim 48\text{ms}$ ($2.5\times$ speedup), with only $1.8\%$ miss rate and $\le 2\text{px}$ drift. Downscaling to $0.4$ doubles the miss rate to $4.5\%$. |
| Card-Back Fill (`min_fill`) | `0.70` | True purple card-back bounding-box fill ratio: $0.89 \sim 0.97$; false purple blob inside revealed cards: $0.49$. $0.70$ provides $\approx 0.20$ safety margin on both sides. |
| Card-Face Similarity | `0.75` | True matching pairs: $p_5=0.92, \text{median}=0.9827$; non-matching pairs: $\max=0.4618$. Robust gap is $+0.2333$. $0.75$ chosen conservatively due to high mismatch penalty. |
| Card-Face Stability | `0.98` | Settled card faces correlate with their previous frame at **exactly 1.000**; during flip animations, similarity drops to $0.21 \sim 0.93$. $0.98$ provides an unambiguous completion gate. |

### Negative Results (Rejected Designs)

Recording rejected approaches preserves valuable engineering context:

- **Texture Probe (`require_texture_below`)**: Intended to confirm monster texture beneath plaques. Measured across 219 real targets, values ranged $41.8 \sim 72.4$ against a threshold of $12.0$, yielding $0\%$ rejection with no negative separation. Honestly disabled rather than artificially tuned.
- **Search Region Cropping (ROI Cropping)**: Attempted to restrict matching to the island area to save pixels. Measured $105\sim 129\text{ms}$ vs. $103\sim 108\text{ms}$ baseline: non-contiguous memory slicing overhead exceeded the benefit of removing $25\%$ pixels.
- **Color-Based "Glow" Gate**: Attempted to detect card animation via green/yellow hue proxies. Failed twice: gold and amber faces (coins, XP stars) naturally fell into the hue window, deadlocking levels. Lesson: **Never gate on a proxy for something you can measure directly (inter-frame stability).**

---

## 4. Memory Minigame Algorithms & State Machine

The memory minigame presents core engineering challenges: **arbitrary non-grid layouts, wide variety of card faces, and bounded mismatch budgets.**

### 4.1 Algorithm Design
1. **Scale-Invariant HSV Back Detection**: Card back sizes vary by $1.9\times$ across levels ($63\text{px} \sim 122\text{px}$). HSV color segmentation naturally achieves scale invariance.
2. **Row-Band Clustering**: For arbitrary non-grid layouts, $y$-density clustering identifies visual rows; sorting by $x$ within rows restores natural left-to-right, top-to-bottom reading order.
3. **Runtime Face Clustering (`FaceRegistry`)**: Card crops are inset by $18\%$ and normalized to $48\times 48$, retaining color information, and dynamically grouped via Pearson correlation.

### 4.2 Mathematical Bound on Mismatches
For a board with $n$ pairs ($2n$ cards):
- Game chances formula: $\text{Chances} = \lceil 1.5 \times n \rceil$
- State machine worst-case sequence (`AB | CA | DB | EC | ...`):
  $$\text{Max Mismatches} \le 1 + (n - 2) = n - 1$$
- This leaves a $\approx 50\%$ safety margin, mathematically guaranteeing $100\%$ completion provided perception makes no false-positive match.

### 4.3 Concurrent Dual-Card Read Optimization
During the scan-first strategy, flipping two cards represents two independent reads. The engine executes: **Click Card 1 & 2 in rapid succession $\rightarrow$ Single settling wait $\rightarrow$ Synchronous dual-card extraction**, cutting settle latency by $50\%$ and compressing live reveal time to $\approx 1.10\text{s}/\text{card}$ (primarily bounded by the game's built-in flip animation duration). Trade-off: incurs at most one additional mismatch under the worst-case scenario compared to conservative matching.

---

## 5. Four Critical Bug Post-Mortems

Neither unit tests nor synthetic replays caught these issues; only live execution revealed them:

### 1. Inconsistent DPI Awareness
- **Symptom**: Under $150\%$ OS scaling, clicks landed completely off-target.
- **Root Cause**: `QApplication` initialisation marked the process as System DPI-aware, causing `GetClientRect` to return virtual scaled bounds ($1536\times 1152$) while the game rendered into unscaled $1024\times 768$.
- **Fix**: Wrapped capture and clicking in a `dpi_unaware_thread()` context manager, aligning coordinate spaces strictly.

### 2. False Purple Detection Inside Revealed Cards
- **Symptom**: Revealed purple monsters were misclassified as card backs, causing perpetual waiting loops.
- **Root Cause**: Local purple patches satisfied both HSV and aspect-ratio bounds.
- **Fix**: Added a bounding-box fill-ratio metric (`min_fill`). Real backs measure $\ge 0.89$, while internal blobs score $0.49$. Setting threshold to $0.70$ eliminated false positives.

### 3. Level Transition Race Condition
- **Symptom**: With `AABB` card order on Level 1, both pairs matched instantly in $< 1.5\text{s}$. The engine halted on `GEOMETRY_DRIFT` (detecting 6 cards on a 4-slot board).
- **Root Cause**: The engine was faster than expected; the game loaded Level 2 before the runner took its final verification capture, comparing Level 2's board against Level 1's slot table.
- **Fix**: Prioritize solver state over live captures. The loop queries `solver.is_solved()` first; if all pairs are accounted for, it transitions immediately without an unnecessary capture.

### 4. Physical Cursor Interference Hypothesis
- **Symptom**: Occasional misclicks on island buildings caused camera panning.
- **Root Cause**: A subtle $8\sim 25\text{ms}$ window exists between synthesized move and click messages. If the user rapidly moves their physical cursor, real OS `WM_MOUSEMOVE` messages penetrate the window, diverting the internal hover target.
- **Fix**: Implemented cursor movement safety gating.

---

## 6. Testing

### Test Suite Distribution

| Test Area / Subsystem | Test Count | Focus & Verification Goals |
| :--- | :---: | :--- |
| **Core Architecture** (Window / Geometry / Vision / Guard / Engine / UI) | 251 tests | Win32 message wrappers, DPI context, template matching, guard rule chains, and async UI lifecycle |
| **Minigame Solver & Reading Order** | 84 tests | State machine worst-case bound proofs, row-band clustering, and arbitrary layout order recovery |
| **Control Loop Pipeline** | 48 tests | State machine transitions, rapid level transition races, and single-turn timeout recovery |
| **Card Back Detection & Slot Tracking** | 42 tests | Scale-invariant HSV segmentation, dynamic board reconstruction, and slot drift verification |
| **Card-Face Clustering & Stability Machine** | 35 tests | Dynamic correlation clustering, settling stability gates, and concurrent dual-card reads |
| **Tooling & Encoding Recovery** | 12 tests | GB18030 mixed-encoding recovery and automated publish audit gates |
| **Total** | **472 passed** | **Runtime ~78 seconds** (headless offscreen execution) |

### Core Testing Conventions

| Convention | Implementation Strategy | Engineering Benefit & Rationale |
| :--- | :--- | :--- |
| **Full Headless Decoupling** | Runs on `offscreen` platform with pure functions and dependency injection (`FakeBoard`, `FakeClock`) | Zero game or GPU dependency; enables sub-minute deterministic execution on headless CI |
| **Assert Stable ASCII Codes** | Asserts solely on stable ASCII enum strings (`Verdict.code`, `GuardBlock.code`) | Log rephrasing and localization tweaks never break the test suite |
| **True Texture Synthesis** | All synthetic mock test assets inject Gaussian noise or structured high-frequency texture | Prevents mathematical undefinition when dividing by standard deviation in normalized cross-correlation |

---

## 7. Technical Records & Deep Dives

For raw measurements, test protocols, and detailed decision archives, refer to:

- [**`calibration.md`**](../.agents/rules/calibration.md) — Measurement distributions, rejected approaches, misclick hypothesis experiments.
- [**`minigame_memory.md`**](../.agents/rules/minigame_memory.md) — 9-level memory game empirical logs, timing breakdowns, formal state machine proofs.
- [**`workflow.md`**](../.agents/rules/workflow.md) — Windows PowerShell traps and encoding recovery procedures.
