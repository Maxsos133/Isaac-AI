# Isaac AI

A reinforcement learning agent that plays **The Binding of Isaac: Repentance+**
by running many copies of the real game in lockstep on one machine.

The end goal is an agent that plays from screen pixels alone. A telemetry mod
supplies privileged state as training scaffolding and is dropped at deployment —
the pixel student never sees it, and the critic that does is discarded.

This is a rebuild. An earlier attempt stalled, and the diagnosis was not reward
design but **throughput**: it scraped state out of `log.txt`, injected keystrokes
with `SendInput` (which needs window focus, which is the only reason it needed
three VirtualBox VMs), and stepped in real time with `sleep`. Its ceiling was
~30 agent steps/s and real sessions completed 768–15,000 steps, against the
millions this needs. Every "fix" in that project was compensating for those
three choices rather than for anything about the learning problem.

```
previous attempt     ~30 agent steps/s        768–15,000 steps per session
this one            234.8 agent steps/s       1,000,000 steps in ~100 minutes
```

## How it gets there

**Synchronous stepping.** The mod holds a TCP socket and blocks inside
`MC_POST_UPDATE` until the agent answers. The game physically cannot advance
past a tick the agent has not seen, so stale observations and missed actions do
not exist as failure modes. luasocket ships with the game; `--luadebug` unlocks
it.

**Mod-driven input.** Control goes through `MC_INPUT_ACTION`, not synthetic
keystrokes. No window focus required, so instances run unfocused and occluded —
which is what makes many-instances-per-host possible at all.

**Many instances, one host.** Steam blocks a second launch only in its client
UI; `isaac-ng.exe` with `steam_appid.txt` present runs as many copies as the
machine can hold. Isaac's logic tick is pinned at 30 Hz, so that is a hard
per-instance ceiling and throughput scales purely with instance count.

**Frozen save snapshots.** Unlock state determines the item pool, so letting it
drift would make the environment non-stationary and runs incomparable. Every
session is seeded from a pristine snapshot. Progression through unlock tiers is
an explicit dial, not something the agent does to itself mid-training.

Measured on a Ryzen 9 7950X / 32 GB / RTX 4080 at 20 instances: **469.6 game
ticks/s, 234.8 agent steps/s**, 23.5 ticks/s per instance against the 30 cap.
Sustained training runs at 160–210 steps/s depending on how often episodes
reset.

## What it learns

An episode is **one attempt at a real floor** — start room, real generated
rooms, one health pool, find the way onward.

Combat and navigation were originally trained as separate synthetic tasks. That
was abandoned, because nearly every bug the project hit came from the synthetic
scaffolding rather than from the game: a spawn exploit, wall-camping, an aim
collapse, a room-size confound, curriculum oscillation. The game generates
perfectly good content and we kept building a fake version of it and then
debugging the fake version. A floor *contains* combat, so no separate combat
task is needed.

**Entity-set policy.** Observations are a padded, masked set of entities plus a
scalar vector, an obstacle grid, and a fixed door block. Each entity is embedded
independently and pooled (mean and max), so the encoder is permutation-invariant
and count-agnostic — one policy handles one fly or six chargers. Entity
positions are encoded *relative to the player*, because the encoder pools
entities before player state is concatenated.

**Auto-tuned curriculum.** On floors the dial is not enemy count — the game
authors the encounters — but **how much of the floor counts as success**: clear
one room, then two, then more. It rises when the agent beats its target rate and
falls when it starts losing.

## Results

Each run is one variable against the last. Numbers are means over the final
200k steps of each run.

| run | change | return | success | rooms cleared |
| --- | --- | ---: | ---: | ---: |
| floor-v6 | first run where dying was actually charged (−10) | −1.88 | 0.10 | ~0.0 |
| floor-v7 + v7b | death penalty −10 → **−2** | +6.55 | 0.308 | 0.34 |
| floor-v8 | **per-head entropy ceiling** (`--entropy-target 0.5`) | +10.04 | 0.463 | 0.52 |
| floor-v9 | **semantic entity identity** (chest / pedestal / hostile / …) | +10.32 | 0.453 | 0.57 |
| floor-v11 | egocentric obstacle window | +11.9 | 0.477 | 0.57 |
| **floor-v12** | secret-room doors excluded from shaping | +11.0 | **0.498** | **0.61** |
| floor-v13 | blocked-move penalty | — | 0.429 | 0.49 |
| floor-v14 | **fresh restart**, no resume | — | 0.248 | 0.26 |
| floor-v15 | conv encoder, enemy-type embedding | — | 0.177 | 0.19 |
| floor-v16 | reward rebalance toward combat | — | 0.151 | 0.12 |
| floor-v17 | raw entity identity, creep plane | — | 0.090 | 0.08 |

**floor-v12 is the high-water mark, and everything after it is a regression.**
It is also the only run in which the curriculum ever moved (`difficulty` 0.090).
The decline is monotonic across six consecutive changes — a fresh restart, a
convolutional obstacle encoder, learned enemy identity, a reward rebalance and a
creep plane — none of which helped, while the observation grew from a 405-value
grid to 945 and performance fell sevenfold.

Part of that gap is training budget: v12 sat at the end of a resumed lineage with
~4M cumulative steps, against 1M for each fresh run. But at *matched* ~1M, the
original lineage reached 0.34 and every later fresh run came in below it. The
changes are net-negative on their own terms.

The next step is subtraction, not another feature: reconstruct v12's rewards and
observation, confirm ~0.61 reproduces, then move one variable at a time against a
baseline that is known to work. v12's checkpoint cannot be loaded — the
observation changed four times after it — so this means retraining.

Two earlier results are still worth calling out.

**The death penalty.** At −10 a single death cost more than a whole good episode
earned, and because PPO normalises advantages per minibatch, one such spike
inflated the batch deviation and squashed the small shaping differences carrying
the actual gradient. The dominant gradient became "do not engage." Dropping it
to −2 was the entire difference between a run that learned nothing and one that
learned.

**The entropy ceiling.** The entropy bonus is applied to the summed entropy of
all four action heads, so a head receiving no learning gradient gets pushed to
the ln(3)≈1.099 uniform ceiling at full strength, forever. That is the cause of
every dead action axis this project has had. Clamping each head's bonus at a
target — above it a head contributes a constant and therefore no gradient, below
it is still pushed back up — brought `shoot_x` from 1.084 (abandoned) to 0.712,
and produced the first policy here with **no dead axes**.

After that change the agent also learned an actual maneuver: it closes the
*horizontal* gap to put an enemy on its column, then fires vertically. Movement
closes the horizontal gap in 27/32 probe placements against 16/32 (chance) for
the vertical — measured by `diagnose_shoot_axis.py`, and it holds with the doors
removed, so it is enemy-driven rather than exploration-driven.

### What does not work yet

Stated plainly, because the interesting part of this project is the failures.

- **The floor curriculum has never advanced.** `difficulty` is still 0.000 after
  several million steps. The bar is "clear one room in 72% of episodes" and the
  agent is at ~46%.
- **It has never descended a floor.** `descended` is 0.0 in every run.
- **The pixel student plateaus at difficulty ~0.42** regardless of teacher
  strength (a teacher at 0.75 produced no better student than one at 0.57),
  input resolution (240×135 raised imitation agreement to 0.814 and moved
  performance not at all), or RL fine-tuning. The pipeline works end to end;
  the student does not get better. Parked, not broken.
- **Multi-room backtracking has no gradient.** `door_potential` only looks at
  the current room and returns 0.0 when nothing there is unvisited, so an agent
  stranded in territory it has already emptied is on a flat potential with
  nothing telling it which way to walk.

## Running it

Requires Windows (the harness is Win32-bound), Python 3.11+, and a Steam copy of
Repentance+.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install torch numpy opencv-python windows-capture
```

Copy `config.example.toml` to `config.toml` and point the three `[game]` paths
at your own install. Then prepare a save snapshot — File 1, Hard mode, the
unlock tier you want to train at — and put it in `saves/`.

```bash
# copy the bridge mod into the game
.venv/Scripts/python.exe -m isaac_ai deploy

# bring up a fleet and drive it with random actions
.venv/Scripts/python.exe -m isaac_ai smoke --instances 20 --steps 400

# train on real floors
.venv/Scripts/python.exe -m isaac_ai train-floor --instances 20 --steps 1000000 \
  --run-name floor-v10 --entropy-target 0.5 --resume runs/floor-v9/policy.pt
```

Metrics stream to `runs/<name>/metrics.jsonl`; checkpoints to
`runs/<name>/policy.pt`. Tests are unittest, not pytest:

```bash
.venv/Scripts/python.exe tests/test_learning.py
```

`--resume` is a *widening* transfer, not a plain resume: it copies every tensor
whose shape matches and, where a layer's input grew, keeps the old weights in
the leading columns and zeroes the new ones — so a resumed policy starts out
computing exactly what the checkpoint did, and new features earn influence by
gradient. This only holds while new features are **appended**; inserting one
shifts every later column and scrambles the transfer with no error at all.

## Diagnostics

Training curves have been wrong about this project more often than they have
been right, so most of the debugging tooling exists to answer a specific
question directly. Several need no game at all and run in seconds.

| script | answers |
| --- | --- |
| `probe_heads.py` | does a checkpoint aim? 8 directions × 2 ranges, no game |
| `diagnose_shoot_axis.py` | is a dead shoot axis a decomposition or abandonment? Reports **distinct move responses**, because a constant direction scores ~50% alignment by construction |
| `diagnose_student.py` | student vs teacher vs random at *fixed* difficulty |
| `play_encounter.py` | puts a human on the same encounters — is the task even fair? |
| `probe_entity_flags.py` | do the semantic flags resolve and fire in real play? Tallies raw variants and escalates any producing **no** flag |
| `probe_floor_reset.py` | does the episode reset route restarts and reseeds correctly, at real fleet size? |
| `probe_capture.py` / `probe_pixels.py` | capture cost, staleness, distinctness, motion |

## The failure mode this project keeps hitting

**A metric or reward counts an event rather than actual progress, the curves
read as "not learning", and the agent is quietly gaming the measure.** A dozen
instances so far, and almost every one was caught by watching the game rather
than by any metric. A representative sample:

- floor `new_room` paid per transition → the agent shuttled between two rooms
- a navigation curriculum required N *crossings* → one door, N times
- `backtrack_ratio` scored zero-room episodes as 0.0 → inactivity read as good
- summed entropy hid that a shoot axis was dead for three runs and two students
- `move_agreement` compared two *sampled* actions, capping a perfect student at
  ~0.5, so a run 41% of the way there read as "stalled at chance"
- enemies spawned relative to the player, so the agent chose where they appeared
  — worth 0.48 of win rate, and inherited by every student distilled from it
- door shaping paid **−2.885 for walking through a door**
- the −10 death penalty never fired, and `deaths` was never counted at all
- a hand-written chest-variant list missed 2 of 13 entries that were **41% of
  every pickup observed**, so the flag read zero — indistinguishable from
  "chests are rare"
- and the instruments caught it too: `probe_heads.py` synthesised a room with no
  doors, and duly reported a floor policy's movement heads as abandoned while
  the agent was visibly walking to doors
- the obstacle grid repeated the entity bug exactly: encoded in **room**
  coordinates while everything the agent aims with is player-relative, so
  moving a rock onto the line of fire changed the policy by a total variation
  of 0.019. It was reacting to how many obstacles existed, not to where any of
  them was

## Things that cost an experiment to learn

Platform facts, all empirical. Do not re-derive these.

- **`--set-stage` is not a faithful game state.** It boots into a run, but grants
  the D6 (an unlock-gated item a zero-unlock save does not have) and pins
  difficulty to Normal. A console `restart` does not clean it up.
- **Isaac's menus ignore synthetic navigation keys.** Arrow keys are ATTACK;
  menu movement is WASD; neither moves the main-menu cursor via `SendInput`, and
  `PostMessage` does nothing at all. Only SPACE registers, and only with real
  focus. Hence run entry via CONTINUE, which needs no cursor movement.
- **`AttachThreadInput` will hang the launcher.** Isaac stops pumping Win32
  messages while generating a floor and while blocked on the agent; attaching to
  a thread in that state blocks forever. Same reason `MoveWindow` hangs — use
  `SetWindowPos` with `SWP_ASYNCWINDOWPOS`, which posts instead of sending.
- **`PrintWindow` cannot capture a blocked game; Windows Graphics Capture can.**
  WGC reads the compositor's surface and returns a live frame from a blocked,
  occluded instance. Capture is free: 360.0 game ticks/s with it on against
  359.9 off.
- **A blocked game does not present**, so the newest frame is always the
  *previous* tick's. ~14ms old at the median, inside a 33ms tick. Do not go
  looking for the missing frame.
- **WGC captures the window, not the client area** — crop to the client rect or
  the title bar is 11% of your input.
- **A dead player freezes the whole fleet.** Isaac stops running mod callbacks
  once the game-over screen takes over, so that instance can never answer — and
  reads are sequential and blocking, so every instance stalls behind it for the
  full timeout. Anything stepping the env must restart on `is_dead`, which is
  visible during the death animation.
- **A crashed game is not a closed connection.** A clean shutdown ends the
  stream and `readline` returns `b""`; an access violation drops the socket with
  an RST and raises `ConnectionResetError`, which is an `OSError` and not the
  error type the fleet's handler catches. That cost one run 424,000 steps at 57%
  complete.
- **Spawning into a cleared room does not re-lock it.** `room:IsClear()` stays
  true with live enemies present.
- **`MC_POST_ENTITY_KILL` fires when the entity is already dead**, so
  `IsVulnerableEnemy()` returns false there — which silently disabled a kill
  reward for an entire 600k-step run.
- **A Lua `local` is only visible to code compiled after it.** Referencing a
  helper defined lower in the file becomes a nil global lookup, which throws
  inside the callback, skips the send, and blocks the agent forever.
- **A ramp that steps per update while its evidence arrives per episode is an
  open-loop integrator.** During distillation the student drove 90% of episodes
  and only the other 10% fed the success window; difficulty ran 0.00 → 1.00 →
  0.00 at full scale. Any auto-tuned dial needs its step rate tied to its
  evidence rate, not to the training loop's.
- **All instances share one savedata directory**, including `log.txt`, which is
  truncated per launch. Multi-instance logs are not reliable for diagnosis.

More detail in [`spike/FINDINGS.md`](spike/FINDINGS.md).

## Layout

| Path | What |
| --- | --- |
| `mod/isaac_ai/` | The bridge mod: observations, input, episode resets |
| `src/isaac_ai/bridge.py` | Socket protocol, one listener per instance |
| `src/isaac_ai/launcher.py` | Save seeding, sequential bring-up, run entry |
| `src/isaac_ai/env.py` | Vectorized environment, observation encoding, rewards |
| `src/isaac_ai/floors.py` | Floor episodes, shaping, restart/reseed routing |
| `src/isaac_ai/policy.py` | Entity-set actor-critic |
| `src/isaac_ai/ppo.py` | PPO trainer |
| `src/isaac_ai/capture.py` | Graphics Capture, cropping, frame stacking |
| `src/isaac_ai/pixel_policy.py` | Pixel actor with a privileged critic |
| `src/isaac_ai/distill.py` | Teacher-to-student distillation |
| `src/isaac_ai/windows.py` | Win32: windows, focus, keys, frame capture |
| `scripts/` | Probes and diagnostics |
| `saves/` | Frozen save snapshots, one per unlock tier |
| `spike/FINDINGS.md` | Phase 0 measurements, all empirical |
| `STATUS.md` | Current state of the work |

## Status

Phase 0 (feasibility), Phase 1 (harness) and Phase 2 (a state-based teacher over
an entity-set encoder, trained on real floors) are working. Phase 3 — distilling
that teacher into a pixels-only student with an asymmetric critic — is built and
runs end to end, but the student plateaus well below its teacher and is
currently parked.

`STATUS.md` has the current run, what is being tested, and what to look at next.
