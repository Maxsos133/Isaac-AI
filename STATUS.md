# Status — 2026-08-18

Where the work is, for picking up cold. Architecture and the platform facts that
cost live experiments live in `README.md`; this is the state of the work and
what to do next.

## Read this first

**floor-v28 clears more than one room an episode and the curriculum can finally
hold a two-room target.** The combat gate — rejected once on v22's evidence —
works on a policy that can fight.

| | cleared | seen | success | ended_idle | peak succ | peak difficulty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v12 (old high-water) | 0.56 | 1.76 | 0.445 | — | 0.775 | 0.090 |
| v27b | 0.867 | 2.56 | 0.550 | 0.453 | 0.775 | 0.120 |
| **v28** | **1.056** | **2.91** | **0.609** | **0.368** | **0.825** | **0.210** |

`rooms_cleared` above 1.0 for the first time, +22% on v27b. **Peak success 0.825
finally beats v12's 0.775**, the last record the old high-water mark still held.
`blocking -> irrelevant` reached **0.0382**, also a record, and this time
alongside the best task performance rather than in isolation. 20/20 instances
throughout.

### The gate was right the second time, for a measurable reason

`SUPPRESS_SHAPING_IN_COMBAT` was tested in v22 and came out neutral, and was
reverted. Retested on v28 it is worth +22% cleared. The difference is the policy
it was asked about:

```
v22   fires ok 25%, could not finish rooms   -> removing the door pull changed nothing
v28   kills 6.64/episode, fires ok 84%       -> converts, cleared 0.867 -> 1.056
```

v22 asked "does the door pull stop it finishing rooms" of an agent that could not
finish rooms regardless. **A null result is a fact about the policy that was
tested, not only about the change.**

**The tripwire improved rather than held.** `ended_idle` fell 0.453 -> 0.368. The
risk was that an agent with no shaping gradient inside a contested room would
stall; it fights instead. Clears-per-room-entered went 35% -> 39% mid-run,
settling at 36-37%.

**Where it is less clean:** the last million flattened. Bands ran 1.046 -> 1.085
-> 1.053 -> 1.019, and the whole-run slope is +0.0058 per 100k against the
+0.0132 measured at the 2M mark. Most of the gain landed early.

### The curriculum is no longer bouncing off the boundary

```
while target_rooms == 1 (1008 updates): success 0.572   cleared 0.97
while target_rooms == 2 ( 164 updates): success 0.614   cleared 1.17
raises above 0.72, lowers below 0.48, holds between
```

At a two-room bar it scores **0.614** — inside the hold band. In v26 the same
test scored ~0.40 and the dial collapsed within four updates. Difficulty was
above zero for **31%** of this run against v26's 4.8%, and `target_rooms` was 2
for 164 updates against 21.

**This reverses the advice recorded earlier that `--start-difficulty` is
pointless.** That was correct when success at target 2 was below 0.48: the peak
was the failure point and resuming there just walked back down. It is now wrong.
The dial ends at 0.060 not because it failed at 2 but because success sits in the
middle of the deadband and random-walks. Carrying it forward is now the
difference between starting at a one-room bar and a two-room one.

### Next

```bash
.venv/Scripts/python.exe -m isaac_ai train-floor \
  --instances 20 --steps 3000000 --run-name floor-v29 \
  --resume runs/floor-v28/policy.pt --start-difficulty 0.12 \
  --entropy-target 0.5 --min-action-prob 0.05 --relaunch-crashed
```

First run to begin at a two-room target. `rooms_cleared` is the honest column
through a curriculum change; the bar to beat is **1.056**. Watch whether
`difficulty` climbs past 0.21 — that needs success above 0.72 at target 2, which
it has touched (peak 0.825) but not held.

3M rather than 5M because the last million of v28 flattened. If v29 flattens too
while `difficulty` stays put, budget is done on this configuration and the next
question is what caps clears-per-room-entered at ~37%.

---

**Superseded: floor-v26 broke both all-time records, `rooms_cleared` 0.836 and
difficulty 0.120 — the first time anything passed floor-v12.**

| | floor steps | cleared | seen | success | peak succ | peak difficulty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v12 (old high-water) | ~4M | 0.56 | 1.76 | 0.445 | 0.775 | 0.090 |
| v22 | 7.5M | 0.528 | 1.94 | 0.404 | 0.700 | 0.000 |
| v25 | 6.0M | 0.588 | 2.15 | 0.444 | 0.750 | 0.090 |
| **v26** | **9.0M** | **0.836** | **2.41** | **0.534** | **0.775** | **0.120** |

`rooms_cleared` is 49% above v12 and 58% above v22. Peak success equals v12's
0.775; peak difficulty **exceeds** it. Nothing changed between v25 and v26 — same
configuration, 3M more steps.

### Two readings here were wrong, both from single checkpoints on a dip

**The halfway call.** At 1.5M into v26 the bands read 0.695 / 0.722 / 0.725 /
0.735 / 0.691 / 0.571 and it was reported as "flat, near this configuration's
ceiling, spend the next run on something else". The remaining 1.5M went 0.722 /
0.801 / 0.825. The 0.571 was a trough, not a plateau.

**The aim decay.** After v25 this file recorded "the aim decays, slowly rather
than not at all". It does not:

```
                combat-v13   v24    v25    v26 mid   v26 final
fires ok               88%   88%    59%        59%     27/32 84%
distinct moves           4     3      2          5           5
aligns y             31/32 21/32  15/32      20/32       22/32
```

v25's 59% was a trough too. **`fires ok` oscillates between roughly 59% and 88%,
and a single-checkpoint reading of it is unreliable** — which is what produced
both wrong calls. Read it as a band across several checkpoints, or not at all.

### What is not driving this

`blocking -> irrelevant` is **0.0058** at v26, down from v24's record 0.0357 and
below the 0.010-0.029 band most runs sit in. The agent is clearing nearly twice
as many rooms as v25 while reading obstacles *less*. Taken with the kiting
verdict — wall-camping beats kiting 0.40 to 0.07 — obstacle awareness is not what
produces the gains here, and three runs were spent on the assumption that it was.

### Fixed: the floor curriculum no longer resets on resume

`train` has had `--start-difficulty` since the combat curriculum existed;
`train-floor` never got it, so the floor dial re-climbed from 0.000 on **every**
resume in this lineage — v19b, v20, v21, v22, v23, v24, v25, v26. v25 reached
0.090 and v26 began again at zero.

It cost little so far only because `target_rooms` maps anything under ~0.13 to 1,
so the task never actually changed. **That stops being true now**: v26 peaked at
0.120, and 0.13 is where the target becomes 2 rooms. `train-floor` now accepts
`--start-difficulty`, and the next resume should carry the dial forward or it
throws away the first real curriculum advance this project has had.

### Next

```bash
.venv/Scripts/python.exe -m isaac_ai train-floor \
  --instances 20 --steps 3000000 --run-name floor-v27 \
  --resume runs/floor-v26/policy.pt --start-difficulty 0.12 \
  --entropy-target 0.5 --min-action-prob 0.05
```

Budget is still paying — 0.588 -> 0.836 over 3M with no change — and this is the
first run that starts with the dial where the last one left it. Expect
`success_rate` to fall when `target_rooms` reaches 2: the bar moves from "clear
one room" to "clear two", and `rooms_cleared` is the column that stays honest
through it.

---

**Superseded: floor-v25 was the best run this project had produced, and the floor
curriculum moved for only the second time ever.**

| | floor steps | cleared | seen | success | peak succ | peak difficulty |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v21 | 6.5M | 0.470 | 1.99 | 0.380 | 0.600 | 0.000 |
| v22 | 7.5M | 0.528 | 1.94 | 0.404 | 0.700 | 0.000 |
| **v25** | **6.0M** | **0.588** | **2.15** | **0.444** | **0.750** | **0.090** |
| v12 (old high-water) | ~4M | 0.56 | 1.76 | 0.445 | 0.775 | 0.090 |

It beats every previous run at *less* budget, and the final band (0.598) is the
highest of the run — it had not plateaued. `difficulty` reached **0.090**,
equalling floor-v12's all-time high and only the second time the dial has ever
left 0.000. One instance crashed at some point and the fleet carried on at 19,
exactly as the bridge fix intends.

### The aim decays — slowly, not never

This corrects what was written here after v24.

```
                   combat-v13     v24 (+1M floors)   v25 (+5M more)
fires ok           28/32  88%      28/32  88%         19/32  59%
distinct moves          4               3                  2
blocking -> irrelevant  0.0037          0.0357             0.0104
```

"Floors did not decay the aim" was measured over one million steps and was true
there. Over six it is false: `fires ok` erodes to 59% and the record obstacle
sensitivity falls back into the 0.010-0.029 band every other run has occupied.

**Which makes the result more interesting, not less: v25 posted the best numbers
in the project's history while actively losing the capability that produced
them.** The obvious question is what it does if the aim is held.

Do **not** read the doors-removed control (24/32, 5 distinct against 19/32 and 2)
as evidence combat behaviour is hiding under the door pull. That comparison was
over-read three times already; a targetable door is present on 88-97% of real
steps, so a doorless room is out-of-distribution and the row is a hint at most.

### Next

```bash
.venv/Scripts/python.exe -m isaac_ai train-floor \
  --instances 20 --steps 3000000 --run-name floor-v26 \
  --resume runs/floor-v25/policy.pt \
  --entropy-target 0.5 --min-action-prob 0.05
```

Still climbing at the stop, and `difficulty` moving is a qualitative change worth
observing before altering anything: `target_rooms` rises with it, so for the
first time this agent faces a goal that moves.

**The decision this sets up.** If `fires ok` keeps sliding while `cleared`
flattens, the answer is to stop letting aim erode — re-warm from `combat-v13`
periodically, or interleave combat episodes with floor ones. That is a real
experiment with a measured motivation, and it is the first candidate that is not
another guess at perception or coefficients.

---

**The agent can aim now. Combat pretraining worked, and the two questions that
blocked it for six runs are both settled.**

| | fires ok | aligns x | aligns y | `blocking -> irrelevant` | cleared |
| --- | ---: | ---: | ---: | ---: | ---: |
| floor-v22 (7.5M floors) | 8/32 (25%) | 16/32 | 7/32 | 0.0115 | 0.528 |
| **combat-v13** (1M combat) | **28/32 (88%)** | **27/32** | **31/32** | 0.0037 | — |
| **floor-v24** (combat-v13 + 1M floors) | **28/32 (88%)** | 16/32 | 21/32 | **0.0357** | 0.341 |
| floor-v19 (1M fresh floors) | 8/32 | 16/32 | 9/32 | 0.0072 | 0.280 |

At **matched floor budget** v24 beats a fresh floor run on every column
(0.341 / 1.73 / 0.266 against 0.280 / 1.44 / 0.242) while carrying aim no floor
policy has ever had. It trails v22 only because v22 has 7.5x the floor training.
Slope +0.0102 per 100k, the steepest of any recent run, and `rooms_seen` was
still climbing at the stop.

**Two predictions were wrong, both favourably.** Floors did not decay the aim —
`fires ok` is *identical* after a million steps of door shaping. And floors
*taught* the obstacle sensitivity the isolated task could not: combat-v13 arrived
at 0.0037 and v24 finished at **0.0357**, the highest ever recorded here and
outside the 0.010-0.029 band every prior run occupied.

### Why combat worked this time

`train` never accepted `--entropy-target` or `--min-action-prob`. Every combat
teacher this project ever had — including combat-v6, whose 0.73 made isolated
combat look promising and whose exploit correction to ~0.40 made it look dead —
was trained with nothing stopping an ungradiented head being dragged to the
ln(3)=1.099 uniform ceiling. That is almost certainly why combat-v5 through v7
could not shoot horizontally for three consecutive runs. With the flags wired in,
combat-v13 ended with all four heads alive (`shoot_x` 0.61, `shoot_y` 0.50,
min probs 0.087 and 0.064 against a 0.05 floor) and difficulty **0.42**.

That 0.42 is level with the honest post-exploit-fix ~0.40 of combat-v8..v12 — but
those were in bare rooms and this one had a median of **8 obstacles per room**.
Same difficulty on a harder task.

### The wall stance is correct play, measured at last

`diagnose_kiting.py` existed to settle "is standing at a wall and trading hits
actually wrong", was argued three times, and had never been run. Hand-written
controllers against the teacher on the same fleet at fixed difficulty:

| difficulty | teacher | kite | wall | random |
| --- | ---: | ---: | ---: | ---: |
| 0.30 | 0.83 | 0.12 | 0.79 | 0.04 |
| 0.45 | 0.29 | 0.04 | 0.25 | 0.00 |
| 0.60 | 0.17 | 0.04 | 0.17 | 0.00 |
| **overall** | **0.43** | **0.07** | **0.40** | 0.01 |

**Wall-camping beats kiting nearly 6 to 1**, and kiting is barely above random.
The teacher tracks the hand-written wall controller at every difficulty, so it
rediscovered the best simple heuristic available at base 3.5 damage with no
items. Watching v13 play — hugging a wall, shooting across, never fleeing — is
watching it be right.

**This reframes the retreat work, uncomfortably.** Every floor policy's dominant
response to an enemy has been to *increase distance*, run after run, `aligns_y`
far below chance. That is kiting, which measures 0.07. The floor agent has been
executing the losing tactic the whole time — and floor-v23's obstacle rays, built
to make retreat succeed, worked mechanically and paid nothing because **making a
losing tactic more reliable is worth nothing.** That is a cleaner account of v23
than the one given at the time.

It also puts a question mark on the cornering result. Blocked retreat is 5.25x
enriched at death, but if retreating is losing play then that may mark the deaths
where the agent *chose* badly rather than deaths *caused* by rocks. Nothing
distinguishes those yet, so obstacle perception should not be treated as the
lever until something does.

### Next

```bash
.venv/Scripts/python.exe -m isaac_ai train-floor \
  --instances 20 --steps 3000000 --run-name floor-v25 \
  --resume runs/floor-v24/policy.pt \
  --entropy-target 0.5 --min-action-prob 0.05
```

v24 is the best trajectory available: best aim ever measured on floors, best
obstacle sensitivity ever recorded, steepest slope, and `rooms_seen` still
climbing at 1.77 against v22's 1.94. What it lacks is budget — it has 1M of floor
training against v22's 7.5M. Give it the budget before changing anything.

### One more instrument trap, for the list

`probe_room_contents.py` first reported 200 rooms with zero obstacles and no
variation. **`_exchange_all` sends and receives but does not refresh
`env._latest`** — only `_prime()` at reset and each env's own `step()` loop do —
so the probe read one cached reset observation 200 times. It was caught only
because "every real Basement room is bare" is not believable, which is not a
control. The probe now counts **distinct room indices**, so a jump that never
happens can never again masquerade as rooms that are all alike.

Corrected, the same probe settled the combat-obstacle question: start room 0
interior obstacles every time, jumped rooms a median of 8 and mean 12.3 across 27
distinct rooms, with `ROOMSHAPE_1x1` on **160 of 160** — so the 4x-area confound
that sank the previous room-jump attempt is gone, not merely reduced.
`curriculum.py` now sends `new_room: True`.

---

**Superseded by the above, kept for the reasoning: budget was the only thing that
ever moved `rooms_cleared`, and three targeted changes each failed the same
test.**

| run | steps | cumulative | cleared | note |
| --- | ---: | ---: | ---: | --- |
| v19b | 1M | 2.0M | 0.29 | reconstructed v12 baseline |
| **v21** | 4.5M | **6.5M** | **0.470** | **no change — pure budget** |
| v22 | 1M | 7.5M | 0.528 | door shaping gated off during fights |
| v23 | 1M | 8.5M | 0.484 | eight obstacle rays in the scalars |

**v21 settled the plateau question: there wasn't one.** Held at the v19b
configuration for 4.5M steps it climbed monotonically 0.29 -> 0.47, no decay, no
instability, `still_rate` flat at 0.048, and it was still rising at +0.0089 per
100k when stopped. The "opens strong then decays" shape from v19 and v20 did not
repeat. Peak success 0.600, the best since v13. Against the original lineage it
is behind at matched budget (0.47 at 6.5M against v9b's 0.61 at 4M) — same
direction, roughly half the slope.

**That climb rate is now the bar every change has to beat**, and it is why v22
and v23 both count as failures despite v22 posting a higher number than v21:

```
v22   ended 0.528   bar was ~0.559   own slope +0.0091 (identical to baseline)
v23   ended 0.484   bar was ~0.619   own slope +0.0026 (worse than baseline)
```

### v22: the door reward was not suppressing combat

`probe_door_pull.py` first established the premise was real, and corrected a
claim STATUS had carried for several runs. **"Isaac locks the doors during a
fight and `door_potential` skips locked doors" is false.** Of 92,881 doors
observed, **77.5% were shut but only 0.4% locked** — combat bars a door without
locking it, and `door_is_targetable` deliberately does not check `open`. So on
**97.1%** of combat steps the shaping was pulling toward a door at a mean
potential of **0.728**, *higher* than in a quiet room, across the 70.7% of the
agent's life spent in contested rooms.

Gating the local door term off while enemies are alive (`floor_potential`,
`SUPPRESS_SHAPING_IN_COMBAT`) changed nothing that mattered:

- `ended_idle` 0.391 against v21's 0.398. Removing the entire navigation gradient
  from 70% of steps changed idling **not at all** — so the agent was never using
  door shaping to decide what to do inside a fight, which is also why removing it
  did so little.
- The mechanism test failed. The premise was that combat behaviour visible in the
  doors-removed control would surface once the pull was gone. Instead the
  *control* collapsed: v21 read 28/32 fires-ok with doors blanked, v22 read
  16/32. The gap narrowed from the wrong end.

Reverted, kept switchable, reasoning recorded at the constant.

### The doors-removed control is out-of-distribution and was over-read

`diagnose_shoot_axis.py`'s `doors=none` row was cited repeatedly as evidence that
combat behaviour was hiding under the door pull. It is not evidence of anything
about real play: `probe_door_pull.py` measured a targetable door present on
88-97% of steps, so **a room with no doors never occurs**. Blanking all 104 door
values puts the policy in a state it has never seen. This is the
`probe_heads.py` doorless-room trap, one level up, and it survived three
messages of being quoted as a finding.

### v23: the obstacle information is not ignored because of its form

`probe_input_sensitivity.py` (new, no game needed) measures how much each input
block moves the action distribution. On v22 it found the obstacle blocks were the
weakest inputs in the network by a wide margin — ego window **0.013** jacobian
per value against doors' **0.106** and the entity block's **0.194**.

The theory was that what the policy reads is *relational and low-dimensional*
(doors: 13 features of direction and distance; scalars: 21 hand-built
quantities), while the ego grid is 243 raw booleans. So obstacles were given the
same form: eight ray-cast distances to the nearest blocking tile, one per
movement direction, appended to the scalars. Warm start verified exact —
`25 copied, 1 widened (21->29), 0 fresh`, and scrambling the rays moved the
logits by 0.000e+00.

**The rays came out as the weakest input in the network**, below even the raw
grid they replaced:

```
boxed in on all eight sides vs wide open : TV 0.0283
blocked right vs blocked left            : TV 0.0164
jacobian per value  rays 0.012 | ego 0.019 | doors 0.170 | entities 0.473
```

So the form was not the problem either.

**What v23 did change:** `blocked_rate` fell to **0.058** and `ended_died` to
**0.495**, both the lowest of any run, while `ended_idle` rose to 0.503 and
`still_rate` spiked to 0.116 before settling. The agent got better at not being
blocked and not dying, and spent it on standing still rather than on clearing.

That is the same trade v20 made. **Give this agent a way to reduce risk and it
takes it, because clearing rooms is still the unprofitable option** — navigation
nets ~2.3x combat and nothing in v21, v22 or v23 touched that.

---

**Correction, and it overturns the summary below: obstacles *do* cause deaths.**
`diagnose_death_cause.py` measured what dealt the fatal damage — enemies 71.4%,
projectiles 20.0% — and that was written up as "enemies kill it, the room does
not." Damage source is not cause. `diagnose_cornered.py`, 54 deaths at 20
instances under v19b:

| population | n | free neighbours | 3x3 solid | 5x5 solid | **blocked retreat** | recent blocked |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 49,717 | 5.23 | 2.77 | 10.26 | **12.7%** | 8.9% |
| taking damage | 181 | 4.57 | 3.43 | 11.54 | **25.8%** | 18.1% |
| **at death** | 54 | **4.13** | 3.87 | 12.61 | **41.2%** | 26.1% |

**Blocked retreat is 3.24x enriched at death** — 41.2% against a 12.7% baseline.
The agent dies with about one fewer escape route than normal (4.13 vs 5.23) and
more solid tiles packed around it.

**The monotonic gradient is what makes this real rather than a small-sample
artefact.** Every column moves the same direction across baseline -> damaged ->
dead, 12.7% -> 25.8% -> 41.2%. Noise does not order itself. The 54 deaths are
still a small base and the effect size, not the direction, is what deserves
another look.

So the mechanism is: **the agent's only defensive tactic is retreat, and retreat
fails about 40% of the time because of geometry it cannot see.** It is already
trying to flee — the shoot-axis probe has shown its dominant movement response to
an enemy is to increase distance, `aligns_y` far below the 50% chance floor, run
after run. It just walks into a rock while doing it, and an enemy finishes it.

### What that rules out

**An empty-room combat task is the wrong setting**, and the objection raised
against it was correct. Training the fight where retreat always succeeds teaches
a tactic that fails in exactly the 41% of cases that kill.

**And `blocked_move` is probably not the fix either, despite now looking
well-motivated.** It was tried as a clean single variable in floor-v13 — the only
change, resumed from v12 — and cleared fell 0.61 -> 0.49. The likely reason it
failed then still holds: `blocking -> irrelevant` has never exceeded 0.029 and
currently sits at 0.008, so the agent cannot predict *which* steps will be
blocked. Charging for an outcome it cannot foresee adds variance, not gradient.

The honest position is that the obstacle finding is real, important, and does not
yet have a lever attached to it. A penalty for being blocked has failed once; the
missing capability is representing where obstacles are relative to a retreat
direction, and no run has moved that number.

---

**The other blocker, stated as a number for the first time: navigation nets 2.2x
what combat does, so declining fights is correct play and the agent is doing it
correctly.** floor-v20 raised `kill` 0.50 -> 1.50 to fix exactly this and did not
close the gap. Stopped at 635k of 1M, below the baseline it was meant to beat.

Per episode, v20's last 100k, straight from the reward decomposition:

```
navigation   door_shaping +5.55 + new_room +1.50            = +7.05   no cost at all
combat       kill +4.34 + dealt +3.80 + room_clear +0.52    = +8.65
             damage_taken -4.33 + floor_death -1.16         = -5.49
             ----------------------------------------------------------
             NET navigation +7.05      vs      NET combat   = +3.16
```

Combat's *gross* is the larger of the two. `damage_taken` and `floor_death` eat
**63%** of it, while walking to a door costs nothing. Tripling `kill` moved
combat's net from roughly +1.5 to +3.2 and left navigation twice as profitable,
so it never changed the ranking of the two strategies.

### What v20 did, and how it failed

| | v19b final | v20 first 100k | v20 final 200k |
| --- | ---: | ---: | ---: |
| `rooms_cleared` | 0.293 | **0.432** | **0.231** |
| `success_rate` | 0.253 | 0.348 | 0.220 |
| `still_rate` | 0.080 | 0.083 | 0.056 |

It opened well above v19b and decayed past it, trend **-0.039 cleared per 100k**.
That is the second consecutive run with this shape: v19 peaked at 500k and
collapsed by 1M, v20 peaked in its first 100k.

**It is not the failure that was being watched for.** `still_rate` *fell* all
run, so nothing was farmed standing still. The decay is disengagement:

```
                100k    600k
kills per room  2.08 -> 1.86     (flat)
clears per room 0.246 -> 0.099   (down 60%)
ended_died      0.705 -> 0.566
ended_idle      0.295 -> 0.434
damage dealt    +4.95 -> +3.82
damage taken    -5.07 -> -4.33
```

The agent kills about the same number of enemies per room throughout and
**finishes fewer and fewer of them**, while dying less, idling more, and both
dealing and taking less damage. It learned to touch fights and leave.

### Why not just raise `kill` again

Each obvious move has a documented failure in this repo:

- cutting `door_shaping` — killed nav-v1, whose policy gradient came out ten
  times smaller than the entropy bonus
- `damage_taken` at -2.50 — tried, reverted, made encounters net-negative and
  the dominant gradient "do not engage"
- inflating combat gross — what let v16 farm `damage_dealt` standing still,
  `still_rate` 0.042 -> 0.223

And the probes say the deeper reason: 2-4 distinct move responses with `aligns_x`
at or below chance. **Combat pays badly because the agent is bad at it.** No
coefficient makes a losing fight profitable.

### Where that leaves the argument

Read against the last six runs, this reframes them. Every fix worked and none
moved `rooms_cleared`:

| fixed | evidence | cleared |
| --- | --- | ---: |
| reward SNR | v19 reversed a six-run decline | 0.28 |
| dead action axes | `min_prob_shoot_x` 0.024 -> 0.122 | 0.29 |
| enemy response | 5 distinct moves, 62% alignment — records | 0.29 |
| stranding | `exhausted` 0.326 -> 0.093 | 0.29 |
| wall-grinding | `blocked_rate` 0.114 -> 0.075 | 0.29 |
| hazards | ruled out: 91% of deaths are enemies | — |
| combat reward | `kill` 0.50 -> 1.50, net still 2.2x behind | 0.23 |

`rooms_cleared` has sat at 0.23-0.31 across all of it. The one thing never
addressed on floors is **the ability to aim**, and this project has already
demonstrated it is learnable in isolation: `combat-v6` reached curriculum
difficulty **0.73** on the isolated encounter task, against ~0.40 honest
post-exploit-fix. `ActorCritic` is shared between the two tasks, so a combat
policy warm-starts a floor policy directly — that is what
`--resume runs/combat-*/policy.pt` was built for.

That was the recommendation at the end of floor-v14 ("retrain combat from
scratch on the current observation, then warm-start floors from it"), and it was
never done — v15 through v20 went to perception and reward instead. Six runs
later, perception and reward are both measured as fixed and the number has not
moved.

---

| category | share of damage | share of killing blows |
| --- | ---: | ---: |
| **enemy contact** | **71.9%** | **71.4%** |
| **projectile** | **19.1%** | **20.0%** |
| creep | 3.9% | 5.7% |
| fire | 2.0% | 0.0% |
| explosion | 1.2% | 0.0% |
| spikes | **0.0%** | 0.0% |
| other | 2.0% | 1.4% |
| **environmental total** | **7.0%** | **5.7%** |

91% of the damage and 91% of the deaths are enemies and the things they shoot.
Cross-checks against training: 70 deaths over ~100 episodes is a 70% death rate
against v19b's logged `ended_died` 0.672, and the mod reported no unresolved
damage flags.

**So obstacle *hazard* perception is not the lever, and neither the plane-3
(`damaging`) change nor `blocked_move` addresses what is actually killing it.**
That closes the question v14 through v17 were spent on from the other end: the
room is not the threat.

Two limits on that conclusion, both real:

- **`spikes` at exactly 0.0% is not trustworthy on its own** — it is the
  project's signature ambiguity, a zero that could be "never touched one" or
  "flag never fires". It is *bounded* rather than resolved: `fire`, `explosion`
  and `creep` all fired, so the flag path works, and any spike damage that
  missed its flag would fall through to `other`, which is 2.0%. Environmental
  cannot exceed ~9% however that zero is explained.
- **This measures what damages the agent, not what stops it clearing a room.**
  Obstacles could still block line of fire and cap the kill rate without ever
  appearing here. What is ruled out is hazard *avoidance* as the lever, not the
  grid entirely.

**Contact damage at 72% is the specific finding.** The agent aims better than it
ever has (5 distinct move responses, 62% horizontal alignment) and still absorbs
most of its damage from enemies physically reaching it. It is not failing to
find enemies; it is failing to kill them before they arrive, or to avoid them
when they do.

Worth pricing before touching a coefficient: under v12's rewards a 10 hp gaper
pays `kill` 0.50 + `damage_dealt` 1.00 = **+1.50** and one contact hit costs
**-1.00**, so killing it while taking two hits is **-0.50**. That is the same
arithmetic v16 correctly identified and then over-corrected by moving four terms
at once. `kill` alone is the one term that requires finishing something and
cannot be farmed standing still — `damage_dealt` is the farmable one, and v18
established that 0.10 is the right value for it.

---

**`floor-v19b` finished. Aiming is the best it has ever been measured and it
bought nothing. Obstacles were the remaining suspect, and the probe above ruled
them out.**

The second million fixed everything it was supposed to and moved the task metric
by 0.01:

| | v19 (1M) | v19b (2M) |
| --- | ---: | ---: |
| `min_prob_shoot_x` | **0.024** (dying) | **0.122** (rescued) |
| distinct move responses / 32 | 1 | **5** — project record, beats v8's 4 |
| horizontal alignment | 16/32 (chance) | **20/32 (62%)** — above chance |
| `exhausted` | 0.326 | **0.093** |
| `blocked_rate` | 0.114 | **0.075** |
| `rooms_seen` | 1.44 | 1.59 |
| **`rooms_cleared`** | **0.28** | **0.29** |
| `blocking -> irrelevant` | 0.0040 | 0.0078 |
| `ended_died` | 0.571 | **0.672** |

`--min-action-prob 0.05` did exactly its job: the shortfall penalty fired against
the 0.024 it inherited, pushed it back to 0.11-0.13, and held every action alive
for the whole run. Enemy responsiveness followed and is now the highest ever
recorded here — and it is enemy-driven, not door-driven: with doors removed the
probe reads 6 distinct and 19/32.

**And clearing did not move.** Against the original lineage at matched cumulative
budget, recomputed identically:

| run | ~cum | cleared | seen | success |
| --- | ---: | ---: | ---: | ---: |
| v7 | 1M | 0.20 | 1.24 | 0.175 |
| v7b | 2M | **0.33** | 1.47 | 0.295 |
| v8 | 3M | 0.52 | 1.60 | 0.464 |
| v9b | 4M | 0.61 | 1.79 | 0.446 |
| v19 | 1M | **0.28** | 1.44 | 0.242 |
| v19b | 2M | **0.29** | 1.59 | 0.253 |

v19 started ahead of v7 and v19b finished behind v7b. The original lineage gained
**+0.13** on its second million; this one gained **+0.01**. (v8's jump to 0.52 is
not budget — that is the entropy ceiling, which v19 already has.) The trend over
v19b's last 400k is +0.024 cleared per 100k, so it is still creeping up, but
nothing like the original slope.

### What this rules out

Every hypothesis this project has spent a run on is now fixed or best-ever, and
the task metric is flat:

- dead action axes — rescued, all four alive
- enemy blindness — 5 distinct responses, 62% horizontal alignment
- stranding — `exhausted` 0.326 -> 0.093
- wall-grinding — `blocked_rate` 0.114 -> 0.075, without the penalty
- reward SNR — fixed in v19, and it is what reversed the decline

**The one measure that has never moved is obstacle use.** `blocking ->
irrelevant` is 0.0078, back inside the 0.010-0.028 band every run has occupied,
after touching 0.0287 at v19's 500k and not holding it. Meanwhile `ended_died` is
the highest of any run and rising.

So: the agent aims better than it ever has, navigates better than it ever has,
dies more than it ever has, and clears the same 0.29. **Do not spend the next run
on aiming.**

### Next: measure the cause of death before picking a lever

The tempting move is to add obstacle information — plane 3, a better encoder.
**That is exactly what v14 through v17 did and it cost four runs.** The agent
already has 405 room-absolute and 243 egocentric obstacle values and reads
neither. Perception cannot beat an absent gradient.

What is missing is an instrument: nothing here reports **what kills it**.
`diagnose_death.py` is about restart mechanics, not cause of death. Deaths are
67% of episodes and the single largest reward term, and the project is about to
choose between "cannot fight" and "cannot avoid hazards" with no measurement
either way. Build that probe first.

---

**`floor-v19`, the v12 reconstruction, finished 1M and reversed the decline. It
is the best fresh run this project has produced.** Final 200k:

| | cleared | seen | success | ended_died | blocked | peak succ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| v7 (fresh 1M) | 0.20 | 1.24 | 0.175 | — | — | 0.325 |
| v14b | 0.24 | 1.37 | 0.231 | 0.532 | 0.074 | 0.350 |
| v15 | 0.24 | 1.39 | 0.200 | 0.505 | 0.062 | 0.350 |
| v16 | 0.13 | 1.16 | 0.139 | 0.570 | 0.069 | 0.300 |
| v17 | 0.11 | 0.99 | 0.110 | 0.492 | 0.084 | 0.250 |
| v18 | 0.14 | 1.10 | 0.139 | 0.474 | 0.082 | 0.275 |
| **v19** | **0.28** | **1.44** | **0.242** | 0.571 | 0.114 | **0.450** |

Every row recomputed the same way (mean over each run's own final 200k), so they
are comparable to each other even where they differ slightly from the older table
further down. v7's end-reason columns are blank because that logging landed in
v9b.

**The reward-SNR diagnosis is supported.** Throughout the run, against v18 at
matched steps: `value_loss` roughly half, `policy_loss` 40-80% larger in
magnitude, `clip_fraction` ~50% higher. Advantages are normalised, so a larger
`policy_loss` means the chosen actions correlate better with advantage — the
quantity the argument predicted would rise.

**Correcting the bar that was set before the run:** it was stated as "beat
v7+v7b's 0.34 at matched budget", which conflated budgets — v7+v7b is ~2M
cumulative. At matched *fresh 1M* the bar is v7's own 0.20-0.26, and v19 cleared
it. 0.34 is what a second million is for.

### The finding: responsiveness peaked at 500k and was traded away

Probed on three checkpoints of the same run, same instrument:

| | 222k | 500k | 1M |
| --- | ---: | ---: | ---: |
| distinct move responses / 32 | 2 | **3** | **1** |
| `blocking -> irrelevant` | 0.0072 | **0.0287** | **0.0040** |
| summed entropy | 3.97 | 3.29 | 2.75 |

0.0287 at 500k is the highest this project has ever recorded — the range across
every prior run is 0.010-0.028, against an untrained floor of 0.0001. By 1M both
measures had collapsed back to constant-policy values while `rooms_cleared` was
still rising.

**A dead action formed, and it is the likely cause.** Final bands:
`min_prob_shoot_x` **0.024**, `min_prob_move_y` **0.043**. v19 ran *without*
`--min-action-prob` on the reasoning that v12 predated it. That was a mistake:
the floor is confirmed inactive whenever the probabilities are healthy — measured
on v18, min probs 0.13-0.19 against a 0.05 floor produce no shortfall and no
gradient — so it costs nothing when not needed and is the exact instrument built
to stop this. It is the v13 pathology, one level down in the shoot heads.

**`blocked_rate` 0.114 is the highest of any run**, as predicted: `blocked_move`
is 0.0 in this configuration and v12 itself was pinned against geometry on 18.6%
of its move attempts.

**Deaths are also the highest of any run** (0.571), so the backing-off-while-
firing behaviour — visible on screen and confirmed by probe, movement closing the
gap on 5-7 of 32 placements against a 50% chance floor — is not paying
defensively. It engages more and dies more.

### Next

```bash
.venv/Scripts/python.exe -m isaac_ai train-floor \
  --instances 20 --steps 1000000 --run-name floor-v19b \
  --resume runs/floor-v19/policy.pt \
  --entropy-target 0.5 --min-action-prob 0.05
```

This mirrors v7 -> v7b, which is how the original lineage went 0.26 -> 0.34, and
it tests the budget hypothesis on the one configuration now known to work. The
action floor is insurance rather than a second variable: inactive unless the
pathology recurs.

**Then `blocked_move` back on, alone.** It targets the measured defect — a
blocked step costs the -0.002 step penalty and nothing else — and `blocked_rate`
0.114 says the defect is live. After that, grid plane 3 (`damaging`) alone, which
is the only dropped plane that costs information rather than precision: a spiked
rock currently reads as an ordinary rock.

---

**`floor-v18` was stopped at 801k of 1M.** The reconstruction described below is
what became v19.

v18's one change against v17 — `damage_dealt` 0.30 -> 0.10 — **worked, and was
not the binding constraint.** `still_rate` fell 0.136 -> 0.078 at matched steps
and the agent dealt 72% *more* damage while being paid a third as much for it, so
cutting the farmable term reduced parking rather than fighting. Headline metrics
did not move: 0.14 cleared over the final 200k, `difficulty` 0.000 throughout.

### The finding that changed the diagnosis

**The observation reaches the trunk. The action heads throw it away.** Six runs
have been spent on perception on the assumption that the agent could not see;
that assumption is now measured and wrong.

Probed on v18's checkpoint at 665k, with no game running:

```
entity encoding, enemy-right vs enemy-left    L1 3.59    input is clearly different
entity branch output                          45% change  encoder responds strongly
trunk representation                          14% change  survives to the trunk
shoot_x logits                                 0.008      head discards it
```

The heads were still near their `gain=0.01` initialisation after 665k steps
(mean |W| 0.013-0.019), which is why the logits span only about ±1 and the
softmax stays near uniform. Both regularisers were confirmed *inactive* and so
cannot be the cause: `entropy_target` 0.5 against heads at 0.94 earns a clamped
constant and no gradient, and `min_action_prob` 0.05 against min probs of
0.13-0.19 has no shortfall to penalise.

What the policy actually was, invariant across every state tested:

```
move_x   [0.65 left   0.08 still  0.27 right]     prefers left  2.4 : 1
move_y   [0.19 up     0.25 still  0.56 down]      prefers down  3.0 : 1
shoot_y  [0.77 up     0.12        0.11    ]       shoots up 77%, always
```

Walks down-left into the bottom wall and shoots up. **This was reported by eye
first and the probes reproduced it exactly** — the ninth time watching the game
beat reading the curves, and worth the reminder that it is still the fastest
instrument in the project.

Sensitivity, as total variation:

| perturbation | v18 @ 665k | untrained |
| --- | ---: | ---: |
| which door the room has | 0.093 | — |
| enemy right -> enemy **left** | 0.026 | — |
| rock moved onto the line of fire | 0.011 | 0.0001 |
| spikes placed **on** the player | 0.003 | 0.0002 |
| distinct move responses over 32 enemy placements | **1** | **1** |

The last row is the one that matters: after 665k steps the policy had the same
distinct-move-response count as a randomly initialised network. It is 3.5x more
responsive to which door exists than to which side the enemy is on.

### Why the reward is the leading suspect

Measured on v18 at 500-665k, per episode:

```
door_shaping  +2.47      dense, per-step, attributable to the action just taken
combat terms  12.93 total magnitude
              (dealt +2.99, kill +3.45, clear +1.68, taken -3.83, death -0.98)
                                                          ratio  1 : 5.2
```

For a policy that cannot yet aim or dodge, damage taken and dealt arrive very
nearly at random with respect to the action chosen on any given step, while
`door_shaping` is dense and attributable. The v16 rebalance quadrupled
`room_clear`, tripled `kill` and *cut* `door_shaping` — it buried the one signal
the agent could learn from under five times its magnitude in noise. v12 had the
opposite balance and is the only run whose curriculum ever moved.

This is a hypothesis with a mechanism and a measurement behind it, not a proof.
What is proven is that the policy is near-constant and that the heads, not the
encoder, are where the information dies.

### The reconstruction, and what is deliberately not v12

Built, tested (125 pass, 3 correctly self-skipped on `GRID_CLASSES == 3`) and
**deployed**. Shapes verified live: grid 405, ego 243, entity 22, scalar 21,
door 13.

| | v18 | reconstruction | v12 |
| --- | ---: | ---: | ---: |
| `door_shaping` | 3.00 | **4.00** | 4.00 |
| `room_clear` | 12.00 | **3.00** | 3.00 |
| `kill` | 1.50 | **0.50** | 0.50 |
| `damage_dealt` | 0.10 | 0.10 | 0.10 |
| `blocked_move` | -0.01 | **0.0** | 0.0 |
| `GRID_CLASSES` | 7 | **3** | 3 |
| `ENTITY_FEATURES` | 23 | **22** | 22 |
| `SCALAR_FEATURES` | 26 | **21** | 21 |
| ego encoder | conv | **flat** | flat |
| type embedding | yes | **no** | no |

Source came from `95fee25` (the v14-era tree, which is v12's observation plus the
property planes) with `GRID_CLASSES` set back to 3 — bits 0-2 reproduce v12's
encoding bit for bit, so that constant is the entire switch.

**Three deviations from v12 are deliberate**, and are the first suspects if the
reconstruction does not reproduce:

- door tiles read free when open (v13 fix; v12 read every doorway as wall)
- the player's tile comes from `GetGridIndex` (v14 fix; v12 derived it and was
  wrong in 24% of half-shaped rooms)
- the bridge survives a game crash (v9b fix; harness only)

Reverting known bug fixes to chase a number is the wrong trade, but naming them
now is cheaper than discovering them later.

**The mod had to be redeployed.** The deployed copy was still v17's, and a v12
`env.py` reading it would have had `ENTITY_KINDS.get(kind, 0)` file every
unmapped entity as **enemy**. Caught before the run, not after. Any revert that
touches `env.py` touches `main.lua` too — they are a matched pair.

### How to read the run

**0.61 is not the target at 1M.** v12 sat at the end of a resumed lineage with
~4M cumulative steps. The honest bar for a fresh 1M is v7+v7b's **0.34 cleared**
at matched budget, which every fresh run since has come in under.

**Gate at ~200k on the probes, not at 1M on `rooms_cleared`.** v15 planned
exactly this check and never ran it, and four runs went by. The primary gate is
`diagnose_shoot_axis.py`'s **distinct move responses**, because it has a known
good value — v8 scored 4 distinct and 27/32 horizontal alignment. 1 distinct is
untrained-equivalent and means stop.

`diagnose_grid_use.py`'s `blocking -> irrelevant` is worth logging but **cannot
be a pass/fail gate: no policy in this project has ever moved it.** It has read
0.010-0.028 in every run measured, against an untrained floor of 0.0001. There is
no historical example of success to threshold against.

```bash
.venv/Scripts/python.exe -m isaac_ai train-floor \
  --instances 20 --steps 1000000 --run-name floor-v19 \
  --entropy-target 0.5
```

No `--min-action-prob`: v12 predates it, and it is confirmed inactive at these
entropies anyway. No `--resume`: v12's checkpoint cannot be loaded and this
observation is new.

**The most important fact in this file: floor-v12 is the high-water mark and
every change since has made things worse, monotonically.**

| run | cleared | seen | success | still | peak succ | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| v7 | 0.26 | 1.27 | 0.208 | - | 0.325 | fresh |
| v7b | 0.34 | 1.53 | 0.305 | - | 0.450 | resumed |
| v8 | 0.57 | 1.64 | 0.504 | - | 0.600 | entropy ceiling |
| v9b | **0.68** | 1.82 | 0.467 | - | 0.600 | semantic flags |
| v11 | 0.57 | 1.66 | 0.477 | - | 0.725 | egocentric grid |
| **v12** | 0.61 | 1.80 | 0.498 | - | **0.775** | **best overall; difficulty hit 0.090** |
| v13 | 0.49 | 1.67 | 0.429 | 0.050 | 0.675 | blocked-move penalty |
| v14b | 0.26 | 1.36 | 0.248 | 0.032 | 0.350 | **fresh restart** + grid planes |
| v15 | 0.19 | 1.34 | 0.177 | 0.042 | 0.350 | conv encoder, embeddings |
| v16 | 0.12 | 1.18 | 0.151 | 0.125 | 0.300 | reward rebalance |
| v17 | 0.08 | 1.00 | 0.090 | 0.223 | 0.250 | identity expansion, creep |
| v18 | 0.14 | 1.10 | 0.139 | 0.035 | 0.275 | `damage_dealt` 0.30 -> 0.10; **stopped at 801k** |
| **v19** | **0.28** | **1.44** | **0.242** | 0.044 | **0.450** | **v12 reconstruction; decline reversed** |
| v19b | 0.29 | 1.59 | 0.253 | 0.080 | 0.450 | resumed +1M, `--min-action-prob`; aiming fixed, clearing flat |
| v20 | 0.23 | 1.57 | 0.220 | 0.056 | 0.475 | `kill` 0.50 -> 1.50; opened 0.43, decayed below baseline, **stopped at 635k** |
| **v21** | **0.47** | 1.99 | 0.380 | 0.049 | 0.600 | **+4.5M, no change. Budget was the constraint; climbed 0.29 -> 0.47, still rising** |
| v22 | 0.53 | 1.94 | 0.404 | 0.069 | 0.700 | door shaping gated off in combat; slope unchanged, missed the +0.0091/100k bar |
| v23 | 0.48 | 2.02 | 0.386 | 0.071 | 0.625 | eight obstacle rays; slope fell to +0.0026, `blocked_rate` and deaths lowest ever |
| combat-v13 | - | - | - | - | - | **first policy here that aims: 88% fires-ok, difficulty 0.42 with obstacles, no dead axes** |
| **v24** | 0.34 | 1.73 | 0.266 | 0.035 | 0.525 | **warm start from combat-v13. Aim survived intact; `blocking -> irrelevant` 0.0357, best ever. Beats a fresh 1M floor run on every column** |
| **v25** | **0.59** | **2.15** | **0.444** | 0.054 | **0.750** | **+5M. Best run in the project; `difficulty` reached 0.090, equalling v12. Aim decayed 88% -> 59% while doing it** |
| **v26** | **0.84** | **2.41** | **0.534** | 0.042 | **0.775** | **+3M, no change. Both all-time records broken: cleared 0.836 and difficulty 0.120, first run ever past v12** |
| v27 | 0.86 | 2.38 | 0.549 | 0.044 | 0.775 | +0.76M, no change; stopped early |
| v27b | 0.87 | 2.56 | 0.550 | 0.061 | 0.775 | +5M, no change. Three runs at ~0.85: budget stopped paying |
| **v28** | **1.06** | **2.91** | **0.609** | 0.041 | **0.825** | **combat gate back on. Cleared past 1.0, peak success beats v12 at last, `target_rooms` 2 held for 14% of updates** |

### The correction

The v14 "criterion" was set as *does a fresh run with the fixes beat floor-v7*,
because v7 was the last fresh run. That was a fair control for one question and
a badly misleading frame for the real one: **the lineage v7 -> v12 reached
cleared 0.61 and peak success 0.775**, and nothing since the fresh restart has
come close. The comparison that was reported at the time (v14 0.26 vs v7 0.26)
made a 3x regression look like parity.

Some of that is budget — v12 had ~4M cumulative steps against v14-v17's 1M each.
But at *matched* ~1M cumulative, v7+v7b reached **0.34** and every fresh run with
the accumulated fixes came in below it, declining run over run. So the changes
themselves are net-negative, not merely under-trained.

### What that implies for the next session

**Stop adding. Start subtracting.** The observation has grown at every step —
grid 405 -> 810 -> 945, ego 243 -> 567, entity 17 -> 23, scalars 21 -> 26 — and
each expansion slowed learning without producing a measurable gain. The reward
was rebalanced once and it created a camping strategy that had to be partly
reverted.

The recommended next move is **not another feature**. It is to reconstruct the
v12 configuration — its rewards (`damage_dealt` 0.10, `kill` 0.50, `room_clear`
3.00, `door_shaping` 4.00, no `blocked_move`) and its observation
(`GRID_CLASSES` 3, `ENTITY_FEATURES` 22, `SCALAR_FEATURES` 21, flat ego encoder,
no type embedding) — confirm it reproduces ~0.61 cleared, and only then change
one thing at a time against a baseline that is known to work.

**v12's checkpoint cannot be loaded.** The observation has changed four times
since. Reproducing it means retraining, and there is no way back to the policy
itself. That is the cost of moving fast on the observation, and it is the single
biggest thing to avoid repeating.

### Reading the rest of this file

Sections below are **not in chronological order** — the file grew by prepending,
so newer material sits in the middle. Each section is self-contained and states
which run it refers to. Trust the table above for the arc.

## floor-v8: the entropy ceiling worked, and no axis is dead

1M steps resumed from v7b with `--entropy-target 0.5`, one variable.

| band | return | success | rooms_cleared | shoot_x | shoot_y |
| --- | --- | --- | --- | --- | --- |
| 0–200k | +6.07 | 0.291 | 0.34 | 0.938 | 0.837 |
| 400–600k | +7.13 | 0.325 | 0.35 | 0.817 | 0.755 |
| 600–800k | +8.46 | 0.373 | 0.46 | 0.802 | 0.642 |
| 800k–1M | **+10.04** | **0.463** | **0.52** | **0.727** | 0.595 |

`shoot_x` came off the ceiling — 1.084 (abandoned) to 0.712 — and
`probe_heads.py` reports **no dead axes** for the first time in this project's
history. `rooms_cleared` rose 0.34 → 0.52 and success 0.308 → 0.475, so this is
not an entropy-metric artefact.

**It learned an actual maneuver.** `diagnose_shoot_axis.py`, v8 against v7b:

| | v7b | v8 |
| --- | --- | --- |
| distinct move responses | 2 | 4 |
| closes horizontal gap | 16/32 (chance) | **27/32** |
| closes vertical gap | 16/32 | 16/32 (chance) |
| fires ok (single frame) | 22/32 | 16/32 |

Horizontal alignment far above chance while vertical stays at chance is the
*decomposition* signature the diagnostic was written to detect: the agent closes
the horizontal gap to put the enemy on its column, then fires vertically, where
`shoot_y` (0.595) is its most committed axis. It holds with the doors marked
visited (23/32) and removed entirely (26/32), so it is enemy-driven, not
door-driven. Two distinct outputs are enough to produce it — "go left or right
depending on which side the enemy is" — which is why the count stays low.

**Both probes score a two-phase strategy badly.** `probe_heads.py` fell to 6/16
and `fires ok` to 16/32 *because* of this: they read a single frame, and in that
frame the agent is repositioning rather than firing. Neither number is a
regression. Judge maneuvering policies on `rooms_cleared` and the alignment
rates.

**`difficulty` is still 0.000** after ~2M cumulative steps — success 0.475
against the 0.72 threshold. Closer, not there.

## The dead-axis question, as it stood after v7b

**On v7b it was abandonment, not decomposition** — v8 is what changed it.
`diagnose_shoot_axis.py` places an enemy at 16 angles × 2
ranges and finds **2 distinct movement responses** — the policy answers
`down-left` to 27 of 32 placements regardless of where the enemy is, and fires a
constant `right`. The 50%/50% alignment rates are what a fixed direction scores
against a circle of placements, i.e. chance. Controlled for the exploration pull
by re-running with doors marked visited and with no doors at all: still 2
responses, so it is not door-seeking outranking the enemy. There is no
maneuvering-to-align, so the README's "aligns on one axis by moving" story does
not hold here.

`damage_dealt` is *not* another reward that never fires — the mod counts it, the
observation carries it and `compute_reward` pays 0.10 a point. The problem is
proportion and density: navigation earns roughly +7.5 an episode (door shaping
telescoping ~+4 a room plus `new_room` +1.00, at 1.5 rooms) against combat's
~+2.5, and navigation's arrives *every step* while combat's is sparse. `shoot_x`
earns anything only when an enemy is horizontally aligned — a fraction of an
already-small share — while the entropy term acts on all 16 gradient steps of
every update.

`PPOConfig.entropy_target` clamps each head's entropy bonus per sample, so a
head at or above the ceiling contributes a constant and gets no gradient, while
one that collapses below it is still pushed back up. Default 0.0 keeps the old
behaviour; the logged `entropy` metric is deliberately the true summed entropy,
never the clamped objective, so runs stay comparable.

**Judge this on `rooms_cleared`, not on the entropy numbers.** The failure mode
to watch for is `shoot_x` collapsing to a *constant* "right" — entropy near 0,
which would read as a fix on the metric while the agent still cannot aim. The
honest checks are `rooms_cleared`, and `diagnose_shoot_axis.py`'s distinct-move
count and fires-ok rate, which a constant cannot fake.

## floor-v17: stop hand-classifying the game

```bash
.venv/Scripts/python.exe -m isaac_ai train-floor   --instances 20 --steps 1000000 --run-name floor-v17   --entropy-target 0.5 --min-action-prob 0.05
```

**Fresh** — the observation changed shape again.

### The principle

Every hand-written mapping in this project has been wrong at least once, silently:
the chest list missed 2 of 13 variants that were **41% of all pickups**;
`GRID_DOOR` was filed as solid so every exit read as wall; and
`collectEntities` was a **whitelist of four types**, which silently dropped
creep, fires, laser beams and every attack telegraph.

A whitelist fails for every category nobody thought of, not just the entries you
fumbled. So it is now an **exclusion list**: everything the game reports reaches
the observation with its raw Type, and the learned embedding covers whatever the
semantic flags do not name. Only things that are *ours* are dropped — the player,
its own tears, its familiars.

Hand-maps remain only where the classification is genuinely ours rather than the
game's: `needs_bomb` encodes a fact about *this agent* having no bomb action, and
the shaping needs that decision. Reward logic is our design; the observation is
not.

### What changed

- **`other` entity kind.** 1,004 entity observations per 24,000 now arrive that
  were previously invisible. `ENTITY_KINDS` gained it *last*, because
  `.get(kind, 0)` used to default an unknown entity to **0 = enemy**.
- **A bomb flag, appended** (`ENTITY_FEATURES` 22 -> 23). With three kind flags a
  bomb and an "other" were both "none of the first three" and indistinguishable.
- **Truncation priority.** `MAX_ENTITIES` is 32 and the list was sorted by
  distance alone — safe only while the whitelist kept it short. Now that
  everything arrives, a crowd of pickups could evict the enemies and nothing
  logged would show it. Threats now outrank scenery, distance breaks ties.
  Measured: the list never filled in 24,000 observations, so this is a guard
  rather than a fix.
- **Creep as grid plane 6** (`GRID_CLASSES` 6 -> 7). Creep is `ENTITY_EFFECT`,
  so no grid entity saw it and the whitelist dropped it too. Rasterised by
  `GetGridIndex` rather than sent as entities: it is a spatially distributed
  floor hazard exactly like spikes, and one pool spawns enough effect entities
  to fill all 32 slots on its own.

### The derivation caught itself being too greedy

Deriving creep variants from `EffectVariant` by name found **21** — and 11 were
`PLAYER_CREEP_*`: the player's own holy water, black powder, lemon party, which
damage *enemies*. Marking those as floor hazard would have taught the agent to
flee its own beneficial puddles. Excluded by prefix; the map now resolves 10.

That is the chest lesson with the sign flipped: hand-listing missed entries,
deriving caught too many. Deriving is still right — it just needs the exclusion
the naming already makes explicit.

Verified live: enemy creep is **1,564 tiles across 32,000 observations**, present
in 0.32% of them. Rare, so it will not move the headline metrics — but it is
damage the agent previously had no way to see coming.

### Note on the first measurement

A 5,600-observation sample returned **zero** creep and looked like a pass. It was
not; the rate is 0.32% and the sample was too small. Re-run at full fleet.
A probe that returns nothing is not a pass.

## floor-v16: the reward rebalance

```bash
.venv/Scripts/python.exe -m isaac_ai train-floor   --instances 20 --steps 1000000 --run-name floor-v16   --entropy-target 0.5 --min-action-prob 0.05
```

**Fresh.** The reward changed substantially, so a resumed critic would be
badly miscalibrated and a resumed policy would be one trained to avoid exactly
the behaviour we are now paying for.

### The finding this rests on

Killing a 10 hp enemy paid `damage_dealt` 1.00 + `kill` 0.50 = **+1.50**, while
one contact hit costs **-1.00**. A kill that cost two hits was net **-0.50**.
The agent was not failing to fight — it was correctly declining a losing trade,
and everything followed from that: walk to doors, spray in a fixed direction,
leave. Measured on v15, an episode earned ~6.7 from navigating against ~1.4 from
combat: **17% of the reward for the thing the task is named after**.

| term | was | now |
| --- | ---: | ---: |
| `damage_dealt` | 0.10 | **0.30** |
| `kill` | 0.50 | **1.50** |
| `room_clear` | 3.00 | **12.00** |
| `door_shaping` | 4.00 | **3.00** |
| `damage_taken` | -1.00 | -1.00 (untouched) |

A kill now pays +4.50 and stays +2.50 after two hits.

**`door_shaping` is cut only 4.00 -> 3.00 on purpose.** At 0.50 one step of
approach was worth ~0.004, the policy gradient came out ten times smaller than
the entropy bonus, and nav-v1 spent the whole run maximising randomness. The
rebalance is achieved by *raising combat*, not by gutting navigation, because
gutting navigation has a documented failure mode and raising combat does not.
`damage_taken` stays at -1.00 for the same reason in reverse: -2.50 was tried,
made encounters net-negative, and produced "do not engage".

### Reward decomposition is now logged

`compute_reward` builds a per-term breakdown and returns the total **summed from
it**, so the log cannot disagree with the reward actually paid. `floors.py`
accumulates it per episode including the terms it adds itself, and `r_*` fields
land in `metrics.jsonl`.

Every rebalance before this one was argued from a split *inferred* from
aggregates — rooms_seen times a coefficient, enemies assumed at three a room.
Verified live: **27 episodes, breakdown != episode return on 0 of them.**

Random-policy baseline under the new numbers, which is worth knowing before
reading the trained ones:

```
damage_taken -2.593   damage_dealt +2.411   step -1.591   door_shaping +0.946
floor_death  -0.815   kill         +0.722   new_room +0.444   blocked_move -0.093
-> combat 69% of positive reward, navigation 31%
```

Note `damage_taken` is the single largest term for a random agent. If the
trained policy still ends up avoiding fights, that is where to look first.

### What this does and does not do

It makes fighting **worth doing**. It does not make the agent **able** to fight:
`shoot_x` was still near-uniform after 1M steps in v15. So watch
`rooms_cleared` against `ended_died`. If cleared rises, the trade was the
blocker. If only deaths rise, the agent wants to fight and cannot, and the next
problem is aiming itself rather than the reward.

`mean_return` will rise mechanically because the numbers are bigger. It is not
evidence of anything.

## floor-v15: make the information usable, not more plentiful

```bash
.venv/Scripts/python.exe -m isaac_ai train-floor   --instances 20 --steps 1000000 --run-name floor-v15   --entropy-target 0.5 --min-action-prob 0.05
```

**Fresh — the architecture changed, so there is nothing to resume from.**

The premise: across v10, v12 and v14 the policy's response to moving a rock onto
its own line of fire was **1-3% total variation**, including on v14 — a clean 1M
steps with the window already egocentric and player-centred. Three separate
additions of obstacle information were all ignored. The information was present
and unusable, which points at the encoder rather than the observation.

**1. A conv over the egocentric window**, replacing `Linear(486, hidden)`. To
answer "is there a rock one tile to my right" an MLP must learn that
relationship independently for all 486 inputs, with no notion that neighbouring
cells are neighbours. Two conv layers supply the spatial prior structurally.
Deliberately **not** globally pooled — the player is always at the centre, so
where a thing sits relative to that centre is the entire content.

**2. Enemy type as a learned embedding.** `nn.Embedding(1024, 8)` indexed by the
game's `EntityType`, concatenated per entity before the pooling. An embedding
has no ordinal structure, which is the objection that ruled out a raw type
number everywhere else. Zero-initialised, so every type starts identical and
differences are earned rather than asserted. Type only, not Variant — a Gaper
and a Frowning Gaper share a row for now.

**3. Room type as flags.** It was `room["type"] / 30.0`, a single ordered scalar
over ~30 types — exactly the "curse is twice boss" problem that doors were given
flags to avoid. The mod has been sending a resolved `room.category` all along
and nothing read it. `SCALAR_FEATURES` 21 -> 26.

### Why not combat pretraining

Considered and rejected on this project's own numbers. `combat-v6`'s difficulty
0.73 was **largely the wall-hugging spawn exploit**: v5-v7 sat at 0.74-0.77 with
it, and the moment the player was repositioned before spawning, v8-v12 collapsed
to 0.33-0.45. The README already priced that exploit at "0.48 of win rate at
difficulty 0.75". Honest post-fix combat difficulty is ~0.40, in an empty room —
and a tactic learned in an empty room is exactly the thing that does not
transfer to a floor with rocks and pits in it.

### Verified before running

```
scalar length         26 produced = 26 declared
room categories       normal / curse / other, zero unmapped
entity types          18,610 observations, 15 distinct, max id 252 (table 1024)
encoded == raw        yes
```

The first attempt at that check sampled **zero entities** — 400 random steps at
8 instances never leave the start room — so it was rerun at full size. A probe
that returns nothing is not a pass.

### The early read, at ~200k rather than at the end

`diagnose_grid_use.py`'s **blocking -> irrelevant** number. It has been 0.010 to
0.028 in every run ever measured. If the conv encoder is doing its job that
should move materially, and it is checkable long before the run finishes. If it
does not move, obstacles are not ignored because of the encoder, and the reward
genuinely does not depend on them enough — which is the next thing to change.

## floor-v14: the criterion fired. The bugs were real; they were not the bottleneck.

Fresh, no resume, all four fixes, 1,000,960 steps. Against the agreed benchmark:

| at ~1M steps | success | rooms_cleared | rooms_seen |
| --- | ---: | ---: | ---: |
| v7 + v7b, **no fixes** | 0.305 | 0.34 | 1.53 |
| v14, **all fixes** | 0.248 | 0.26 | 1.36 |

**It does not beat v7. It is slightly behind.** We agreed in advance that this
means the problem is the reward or task design, not bugs, so that is the finding.

Confounds, stated so they are not used as an excuse: v14 was interrupted at 307k
(Adam reset), it pays a `blocked_move` penalty v7 never did, and its observation
is far larger (grid 810 vs 405, plus the ego window and semantic flags), so it
has more to learn per step. None of that turns level into a win.

### The fixes did work — that is what makes this conclusive

```
up-door pathology   v13: P(up) 0.000-0.157, picks still/down from every position
                    v14: P(up) 0.772-0.839, picks UP from every position
blocked_rate        v13 15-17%  ->  v14 7.4%   (random-walk floor 2.5%)
dead actions        none; min_prob 0.15-0.24 across all four heads
```

So this is not "the fixes failed". Four genuine bugs are gone, the harness is
more robust, and the agent navigates properly for the first time. Task
performance did not move.

### What has actually been constant since v5

```
run     enters rooms/ep   clears   share of entered rooms cleared
v7b          1.53          0.34              22%
v9b          1.82          0.68              37%
v12          1.80          0.61              34%
v14b         1.36          0.26              19%
```

**It enters rooms and does not clear them.** STATUS said exactly this at v5 —
"it explores fine and cannot fight" — and every one of the eight runs since has
been about navigation or perception. Combat on floors has never been addressed.

### The project already knows combat is learnable in isolation

`combat-v6` reached curriculum difficulty **0.73** on the isolated encounter
task. Embedded in floors it has never worked at all. The pivot away from that
task was justified by bugs in the synthetic scaffolding — but the scaffolding is
also what made combat learnable, and dropping it dropped the only setting where
the agent ever learned to fight.

`ActorCritic` is shared between the two tasks, so a combat policy warm-starts a
floor policy directly. That is what `--resume runs/combat-*/policy.pt` was for.

**Next: retrain combat from scratch on the current observation, then warm-start
floors from it.** Not another floor variant.

## floor-v14: FRESH. Break the lineage.

```bash
.venv/Scripts/python.exe -m isaac_ai train-floor \
  --instances 20 --steps 1000000 --run-name floor-v14 \
  --entropy-target 0.5 --min-action-prob 0.05
```

**No `--resume`.** Every floor run from v7 to v13 was warm-started from the one
before it — v7 -> v7b -> v8 -> v9 -> v9b -> v10 -> v11 -> v12 -> v13, eight
consecutive resumes on one policy lineage. A warm start deliberately preserves
behaviour, so the down-bias was the thing most protected from every fix aimed at
it. The README already documents this exact trap from combat-v5..v7 and it was
repeated anyway.

**The pathology being fixed.** v13, asked to leave a room whose only exit is the
up door, pushes *down* harder the closer it gets to that door:

```
player y   move_y [up    none   down]
400        [0.157  0.717  0.126]
280        [0.011  0.508  0.482]
170        [0.000  0.200  0.800]   <- standing on the door
```

Control, only a down door: picks down at 0.94-0.97 from every position. So it is
an active downward drive, not indifference — and it accounts for the bottom-wall
hugging, the stuck rooms, and a large share of the 15% blocked rate.

### The three fixes

**1. `GRID_DOOR` was encoded as solid.** Every doorway read as a wall in the
obstacle grid, including the one the shaping was steering towards. A door tile
is now `GRID_FREE` while the door is open and solid only when shut. This is
worse in one direction by construction: a 1x1 room is 9 tiles tall and the ego
window is 9 tall, so the top and bottom walls are *always* in view while the
side walls usually are not — "solid straight ahead" was a far more reliable
signal vertically, which is the direction the policy learned to avoid.

**2. `min_action_prob`, a floor under each individual action.** `entropy_target`
stops a whole head being dragged to uniform and does nothing else. It cannot see
one action inside a head die, because the other two carry the entropy: at
`[0.01, 0.61, 0.37]` the entropy is 0.715, above the 0.5 target, so the clamped
bonus contributed **zero** gradient and nothing pushed P(up) back up. A floor of
0.05 also bounds head entropy at ~0.39, so it subsumes most of what the target
did while actually protecting the action.

**3. `min_prob_*` logging.** The instrument gap. `entropy_move_y` read a healthy
**0.694** for the whole of v13 while P(up) was **0.000**. That is the
summed-entropy failure one level down — the instrument built to catch a dead
axis is blind to a dead action within an axis. Watch `min_prob_move_y`, not the
entropies.

**Grid property planes are now ON** (`GRID_CLASSES` 3 -> 6). They were held dark
only to keep the warm start exact; v14 has nothing to preserve.

### Probed, and a fourth bug found on the way

```
probe_ego_grid      centre 14000/14000   standable 14000/14000   edge 4039/4039
                    2x2 room (16x28) covered on live data
probe_door_targets  secret 1489 seen, all flagged needs_bomb, 0 shaping targets
door tiles freed    41.7 solid over 2.3 open doors — and 44 - 2.3 = 41.7 exactly
```

That last line matters: `grid:ToDoor()` sits inside a `pcall`, so a failure
would have fallen through to "solid" and left the fix silently inert while every
other check still passed. It is live.

**The fourth bug: the player's tile was derived, not asked for.** The first probe
run put the player inside a solid tile on 368/14000 observations — all of them in
"half" room shapes (IH, IV), where the playable area is a *sub-rectangle* of a
15x9 grid whose remainder is wall, so the extents-based derivation misplaced it
by a tile in 24% of those observations. The mod now reports
`room:GetGridIndex(pos)` and Python uses it; the derivation survives only as a
fallback for arbitrary points. Standable went 13632 -> **14000/14000**.

Three of the four bugs found this session were the same shape: **code deriving
something the game can be asked for directly.**

**The stopping criterion.** If a fresh 1M with these fixes does not clearly beat
floor-v7 (the last fresh run) at matched steps, the problem is the reward or
task design, not bugs — and the answer then is to rethink the reward structure,
not to ship a fifteenth variant.

## The real blocker: the agent walks into walls, and it is nearly free to

`diagnose_camping.py`, v12's policy against a random-action control on the same
fleet:

| | trained v12 | random |
| --- | ---: | ---: |
| asked to move but did not | **18.6%** | **2.5%** |
| mean y (0.5 = centre) | 0.71 | 0.63 |

**Pinned against geometry roughly 7x more often than a random walk.** Position
is only mildly abnormal — random already spends 70% of its time in the bottom
half against the policy's 77% — so the striking finding is the *blocking*, not
the corner. The "camping" is a policy repeatedly asking for directions that are
into a wall.

Two candidate causes were tested on the same run:

- **Straight-line shaping holding it against obstacles: partly true.** Something
  solid sits on the line to the shaping target on **38.4%** of blocked steps
  against a **21.5%** baseline while moving — a 1.79x enrichment, reproduced at
  1.91x on a second run. `door_potential` measures closeness with `math.dist`,
  so walking *around* an obstacle increases the distance and the shaping pays a
  penalty for it. Real, but it leaves ~60% of blocked steps unexplained.
- **Pressed against a shut door: refuted.** It is at a door on 9.9% of blocked
  steps against 13.8% while moving — *less* likely, not more, and mean closeness
  to the target is identical (0.67 vs 0.66).

**What is left is that a blocked step is almost free.** It costs the -0.002 step
penalty and nothing else: no position change means no potential change, so
walking into a wall and walking uselessly in the open pay exactly the same.
Meanwhile a step of real progress earns roughly +0.03 of door shaping. There is
an opportunity cost and no actual penalty, which is a very weak gradient for
learning that walls stop you — and explains why the obstacle grid, and then the
egocentric version of it, both went unused. **Adding perception cannot beat an
absent gradient.**

## Next: floor-v13, the blocked-move penalty

```bash
.venv/Scripts/python.exe -m isaac_ai train-floor \
  --instances 20 --steps 1000000 --run-name floor-v13 \
  --resume runs/floor-v12/policy.pt --entropy-target 0.5
```

`blocked_move = -0.01`, charged when the agent asks to move and does not travel.
It is the **only** change: `GRID_CLASSES` is still 3, `ENTITY_FEATURES` still 22,
`DOOR_FEATURES` still 13. No identity work is enabled. Resume from v12 is *26
copied, 0 widened, 0 fresh*.

The ordering it has to preserve, and does:

```
blocked step        -0.0120
standing still      -0.0020
step toward a door  +0.0749
room_clear          +3.00
```

Standing still is deliberately uncharged — strictly better than grinding a wall
and earning nothing either way, so it is a refuge rather than a strategy.
**`still_rate` is logged precisely so that if it becomes one, it shows.**

`blocked_rate` and `still_rate` now stream to `metrics.jsonl`, so this is
measurable without a fleet probe. Success is `blocked_rate` falling from ~15%
towards the **2.5%** a random walk registers, without `still_rate` climbing off
its ~5% baseline.

**Charged per agent step, not per tick.** The first version counted each
`action_repeat` exchange, which doubled the sample and inflated the rate from
14.2% to 22.7% — the gap being Isaac's *acceleration* on the first tick after a
direction change, not obstruction. It would have taxed ordinary turning. After
the fix the env's counter and an independent recount agree exactly:
**1224/9425 = 13.0%** on both.

## Superseded: v13 was going to be the grid property planes

```bash
.venv/Scripts/python.exe -m isaac_ai train-floor \
  --instances 20 --steps 1000000 --run-name floor-v12 \
  --resume runs/floor-v11/policy.pt --entropy-target 0.5
```

**v12's single variable is the secret-room fix, and it needed two halves.**
Removing it from the shaping was not enough: the first run stopped steering at
secret doors and the agent went on walking into them anyway. The encoded door
still read `present=1, unvisited=1`, byte-identical to a sacrifice-room door
that *is* passable, and the policy carried a million steps of "unvisited doors
are worth walking to". **Killing a reward removes the reinforcement but leaves
the habit and the perception that justifies it** — worth remembering generally.

A door needing an explosive is now left out of the observation entirely, so the
slot reads as plain wall. No change of shape, so the resume stays exact.

Watch `ended_idle` fall and `rooms_seen` rise; `exhausted` will *rise* and that
is the metric becoming honest, not a regression. Note that in the first attempt
`exhausted` rose as predicted (0.234 -> 0.356) while `ended_idle` did **not**
move — because removing a false target does not create a true one. If that holds
after this fix, the remedy for those episodes is the non-local potential, not
more door filtering.

**v13's change is already written, tested and probed** — grid tile *properties*
instead of one class each. `GRID_CLASSES = 3 -> 6` in `env.py` is the entire
switch, and nothing else needs editing.

**Why it is safe to have both in the tree at once.** The mod now reports a
**bitmask** per tile rather than a class id, and bits 0-2 carry exactly the
membership the old three classes had — so decoding only those reproduces the
previous encoding bit for bit. Verified: warm start from v11 is *26 copied, 0
widened, 0 fresh*. At 6 it becomes *24 copied, 2 widened (grid 405->810, ego
243->486), 0 fresh*, still exact, because planes are the leading dimension of
the flattened grid and new ones append at the end.

**What the new planes carry.** 3 damaging, 4 destructible, 5 retractable, all
additive:

- `GRID_ROCK_SPIKED` was **solid only**, so a rock that damages on contact was
  reported as an ordinary rock. Plane 3 carries it without touching plane 0.
- `GRID_SPIKES_ONOFF` was merged with permanent spikes; plane 5 separates
  "sometimes safe to cross".
- `GRID_POOP` was merged with `GRID_PILLAR`/`GRID_WALL`; plane 4 marks what can
  be shot away. Rocks are *not* destructible to this agent, which has no bomb
  action.

Membership is resolved from the game's own `GridEntityType` names with the same
missing-name warning as pickups and rooms — an unresolved grid name is the worst
of the three, since it would default a tile to free floor and tell the agent a
rock is walkable.

### Probes, both green after the mod change

`probe_ego_grid.py`, 14,000 observations at 20 instances: centre matches raw
**14000/14000**, player tile standable **13994/14000**, edge ring walled
**2527/2527**, and two large rooms (16x28 raw grids, shapes 8 and 11) covered on
live data rather than only synthetically.

`probe_door_targets.py`: secret doors seen 1510, all 1510 flagged `needs_bomb`,
**shaping target 0**; every other room type unaffected.

**Both probes reported false failures first, for the same reason each time: they
restated logic instead of importing it.** The door probe kept its own copy of
the filter and went on reporting the fixed bug as present. The ego probe decoded
the tile payload as a class id and flagged 22 pits as mismatches, because a pit
moved from value 3 to bit 2. If a probe answers "what would the code do", it has
to call the code.

## floor-v11: the curriculum moved for the first time

1M steps with the egocentric grid. `difficulty` reached **0.030** and
`success_rate` peaked at **0.725** — the first time in this project that the
floor dial has left 0.000.

| band | return | success | rooms_cleared | exhausted |
| --- | ---: | ---: | ---: | ---: |
| 2004k | +11.27 | 0.482 | 0.60 | 0.283 |
| 2404k | +11.33 | 0.479 | 0.61 | 0.249 |
| 2804k | **+11.86** | 0.512 | **0.65** | 0.230 |

Modest against v10 (+10.4 / 0.44 / 0.55) but consistently better, and it broke
the plateau that a plain extra 1M could not.

## The shaping was steering into walls — found by watching, confirmed by probe

**A secret room's door is closed and *unlocked*.** `door_potential` skipped a
door only when `locked` was set, so it targeted secret doors — which need a bomb
the agent has no action for, since `floors.py` hardcodes `bomb: False`. The
agent walked confidently into the wall until the idle limit ended the attempt.

Measured by `probe_door_targets.py` over 900 fleet steps, before and after:

```
target room     seen  ever open  ever locked  needs bomb  shaping target
secret          2450      False        False           -            1068   before
secret          3316      False        False        3316               0   after
library          533      False         True           -               0   control
```

1068 targets was **6% of every step where the potential had a target**. The
locked library door is the control: never open either, and correctly skipped
533/533 by the existing filter — so this was a gap in the filter, not in the
idea.

Deliberately **not** a check on `open`: doors shut for a fight reopen on the
clear, and filtering those out would collapse the potential the moment a fight
starts — a discontinuity paid on *entering a room*, which is the exact shape
that plateaued floor-v1 through v3 at −2.885 a transition. The mod now reports
`needs_bomb` from the game's own `ROOM_SECRET` / `ROOM_SUPERSECRET` /
`ROOM_ULTRASECRET` constants, and `target_type` alongside the category so this
stays diagnosable.

**The probe reported the bug as still present after the fix**, because it kept
its own copy of the filter. The filter now lives in `door_is_targetable()` and
the probe imports it. Any diagnostic asking "what would the shaping do" must
call that rather than restate it.

**Expect `exhausted` to rise in v12 and do not read it as a regression.** A room
whose only unvisited door is secret used to report a positive potential — the
agent had a target, just an impossible one — so it did not count as stranded.
Now it correctly reports nothing to walk towards.

## floor-v10: flat. More steps alone are spent.

1M steps, no change but the step budget. It bought nothing:

| band | return | success | rooms_cleared | exhausted |
| --- | ---: | ---: | ---: | ---: |
| 1003k | +10.59 | 0.440 | 0.56 | 0.242 |
| 1403k | +10.43 | 0.446 | 0.55 | 0.198 |
| 1803k | +10.69 | 0.471 | 0.64 | 0.237 |

Against v9b's final band of +11.46 / 0.518 / 0.81, v10 is flat to slightly
worse. `difficulty` never moved; peak `success_rate` touched **0.700** against
the 0.72 threshold and fell back. So the plateau is structural, not a matter of
budget — which is the result that justifies spending a change on v11.

`exhausted` held at **0.226** across the run, so the stranding is real and
steady, and the non-local potential stays on the list after the grid.

## Built, probed, and ready: the egocentric grid

```bash
.venv/Scripts/python.exe -m isaac_ai train-floor \
  --instances 20 --steps 1000000 --run-name floor-v11 \
  --resume runs/floor-v10/policy.pt --entropy-target 0.5
```

**Probed at 20 instances and passing** — 18,000 observations, and it caught a
real bug first.

```
                        before fix     after fix
player tile standable   13152/18000    17907/18000   (73% -> 99.5%)
centre matches raw      18000/18000    18000/18000
edge ring walled          5658/5658      5399/5399
room shapes                    1, 3       1, 3, 6   incl. a real 2x1 (9x28 grid)
```

**The bug: two coordinate systems that describe different rectangles.**
`GetTopLeftPos`/`GetBottomRightPos` are the centres of the first and last
*walkable* tiles — the playable area, inside the wall ring — while
`GetGridWidth` counts the ring too. Scaling the playable span across the full
width squashed the player into the wall columns near every edge, so the player's
own tile came back **solid on 27% of observations**, which is impossible. The
playable span covers the interior only: columns 1..width-2, i.e. width-3 tile
*steps* between two tile centres.

`centred` could never have caught this — the probe's own tile derivation
mirrored the encoder's, so both were wrong together and agreed 18000/18000.
**`standable` is the load-bearing check**, because it tests against physics
rather than against the code: a player cannot be standing inside a rock. Keep
that shape in mind when writing the next probe.

A real 2x1 room (9x28 raw grid) turned up on the second run and passed, so the
large-room path is verified on live data rather than only synthetically.

**Why.** The room grid is indexed by room tile and goes through a flat
`Linear(405, hidden)`, while everything the agent aims and steers with is
player-relative — and the entity branch is *pooled* before the trunk. So "rock
at room tile (4, 9)" and "enemy at offset (+140, 0)" only ever met as two
unrelated 128-dim summaries. Measured on floor-v10 by
`diagnose_grid_use.py`, holding the obstacle *count* fixed and moving only its
position:

```
clear    -> blocking    0.0283      (total variation, 0 = identical policy)
blocking -> irrelevant  0.0188      <-- same rock count, different place
clear    -> spikes      0.0216
```

Moving a rock from directly on the line of fire to a far corner changed the
policy by under 2%, and spikes against the player changed movement by 2%. The
critic noticed a little (value 4.04 -> 3.65); the actor did not. This is the
same defect the entity encoder had, already worth **+37.5 points of success**
when its positions were made player-relative — the grid never got that fix.

It also explains the observed play directly: enemy due east, and the policy
fires `shoot_y` **up** — it closes the horizontal gap and shoots vertically, so
it always attacks from below and never accounts for what is in the way.

**What was added.** `EGO_RADIUS = 4`, a 9x9x3 window centred on the player's
tile (`EGO_FEATURES = 243`), built from the **raw** tile array rather than the
downsampled 15x9 frame so local detail stays exact in a large room. Off-map
reads as **solid**, not empty — zero-filling would tell the agent it can walk
out through the edge. The room-absolute grid is kept as well, because it carries
where the player is *within* the room, which an egocentric window cannot.

**The resume is bit-identical**, verified numerically rather than argued: the
trunk widens 640 -> 768 with the new columns zeroed, so feeding a completely
different `ego_grid` changes the logits by exactly 0.000e+00 at init. 23 copied,
1 widened, 2 fresh. `PrivilegedCritic` grew the same branch, since the pixel
critic is defined as the teacher's network minus its action heads and a test
pins that.

**Cost:** 13.8us a call, ~11% of one core at 20 instances. It was 26.6us until
`np.pad` turned out to be **11.7us of that** on a 9x15 array — 45% of the encode,
entirely generic-path overhead, against ~1us for allocate-and-assign.

**Not fixed by this:** nothing rewards retreating, which is the fourth observed
behaviour. `damage_taken -1.00` is the only pressure and the learned strategy
requires *approaching* to align, so fleeing has no upside at all.

## floor-v9 + v9b: best run yet, and still climbing at the end

1,003,520 effective steps (v9 died to a crashed instance at 717k; v9b resumed —
add 716,800 to v9b's `global_step`).

| band | return | success | rooms_cleared | rooms_seen | backtrack |
| --- | ---: | ---: | ---: | ---: | ---: |
| 716k | +8.11 | 0.336 | 0.51 | 1.48 | 1.05 |
| 866k | +11.30 | 0.450 | 0.51 | 1.70 | 1.12 |
| 916k | +9.74 | 0.466 | 0.68 | 1.77 | 1.51 |
| 966k–1M | **+11.46** | **0.518** | **0.81** | **1.99** | 2.01 |

`rooms_cleared` 0.52 (v8) -> **0.81**, and every column was still rising when it
stopped. No dead axes; `shoot_x` 0.397 and `shoot_y` 0.485 are both committed,
and movement is now strongly so (`move_x` 0.194, `move_y` 0.123).

**`difficulty` is still 0.000** — the bar is 0.72 success and it reached 0.518.
Closer than ever and not there.

### The stranding question, answered — with one caveat

The end-reason logging landed in v9b. Of every episode:

- **54% end on the idle limit, 46% on death.** Nothing ever hit the 3000-step
  cap, and no instance dropped.
- An idle-ended attempt runs **666 steps**, against a 491-step mean — so it
  explores for ~166 and then stalls for the full 500.
- **~55% of all agent steps** are spent with the idle clock running, i.e. making
  no curriculum-scored progress.
- **28% of give-up episodes ended in a room with no unvisited door**, rising
  0.225 -> 0.320 across the run, alongside `backtrack_ratio` 1.315 -> 1.482 and
  `rooms_seen` 1.697 -> 1.821.

So the shuttling is real, it costs real throughput, and it grows as the agent
explores deeper — a ceiling arriving later, not one being grown out of.

**Caveat: `stranded` is confounded and 28% is an upper bound.** Isaac locks the
doors during a fight and `door_potential` skips locked doors, so an attempt that
stalls mid-combat reads as stranded even though the room is not exhausted.
`enemies_alive` is now recorded at episode end and `exhausted` (no doors *and*
no enemies) is logged from the next run on — that is the number that justifies
building a non-local potential, and it will be lower than 0.28. Do not spend the
work until it is measured.

**Also fixed:** absent end-reasons were omitted rather than logged as 0.0, which
made averaging each key over the updates where it appeared give shares summing
to 1.047. All four are now always logged.

## floor-v9's change: semantic entity identity, built and probed

```bash
.venv/Scripts/python.exe -m isaac_ai train-floor \
  --instances 20 --steps 1000000 --run-name floor-v9 \
  --resume runs/floor-v8/policy.pt --entropy-target 0.5
```

`ENTITY_FEATURES` 17 -> 22: five semantic flags — `consumable`, `pedestal`,
`chest`, `hostile`, `flying` — resolved in Lua from the game's own constants,
flags rather than a variant id for the same reason door categories are flags.
Before them a chest, a coin, a heart and an item pedestal were the identical
vector, and a spiked chest or mimic looked exactly like a normal chest.

**This does not need a fresh retrain.** The flags are *appended*, and
`_warm_start`'s widening branch keeps the old columns and zeroes the new ones,
so a resumed policy starts out computing exactly what v8 computed. Verified
against the real code path: **23 copied, 1 widened (17->22), 0 fresh**, leading
columns identical and new columns exactly zero. Named indices (`ENTITY_CLOSING`
… `ENTITY_FLYING`) replace the positional-from-the-end indexing the tests used,
which broke silently on the append and read a zeroed flag rather than raising.

**Appending is load-bearing.** Inserting a feature anywhere earlier shifts every
later column and scrambles the transfer with no error at all.

**Episodes now log why they ended** (from floor-v10 on; v9 predates it).
`ended_died` / `ended_idle` / `ended_timeout` / `ended_dropped` as shares,
`episode_steps` and `idle_episode_steps` as mean lengths, and `stranded` — the
share of give-up episodes that ended in a room with **no unvisited door**.

That last one is the number that decides whether a non-local potential is worth
building. `door_potential` only looks at the current room and returns 0.0 when
nothing there is unvisited — flat, "no pull in any direction" by its own
docstring. So a one-room backtrack has a real gradient (stepping into a room
with a live door raises the potential immediately) while a **multi-room**
backtrack across emptied territory has none at all, and the agent can only find
its way out by random walk before `idle_limit` truncates at 500 steps. Observed
by eye on v9 and consistent with `backtrack_ratio` rising 1.25 -> 1.34 -> 1.70
across the run: this gets *worse* as the agent explores deeper, so it is a
ceiling arriving later rather than one being grown out of.

The oscillation is not an exploit — `new_room` pays only for rooms not already
in the episode's `_visited` set, shaping is undiscounted so any loop telescopes
to exactly zero, and `_idle_steps` resets only on a genuine new room, clear or
descent. `room_cleared` fires on `MC_PRE_SPAWN_CLEAN_AWARD`, so re-entering a
cleared room does not re-arm it. The reseed the agent ends up at *is* the
designed abandonment firing correctly.

**Fixed alongside it:** a failed instance had `terminated` forced true on every
step, so it emitted a fresh zero-return episode *every step* and would have
flooded the metrics window. It never mattered while a dropped instance took the
whole trainer down with it; now that the bridge fix makes a crash cost only one
instance, it would have.

**Chest variants are derived from the enum by name, not listed.** Listing them
by hand was tried and got it wrong: it missed `PICKUP_LOCKEDCHEST` (60) and
`PICKUP_REDCHEST` (360), which were **41% of every pickup the fleet observed**.
The flag read zero, which is indistinguishable from "no chests appeared" — the
project's signature failure, this time in a mapping. There are thirteen chest
variants in this build. `probe_entity_flags.py` now tallies raw variants and
escalates any that produce *no* flag at all, which is the only symptom that can
be a mapping fault rather than an unlucky sample.

**Why it stopped.** At 15:15:13 one `isaac-ng.exe` took an access violation
(`0xc0000005`). The kernel dropped that socket with an RST, `recv_into` raised
`ConnectionResetError`, and because that is an `OSError` rather than a
`BridgeError` it went straight past `_receive_all`'s handler and killed the
trainer, which then ran `fleet.shutdown()` and closed all twenty instances —
the `ISAAC_AI fatal: receive: closed` at the end of `log.txt`. `send` had
wrapped `OSError` since the beginning; only the read path had not. Every prior
shutdown was orderly, so it never showed. Fixed in `bridge.py`, pinned by
`TestBridgeFailureIsRecoverable` (RST *and* clean close). A crashed instance now
just drops the fleet to 19 and the run carries on.

**`floor-v6` finished and regressed.** At 1M steps: success 0.10 against v5's
0.375, mean return −1.879, 838 deaths, `difficulty` pinned at 0.000 the entire
run. v6 was the first floor run where dying was actually charged — v4's penalty
never fired and v5 counted deaths without paying them — which makes the penalty
the one clean single-variable suspect. At −10.00 a death costs more than a whole
good episode earns (~+8: a new room, a clear, one room of shaping), and PPO
normalises advantages per minibatch, so one such spike inflates the batch
deviation and squashes the shaping differences that carry the gradient. Same
shape as `damage_taken −2.50`, already tried and reverted for making the
dominant gradient "do not engage".

`floor_death` is a separate config key from `death` on purpose: combat counts
deaths but still pays through `compute_reward`'s `events["died"]`, which the mod
almost never delivers, so tuning the floor penalty must not silently redefine
the environment every combat run was measured in.

**What v6 actually learned: navigation, not combat.** Movement commits
(`move_x` entropy 0.686) while both shoot axes sit on the uniform ceiling and
fire a constant `right` — 6/16 correct placements are exactly the six rightward
ones. Confirmed by eye too: it walks to doors and cannot fight.

**`difficulty` has never left 0.000 in any floor run**, which pins
`target_rooms` at its floor of 1: *clear one room, anywhere, at any point*.
Success crossing 0.72 would advance the floor curriculum for the first time.

## The strategic picture

Combat and navigation were trained as separate synthetic tasks. **Nearly every
bug this project has hit came from that scaffolding, not from the game** — the
spawn exploit, wall-camping, the aim collapse, the room-size confound, the
curriculum oscillation. The game generates perfectly good content and we kept
building a fake version of it and then debugging the fake version.

Floors are the correction: an episode is one attempt at a real floor, which
*contains* combat, so no separate combat task is needed. Throughput makes it
viable — 648k agent steps/hour, ~280 floor episodes/hour. Full runs to Mom would
be ~32 episodes/hour, which is the wall the previous project died on.

## What works

**Harness.** 20 instances, synchronous stepping at Isaac's 30 Hz. Measured
469.6 game ticks/s, 234.8 agent steps/s — +54% over 12 instances for 67% more
instances. Stable for hour-long runs.

**Phase 3 pixel pipeline, end to end.** Capture is free (360.0 game ticks/s with
it on against 359.9 off; 0 repeated frames in 1200 reads). Distillation
transfers — `distill-v5` reached 0.814 move agreement. RL fine-tuning with the
asymmetric critic trains. `capture.py`, `pixel_policy.py`, `distill.py`,
`pixel_ppo.py`.

**Floor shaping**, after a fix worth the whole task — see below.

## What is open

**The pixel student plateaus at difficulty ~0.42 regardless of anything.** Not
the teacher (a teacher at 0.75 produced no better student than one at 0.57), not
resolution (240x135 raised imitation to 0.814 agreement and moved performance
not at all), not RL fine-tuning (two runs, +0.02 and +0.00 difficulty). Untested
lever: `entropy_coef` in `PPOConfig`.

**One shoot axis is always sacrificed.** Every combat policy trained here aligns
on one axis by *moving* and fires along the other, leaving one head pinned at the
ln(3)=1.099 uniform ceiling. combat-v5 through v7 could not shoot horizontally
at all and it went unnoticed for three runs. Whether this is a defect or a
legitimate decomposition is genuinely unresolved — the argument that it *is* a
defect is that some rooms cannot be solved shooting one direction.

**Grid hazards vs. floors.** The obstacle grid is new and untested in a finished
run. `floor-v6` is the test.

## Checkpoints — all of them are invalid

`ENTITY_FEATURES` went 15 -> 17 (closing/tangential speed) and `GRID_FEATURES`
(405) was added, so **every checkpoint predating today fails to load**, including
`combat-v6`, which is still the best combat teacher ever trained (difficulty
0.73). Check shapes against `ActorCritic().state_dict()` before planning
anything that depends on an old policy. `distill.load_teacher` refuses a partial
fit deliberately: a teacher with a reinitialised trunk still emits confident
logits and a student would spend the run imitating noise.

`diagnose_student.load_student` loads a student's **actor only** and ignores a
stale critic, so pixel students remain comparable across observation changes.

## The recurring failure mode

**A metric or reward counts an event rather than actual progress, the curves
read as "not learning", and the agent is quietly gaming the measure.** Nine
instances now. Every single one was caught by watching the game, never by the
metrics.

- floor `new_room` paid per transition -> shuttling between two rooms
- nav curriculum required N *crossings* -> one door, N times
- `backtrack_ratio` scored zero-room episodes as 0.0 -> inactivity read as good
- summed entropy hid that a shoot axis was dead for three runs and two students
- `move_agreement` compared two *sampled* actions, capping a perfect student at
  ~0.5, so a run 41% of the way there read as "stalled at chance"
- enemies spawned relative to the player, so the agent chose where they appeared
  (worth 0.48 win rate, inherited by every student distilled from it)
- door shaping paid **-2.885 for walking through a door**, which is what
  plateaued floor-v1/v2/v3
- the `-10` death penalty never fired, and `deaths` was never counted at all
- room jumping put 18% of encounters in a 4x-area room the curriculum could not
  model, halving measured difficulty
- **an instrument had the same disease.** `probe_heads.py` synthesised a room
  with `doors: []` and no grid, so it reported floor-v6's movement heads as
  abandoned (`move_x` 1.091, a coin flip) while the agent was visibly walking to
  doors. A floor policy's move heads are trained almost entirely on where the
  doors are; probing them in a doorless room measures nothing. With doors and a
  walled grid the same checkpoint reads `move_x` 0.686 — committed. The shoot
  finding survived the correction; the movement one was an artefact.
  An all-zero grid is not neutral either: `encode_grid` returns zeros for a room
  it could not read, which means "open floor everywhere", which never occurs.
- reseed's acquisition check watched five derived stats
  (damage/speed/range/tear_delay/max_hearts), so a familiar, a tear modifier, a
  trinket, a held card or any consumable read as pristine and rode across every
  later episode — and `bombs`/`keys`/`coins` are three of the scalar inputs the
  network is shown. The baseline was also a single shared tuple seeded lazily
  from whichever instance first finished an episode alive.
- **the diagnosis itself went unmeasured for six runs.** v14 through v18 were all
  spent on perception — grid planes, a conv encoder, a type embedding, raw entity
  identity, a creep plane — on the assumption that the agent could not see. When
  it was finally measured on v18, the entity branch responded to enemy position
  by 45% and the trunk by 14%; the heads discarded it. The check that would have
  shown this takes seconds and needs no game, and v15 explicitly planned to run
  it "at ~200k rather than at the end" and never did. **A cheap instrument that
  is not run is worth exactly as much as one that does not exist.**

**When something looks like a plateau, watch the agent play before touching a
coefficient — and check the instrument is asking a question the task can
answer.**

## Instruments built for it

Run these instead of inferring from training curves.

| script | answers |
| --- | --- |
| `probe_heads.py` | does a checkpoint aim? 8 directions x 2 ranges, seconds, no game |
| `diagnose_student.py` | student vs teacher vs random at *fixed* difficulty; `--student-b` compares two checkpoints on one fleet |
| `play_encounter.py` | puts a human on the same encounters — is the task even fair? |
| `probe_capture.py` / `probe_pixels.py` | capture cost, staleness, distinctness, motion |
| `probe_reseed.py` | is a cheap reset safe? **one instance only**, and its "distinct floors" counts distinct room *totals* (room index is always 84) |
| `probe_floor_reset.py` | does the reset send the right episodes down the right path, at real fleet size? fields reported, baselines per instance, acquisitions routed to restart |
| `probe_entity_flags.py` | do the semantic flags resolve and fire in real play? Tallies raw pickup variants and escalates any producing **no** flag — the one symptom that is a mapping fault rather than rarity |
| `diagnose_shoot_axis.py` | is a dead shoot axis decomposition or abandonment? 16 angles x 2 ranges, no game. Reports **distinct move responses** — a constant scores ~50% alignment by construction, so the rates alone cannot tell you. Sweeps doors unvisited/visited/none as the control |
| `probe_combat_rooms.py` / `probe_stuck.py` | room variety safety; does the teleport strand the player |

Per-axis entropies (`entropy_move_x` … `entropy_shoot_y`) are logged by both
trainers — a summed figure cannot distinguish "half converged" from "one axis
frozen and one abandoned".

## Fixes made today, and what was reverted

**Kept:** floor potential measured in floor units (below); death penalty paid on
`is_dead`; `deaths` counters; obstacle grid; reseed resets; per-axis entropy;
`rooms_cleared` logging; reposition-before-spawn; closing/tangential speed
features; 20 instances.

**The floor fix that mattered.** `door_potential` collapsed from 0.990 to 0.019
the moment you stepped through a door — that door became "visited" and the next
was across the room — paying **-3.885** against a `new_room` bonus of +1.00.
`floor_potential` now measures `rooms_visited + closeness`, so the room gained
cancels the closeness lost and the transition is free. Shaping is applied
undiscounted (gamma=1) to avoid the `(1-gamma)*phi` leak, and deliberately not
normalised by `rooms_total` because that shrinks the per-step gradient to the
same order as the step penalty — the regime that killed nav-v1.

**Reverted, each measured worse than the baseline:** `damage_taken -2.50`
(combined with closer spawns it made encounters net-negative and the dominant
gradient became "do not engage"); spawn distance `130-70*d` (halved achievable
difficulty); room jumping (the 4x-area confound).

**Resets.** `reseed` regenerates the floor in 0.27s against `restart`'s 0.91s.
Used for everything except a death (a dead player cannot be reseeded around) or
an instance that has acquired something — reseed keeps items by design, so a
pickup would compound across every later episode. Restarts fell 1.70 -> 0.61 per
1000 steps and throughput rose to 214 steps/s.

The acquisition check now reads what the mod reports directly — `collectibles`,
both trinkets, the pocket card and pill, the active item and the three
consumables — with the old derived stats kept only as a backstop for something
that moves them without adding an item (a devil deal's heart cost). One baseline
**per instance**, seeded from the observation immediately after that instance's
own full restart, which is the only moment it is known pristine. Verified at 20
instances by `probe_floor_reset.py`.

Expect restarts to rise as the agent gets better, since clearing rooms drops
pickups — the correct trade, but it eats the reseed saving. Note also that
`reset_done` runs `_restart` then `_reseed` sequentially, so a batch containing
even one death pays both costs: 10 restart + 10 reseed measured 1.24s against
1.09s for restarting all 20.

## Conventions

- `.venv`; **91 tests**, run with
  `.venv/Scripts/python.exe tests/test_learning.py` (unittest, not pytest,
  which is not installed).
- **`PYTHONPATH=src` is no longer needed and must not be written into commands.**
  The project is not packaged, so every doc command used to carry a
  `PYTHONPATH=src` prefix — which is bash syntax and a hard error in PowerShell,
  the shell actually used here (`PYTHONPATH=src : The term ... is not
  recognized`). `.venv/Lib/site-packages/isaac_ai.pth` now puts `src` on the
  path for that interpreter in any shell. It holds an **absolute** path, so
  recreating the venv or moving the project means rewriting that one line.
- Every environment or mod change gets a live probe before a real run, at the
  **real fleet size** — the capture probe deadlocked at 12 instances and passed
  at 2.
- Long-running scripts must line-buffer stdout, or a wedged run is diagnosed
  from window coordinates.
- **A dead player freezes the whole fleet, not just its own instance.** Isaac
  stops running mod callbacks once the game-over screen takes over, so that
  instance can never answer — and `_receive_all` reads sequentially with a
  blocking socket, so all twenty stall behind it for the full 45s
  `GAMEPLAY_TIMEOUT_SECONDS` before it is marked failed, then resume. Anything
  that steps the env **must** call `reset_done` on the done mask; `is_dead` is
  visible during the death animation, which is the window a restart has to land
  in. A probe that stepped without resetting duly froze the fleet with three
  windows sitting on "Dear Diary".
- **The game itself crashes occasionally** — one access violation in ~50 minutes
  across 20 instances, on floor-v7. The harness now survives it. When a run dies
  anyway, the three places to look are the trainer's traceback, `log.txt`'s last
  line, and the Application event log filtered to `isaac-ng.exe` (an APPCRASH
  with `0xc0000005` is the game faulting, not the agent).
- Six workshop mods load into every instance (boss bars, planetarium chance,
  specialist-for-good-items and so on). They are display-only and have been
  present for every run; not a confound.
