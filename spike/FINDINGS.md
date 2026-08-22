# Phase 0 findings

Measured on the host (Ryzen 9 7950X, 32 GB, RTX 4080) against
The Binding of Isaac: Repentance+ v1.9.7.17, appid 250900.

Every claim below was tested, not assumed.

## Results

| Question | Answer | Evidence |
| --- | --- | --- |
| Multiple instances on one host? | **Yes, 12 concurrent, no degradation** | Steam blocks re-launch through the client UI only; launching `isaac-ng.exe` directly with `steam_appid.txt` present works. 8 instances = 4.0 GB RAM, ~30% of 32 threads. |
| Does `--luadebug` unsandbox Lua? | **Yes** | `io`, `os`, `os.execute`, `io.popen`, `package.loadlib` all present. Lua 5.3. |
| Sockets from the mod? | **Yes, luasocket already ships with the game** | `require("socket")` and `require("socket.core")` both succeed. No DLL sourcing needed. |
| Can the mod block the update loop for synchronous stepping? | **Yes** | Blocking `tcp:receive` inside `MC_POST_UPDATE`; 900/900 ticks closed the loop, no hangs, no dropped ticks. |
| Can the mod drive the player without keystrokes? | **Yes** | `MC_INPUT_ACTION` + `InputHook.GET_ACTION_VALUE` moved the player 7162 px per run in all 12 unfocused, occluded instances. |
| Frame capture from unfocused/occluded windows? | **Yes** | `PrintWindow` with `PW_RENDERFULLCONTENT` returns correct 480x291 client bitmaps while fully covered by another window. |
| Boot straight into gameplay? | **Yes** | `--set-stage=1 --set-stage-type=0` generates a floor and drops into `Room 1.2 (Start Room)`. No menu automation at all. |
| Per-instance identity? | **Yes** | The mod reads `ISAAC_AI_PORT` / `ISAAC_AI_INSTANCE` via `os.getenv`. |
| Real tick rate? | **29.7 update ticks/s, 59 fps render** | 300 updates in 10.101 s of `os.clock`. |

## Throughput

12 instances, each closing the loop on every game tick:

```
[0..11]  900 steps in 29.98s = 30.0 steps/s each
AGGREGATE: 360.2 agent steps/s   ->   1,296,634 steps/hour
```

Server turnaround was 33-49 us, so the game tick is the only limit. Per-instance
throughput is pinned at Isaac's fixed 30 Hz logic rate and cannot be raised;
scaling is purely instance count. CPU headroom suggests 20+ instances are
feasible, but 12 leaves room for the GPU trainer and normal desktop use.

For comparison, the previous architecture's ceiling was 3 VMs x 10 steps/s = 30
steps/s, and real sessions completed 768-15,000 steps.

## Required options.ini settings

Original backed up to `backups/options.ini.original`.

| Setting | Value | Why |
| --- | --- | --- |
| `PauseOnFocusLost` | 0 | Otherwise every unfocused instance freezes. Critical. |
| `Fullscreen` | 0 | Many windowed instances. |
| `VSync` | 0 | Render loop should not gate on the display. |
| `SteamCloud` | 0 | Avoids save-sync contention between instances. |
| `MaxRenderScale` | 1 | 480x270 native render, smaller windows, less GPU. |
| `EnableDebugConsole` | 1 | Already on; needed for `Isaac.ExecuteCommand`. |

## Save files and unlock state

Isaac stores unlock progress in `persistentgamedataN.dat`. With `SteamCloud=1`
those live in the Steam Cloud remote folder, **not** in `My Games`; with
`SteamCloud=0` the game uses `My Games` and creates fresh saves if none exist.
Setting `SteamCloud=0` during Phase 0 therefore created three brand-new local
saves rather than touching the real one.

| Save | Size | Meaning |
| --- | --- | --- |
| Fresh / zero unlocks | 4068 bytes | What the training instances are using |
| Real personal save (slot 2) | 14844 bytes | Intact in Steam Cloud, dated before Phase 0 |

Backed up to `backups/save-snapshot-20260812/` from both sources.

**The D6 is not evidence of unlock progress.** Probed on a verified-empty
4068-byte save: `has_D6=true`, `active_item=105` (COLLECTIBLE_D6),
`collectible_count=1`. Isaac starts with the D6 on a zero-unlock save in
Repentance+, so it cannot be used as a signal that a save has progress. Use the
file size instead.

`Game():AchievementUnlocksDisallowed()` does **not** exist in the vanilla
Repentance+ API (`api_missing`) — it is a REPENTOGON addition. So whether a
modded training run can earn unlocks is still unverified, which is one reason
not to depend on the game's own unlock system for curriculum progression.

## Run entry: `--set-stage` is NOT usable, menus need real input

Correction to the Phase 0 table: `--set-stage=1` boots into a run, but **that run
is not a faithful game state.** Isaac spawns holding the D6 (`active_item=105`),
which is an unlock-gated starting item that a zero-unlock save does not grant. A
console `Isaac.ExecuteCommand("restart")` does **not** clean it up — the restarted
run still has the D6 and still reports `difficulty=0`. A menu-started run on the
same save correctly has no active item. So `--set-stage` appears to run with
unlocks granted and difficulty pinned to Normal, and must not be used for
training.

Input findings, corrected by measurement:

| Path | Works? |
| --- | --- |
| Gameplay input via mod `MC_INPUT_ACTION` | **Yes**, unfocused and occluded |
| Menu *confirm* (SPACE) via `SendInput` + focus | **Yes** |
| Menu *navigation* (W/S, arrows) via `SendInput` + focus | **No** — cursor never moves |
| Any key via `PostMessage`/`SendMessage` unfocused | **No** |

An earlier note claiming unfocused `PostMessage` drove the menus was wrong: that
window had just launched and still held focus. Menus need real focus, and even
then only the confirm key registers. Arrow keys are ATTACK in Isaac; menu
movement is WASD — but neither moves the main-menu cursor from synthetic input.

**Consequence — the CONTINUE route.** Menu navigation is not needed at all:
from a cold launch, three identical SPACE presses go title -> file select ->
CONTINUE, landing directly in the in-progress run. The mod then owns the session
and `Isaac.ExecuteCommand("restart")` produces each fresh episode. This requires
the save to be prepared once, by hand, with the desired file, character,
difficulty, and unlock tier, and a run left in progress. That prepared savedata
directory becomes the pristine snapshot every training session restores.

Still to verify: whether `restart` preserves Hard mode. Needs a hard-mode run to
exist first.

## Known open issues

- **Shared savedata directory.** All instances write the same `log.txt`,
  `options.ini`, and save files. Log contention is now irrelevant (we no longer
  scrape it), but concurrent save/options writes need handling — likely by
  never exiting cleanly and treating persistent progress as disposable.
- **Instance count ceiling untested above 12.** RAM is the likely binding
  constraint at ~500 MB per instance.
- **REPENTOGON not evaluated yet.** Deferred; the vanilla API already covers
  everything Phase 0 needed.

## Artifacts

- `bridge_server.py` — the synchronous-stepping probe server.
- Probe mods in the game's `mods/` folder, both left **disabled**:
  `zz_luadebug_probe`, `zz_socket_bridge`. The old `isaac_ai_telemetry` mod is
  also disabled.
