"""Training encounters built from real game content.

The previous project hand-authored a handful of lesson rooms and trained to
convergence on each. Policies mastered the lesson and generalized to nothing.
This instead samples encounters from a distribution over real enemy types,
counts and positions, and moves the distribution as the agent improves — so the
agent is always training near the edge of its ability rather than on a fixed
scene it can memorize.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import random


@dataclass(frozen=True)
class EnemyKind:
    """One spawnable enemy. `threat` orders the difficulty ramp."""

    name: str
    type_id: int
    variant: int
    threat: float


# Basement-appropriate roster. Verified to spawn; ordered roughly by how hard
# they are for a policy that cannot yet aim well.
ROSTER: tuple[EnemyKind, ...] = (
    EnemyKind("fly", 13, 0, 0.5),
    EnemyKind("pooter", 14, 0, 0.8),
    EnemyKind("gaper", 10, 0, 1.0),
    EnemyKind("frowning_gaper", 10, 1, 1.1),
    EnemyKind("horf", 26, 0, 1.2),
    EnemyKind("clotty", 15, 0, 1.3),
    EnemyKind("attack_fly", 18, 0, 1.4),
    EnemyKind("hopper", 29, 0, 1.5),
    EnemyKind("mulligan", 16, 0, 1.6),
    EnemyKind("charger", 23, 0, 2.0),
)


@dataclass
class Encounter:
    """A concrete encounter, ready to send to an instance."""

    groups: list[dict[str, int]]
    min_distance: int
    enemy_count: int
    difficulty: float

    def to_command(self) -> dict:
        return {
            "t": "scenario",
            "enemies": self.groups,
            "min_distance": self.min_distance,
            "heal": True,
            # Spawns are placed at least min_distance from the player, and
            # encounters rebuild in place — so without moving the player first,
            # where it stood at the end of the last fight decides where the next
            # one may appear. Two combat runs learned to hug a wall for exactly
            # that reason.
            "reposition": True,
            # **On.** Off previously because jumping rooms was judged on whether
            # it varied *geometry* — it did not (plain rooms are nearly all the
            # same 520x280 shape) — while landing ~18% of encounters in a 2x2
            # room of four times the area. The curriculum has one difficulty
            # scalar and cannot separate "hard because there are many enemies"
            # from "hard because the room is enormous", so it lowered the enemy
            # count to hold success at target: difficulty settled at 0.73 before
            # room jumping and 0.33-0.39 after, across three runs.
            #
            # Both halves of that have changed.
            #
            # The shape filter now holds completely. `probe_room_contents.py`
            # sampled 160 jumped rooms and got `ROOMSHAPE_1x1` on **160 of 160**,
            # so the 4x-area confound is gone rather than merely reduced.
            #
            # And the thing being asked for is no longer geometry, it is
            # **obstacles**. The encounter is built where the player stands,
            # which is the start room, and the start room is bare — measured, 0
            # interior blocking tiles across 160 samples, every time. Jumped
            # rooms carry a median of 8 and a mean of 12.3, 13.5% of the interior,
            # with only 23% empty.
            #
            # That matters because `diagnose_cornered.py` measured blocked
            # retreat at **5.25x** enriched at death on floor-v23 (33.3% against
            # a 6.3% baseline). A fight learned in a room where retreat always
            # succeeds is learning the wrong tactic for the case that actually
            # kills, and an empty room cannot teach otherwise.
            #
            # Watch `difficulty` for the old failure returning in a new form:
            # obstacle count ranges 0-56 and the dial still has one scalar, so if
            # it oscillates or sags the way it did under room-size variance, this
            # is the first suspect.
            "new_room": True,
        }


@dataclass
class CombatCurriculum:
    """Samples encounters and tracks whether the agent is keeping up.

    Difficulty is a single scalar in [0, 1] that scales both how many enemies
    appear and how dangerous they are allowed to be. It rises when the agent
    wins comfortably and falls when it starts losing, which keeps the training
    signal informative instead of saturating at "always dies" or "always wins".
    """

    difficulty: float = 0.0
    target_success: float = 0.6
    # Per policy update, not per episode: a full sweep of the ramp should take
    # tens of updates so the policy has time to actually learn each level.
    adjust_rate: float = 0.03
    deadband: float = 0.12
    window: int = 60
    # New episodes required between difficulty steps. Derived from the window
    # rather than fixed, or a smaller window than this number would freeze the
    # ramp completely. A quarter means consecutive steps rest on mostly fresh
    # evidence, which is what keeps this a controller rather than an open-loop
    # integrator.
    refresh: int | None = None
    min_enemies: int = 1
    # Raised from 6 after the v3 run pinned the dial at its ceiling, leaving the
    # curriculum unable to measure any further improvement.
    max_enemies: int = 10
    seed: int | None = None

    _outcomes: list[bool] = field(default_factory=list)
    _recorded: int = 0
    _last_advance: int = 0
    _rng: random.Random = field(init=False)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        if self.refresh is None:
            self.refresh = max(1, self.window // 4)

    @property
    def success_rate(self) -> float:
        if not self._outcomes:
            return 0.0
        return sum(self._outcomes) / len(self._outcomes)

    def available(self) -> list[EnemyKind]:
        """Enemies unlocked at the current difficulty.

        The roster opens up gradually so early training is not dominated by the
        one enemy type the agent cannot handle yet.
        """
        ceiling = 0.5 + self.difficulty * 1.7
        allowed = [kind for kind in ROSTER if kind.threat <= ceiling]
        return allowed or [ROSTER[0]]

    def sample(self) -> Encounter:
        pool = self.available()
        span = self.max_enemies - self.min_enemies
        count = self.min_enemies + int(round(span * self.difficulty))
        count = max(self.min_enemies, min(self.max_enemies, count))

        counts: dict[tuple[int, int], int] = {}
        for _ in range(count):
            kind = self._rng.choice(pool)
            key = (kind.type_id, kind.variant)
            counts[key] = counts.get(key, 0) + 1

        groups = [{"type": type_id, "variant": variant, "subtype": 0, "count": n}
                  for (type_id, variant), n in counts.items()]

        # Crowd the agent a little more as it gets better.
        #
        # 130-70*difficulty was tried, on the theory that spawning inside tear
        # range would stop the agent walking to a wall first and force it to
        # aim on both axes. It did unpin the horizontal shoot axis briefly
        # (combat-v10: 1.099 -> 0.87) and then did not hold: combat-v12, a clean
        # run at this reward, settled at difficulty 0.39 against combat-v6's
        # 0.73 and still sacrificed a shoot axis. Crowding costs roughly half
        # the achievable difficulty and buys nothing durable.
        min_distance = int(220 - 60 * self.difficulty)

        return Encounter(groups=groups, min_distance=min_distance,
                         enemy_count=count, difficulty=self.difficulty)

    def record(self, success: bool) -> None:
        """Record an outcome. Deliberately does not move difficulty.

        Encounters last only a couple of seconds, so a twelve-instance fleet
        produces dozens of episodes per policy update. Adjusting on every one of
        them drove difficulty by up to a full unit per update and made the ramp
        oscillate between trivial and impossible faster than the policy could
        learn either. Difficulty now moves once per update, via `advance`.
        """
        self._outcomes.append(success)
        self._recorded += 1
        if len(self._outcomes) > self.window:
            self._outcomes.pop(0)

    def advance(self) -> None:
        """Take one difficulty step. Call once per policy update, not per episode.

        A step is only taken once enough *new* episodes have landed since the
        last one. Stepping on every call instead ties the ramp rate to how often
        the trainer happens to update rather than to how fast evidence arrives,
        and the loop oscillates whenever episodes are slower than updates. It is
        not theoretical: during distillation the student drove 90% of episodes
        and only the other 10% fed this window, so the ramp ran from 0.00 to
        1.00 and back, full scale, while the window was still catching up.
        """
        if len(self._outcomes) < self.window:
            return
        if self._recorded - self._last_advance < self.refresh:
            return
        self._last_advance = self._recorded

        rate = self.success_rate
        # A deadband around the target stops the ramp hunting back and forth
        # when the agent is sitting right where we want it.
        if rate > self.target_success + self.deadband:
            self.difficulty = min(1.0, self.difficulty + self.adjust_rate)
        elif rate < self.target_success - self.deadband:
            self.difficulty = max(0.0, self.difficulty - self.adjust_rate)

    def state(self) -> dict:
        return {
            "difficulty": round(self.difficulty, 3),
            "success_rate": round(self.success_rate, 3),
            "enemy_types": len(self.available()),
            "samples": len(self._outcomes),
        }
