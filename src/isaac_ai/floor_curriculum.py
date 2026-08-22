"""Difficulty ramp for floor progression.

On real floors we do not author the encounters — the game does. So the dial is
not "how many enemies" but **how much of the floor counts as success**: clear one
room, then two, then four. The agent faces whatever the game generated either
way; what moves is the bar it has to reach.

Same control discipline as the combat curriculum, for the same reason: advance
once per policy update, never per episode, with a deadband so the ramp does not
hunt around the target.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class FloorCurriculum:
    difficulty: float = 0.0
    target_success: float = 0.6
    adjust_rate: float = 0.03
    deadband: float = 0.12
    # Smaller than the combat window (60): a floor attempt lasts hundreds of
    # steps rather than ~30, so episodes arrive far more slowly and a large
    # window would leave the ramp reacting to very stale performance.
    window: int = 40

    min_rooms: int = 1
    max_rooms: int = 8
    # Agent steps with no cleared room, no new room and no descent before the
    # attempt is abandoned. Cut from 900 after floor-v1 spent half its episodes
    # running this clock out — a shorter leash wastes less game time on attempts
    # that have already stalled.
    idle_limit: int = 500

    _outcomes: list[bool] = field(default_factory=list)
    _rooms: list[int] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self._outcomes:
            return 0.0
        return sum(self._outcomes) / len(self._outcomes)

    @property
    def mean_rooms(self) -> float:
        if not self._rooms:
            return 0.0
        return sum(self._rooms) / len(self._rooms)

    def target_rooms(self) -> int:
        span = self.max_rooms - self.min_rooms
        return max(self.min_rooms,
                   min(self.max_rooms,
                       self.min_rooms + int(round(span * self.difficulty))))

    def record(self, success: bool, rooms_cleared: int) -> None:
        self._outcomes.append(success)
        self._rooms.append(rooms_cleared)
        if len(self._outcomes) > self.window:
            self._outcomes.pop(0)
            self._rooms.pop(0)

    def advance(self) -> None:
        if len(self._outcomes) < self.window:
            return
        rate = self.success_rate
        if rate > self.target_success + self.deadband:
            self.difficulty = min(1.0, self.difficulty + self.adjust_rate)
        elif rate < self.target_success - self.deadband:
            self.difficulty = max(0.0, self.difficulty - self.adjust_rate)

    def state(self) -> dict:
        return {
            "difficulty": round(self.difficulty, 3),
            "success_rate": round(self.success_rate, 3),
            "target_rooms": self.target_rooms(),
            "mean_rooms": round(self.mean_rooms, 2),
            "samples": len(self._outcomes),
        }
