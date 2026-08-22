"""Difficulty ramp for the navigation task.

The dial is how many rooms must be crossed in a single episode. Unlike the floor
curriculum — which after the termination fix only moved a scoring threshold and
had no effect on training — this one genuinely changes the task: requiring three
traversals instead of one is a longer walk with more decisions in it.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class NavigationCurriculum:
    difficulty: float = 0.0
    target_success: float = 0.6
    adjust_rate: float = 0.03
    deadband: float = 0.12
    window: int = 60

    min_transitions: int = 1
    max_transitions: int = 5

    _outcomes: list[bool] = field(default_factory=list)
    _transitions: list[int] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        if not self._outcomes:
            return 0.0
        return sum(self._outcomes) / len(self._outcomes)

    @property
    def mean_transitions(self) -> float:
        if not self._transitions:
            return 0.0
        return sum(self._transitions) / len(self._transitions)

    def required_transitions(self) -> int:
        span = self.max_transitions - self.min_transitions
        return max(self.min_transitions,
                   min(self.max_transitions,
                       self.min_transitions + int(round(span * self.difficulty))))

    def record(self, success: bool, transitions: int) -> None:
        self._outcomes.append(success)
        self._transitions.append(transitions)
        if len(self._outcomes) > self.window:
            self._outcomes.pop(0)
            self._transitions.pop(0)

    def advance(self) -> None:
        """Once per policy update, never per episode."""
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
            "required": self.required_transitions(),
            "mean_transitions": round(self.mean_transitions, 2),
            "samples": len(self._outcomes),
        }
