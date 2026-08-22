"""The pixel student, and the privileged critic that trains it.

The end goal is an agent that plays from the screen alone, so the actor sees
only stacked frames. Nothing stops the *critic* from seeing more: it exists
solely to score states during training and is thrown away at deployment, so
feeding it the mod's privileged state costs nothing at inference and removes
the need to infer value from pixels — which is the harder half of the problem
and the half that does not have to be solved.

That asymmetry is the whole design. The actor is deliberately weaker than the
critic, and the gap is what makes learning from pixels tractable.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F

from isaac_ai.env import ACTION_DIMS
from isaac_ai.policy import EntitySetEncoder, _init


class PixelEncoder(nn.Module):
    """Convolutional trunk over the stacked frames.

    The strides are the long-standing Atari arrangement, which is tuned for
    roughly this input size and this much spatial detail. The flattened width
    is measured from a dummy forward rather than derived by hand, so changing
    the student's input size in config.toml cannot silently produce a network
    whose first linear layer is the wrong shape.
    """

    def __init__(self, shape: tuple[int, int, int], hidden: int = 512) -> None:
        super().__init__()
        channels, height, width = shape
        self.conv = nn.Sequential(
            _init(nn.Conv2d(channels, 32, kernel_size=8, stride=4)),
            nn.ReLU(),
            _init(nn.Conv2d(32, 64, kernel_size=4, stride=2)),
            nn.ReLU(),
            _init(nn.Conv2d(64, 64, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, channels, height, width)).shape[1]
        self.head = nn.Sequential(_init(nn.Linear(flat, hidden)), nn.ReLU())
        self.output_size = hidden

    def forward(self, frames: torch.Tensor) -> torch.Tensor:
        # Frames arrive as uint8 to keep the capture path and any replay cheap.
        # Scaling here rather than at the call site means no caller can forget
        # and feed the network values in 0-255.
        return self.head(self.conv(frames.float() / 255.0))


class PrivilegedCritic(nn.Module):
    """Value from the teacher's view of the world.

    Deliberately the same encoders the state-based teacher uses: the point is
    that this half of the network already knows how to read a room, so the
    student only has to learn to see.
    """

    def __init__(self, scalar_features: int, door_features: int,
                 hidden: int = 128, grid_features: int | None = None,
                 ego_features: int | None = None) -> None:
        super().__init__()
        from isaac_ai.env import EGO_FEATURES, GRID_FEATURES
        grid_features = GRID_FEATURES if grid_features is None else grid_features
        ego_features = EGO_FEATURES if ego_features is None else ego_features
        self.entity_encoder = EntitySetEncoder(hidden)
        self.scalar_encoder = nn.Sequential(
            _init(nn.Linear(scalar_features, hidden)), nn.ReLU())
        self.door_encoder = nn.Sequential(
            _init(nn.Linear(door_features, hidden)), nn.ReLU())
        self.grid_encoder = nn.Sequential(
            _init(nn.Linear(grid_features, hidden)), nn.ReLU())
        # Mirrors the teacher branch for branch, which is the invariant the
        # seeding test pins: the critic is the teacher's network minus its
        # action heads, so every encoder the teacher grows has to grow here too
        # or the transfer silently starts leaving pieces behind.
        self.ego_encoder = nn.Sequential(
            _init(nn.Linear(ego_features, hidden)), nn.ReLU())
        self.trunk = nn.Sequential(
            _init(nn.Linear(hidden * 6, hidden)), nn.ReLU(),
            _init(nn.Linear(hidden, hidden)), nn.ReLU(),
        )
        self.value = _init(nn.Linear(hidden, 1), gain=1.0)

    def forward(self, state: dict[str, torch.Tensor]) -> torch.Tensor:
        features = torch.cat([
            self.entity_encoder(state["entities"], state["entity_mask"]),
            self.scalar_encoder(state["scalars"]),
            self.door_encoder(state["doors"]),
            self.grid_encoder(state["grid"]),
            self.ego_encoder(state["ego_grid"]),
        ], dim=-1)
        return self.value(self.trunk(features)).squeeze(-1)


class PixelActorCritic(nn.Module):
    """Actor over pixels, critic over privileged state.

    `act` takes both because training needs both; `act_from_pixels` takes only
    frames and is what deployment runs. Keeping the second one in the class,
    rather than reconstructing it later, means the pixels-only path is exercised
    by the tests from the start instead of being written for the first time on
    the day the scaffolding is removed.
    """

    def __init__(self, pixel_shape: tuple[int, int, int],
                 scalar_features: int, door_features: int,
                 hidden: int = 512) -> None:
        super().__init__()
        self.encoder = PixelEncoder(pixel_shape, hidden)
        self.heads = nn.ModuleList([
            _init(nn.Linear(self.encoder.output_size, dim), gain=0.01)
            for dim in ACTION_DIMS
        ])
        self.critic = PrivilegedCritic(scalar_features, door_features)

    def logits(self, frames: torch.Tensor) -> list[torch.Tensor]:
        features = self.encoder(frames)
        return [head(features) for head in self.heads]

    def act_from_pixels(self, frames: torch.Tensor,
                        deterministic: bool = False) -> torch.Tensor:
        """The deployment path: frames in, actions out, nothing privileged."""
        logits = self.logits(frames)
        if deterministic:
            return torch.stack([head.argmax(dim=-1) for head in logits], dim=-1)
        return torch.stack([Categorical(logits=head).sample()
                            for head in logits], dim=-1)

    def act(self, frames: torch.Tensor, state: dict[str, torch.Tensor]):
        logits = self.logits(frames)
        distributions = [Categorical(logits=head) for head in logits]
        actions = torch.stack([d.sample() for d in distributions], dim=-1)
        log_prob = sum(d.log_prob(actions[:, i])
                       for i, d in enumerate(distributions))
        return actions, log_prob, self.critic(state)

    def evaluate(self, frames: torch.Tensor, state: dict[str, torch.Tensor],
                 actions: torch.Tensor):
        logits = self.logits(frames)
        distributions = [Categorical(logits=head) for head in logits]
        log_prob = sum(d.log_prob(actions[:, i])
                       for i, d in enumerate(distributions))
        per_head = [d.entropy() for d in distributions]
        return log_prob, sum(per_head), self.critic(state), per_head


def distillation_loss(student_logits: list[torch.Tensor],
                      teacher_logits: list[torch.Tensor]) -> torch.Tensor:
    """KL(teacher || student), summed over the four action heads.

    Matching the teacher's whole distribution rather than its argmax is what
    carries the information that matters here: combat-v3 hedges between two
    directions when either would do, and a student trained on hard labels has
    to guess which arbitrary tie-break to imitate instead of learning that the
    choice was open.
    """
    total = student_logits[0].new_zeros(())
    for student, teacher in zip(student_logits, teacher_logits):
        total = total + F.kl_div(
            F.log_softmax(student, dim=-1),
            F.log_softmax(teacher.detach(), dim=-1),
            reduction="batchmean",
            log_target=True,
        )
    return total


def frames_to_tensor(frames: np.ndarray, device: torch.device) -> torch.Tensor:
    """Stacked uint8 frames straight from capture into a batch on the device."""
    return torch.as_tensor(frames, dtype=torch.uint8, device=device)
