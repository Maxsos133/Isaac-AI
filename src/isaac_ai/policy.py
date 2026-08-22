"""Actor-critic over an entity set.

The observation is a variable-length set of entities plus a scalar vector. A
flat fixed-length encoding would tie the policy to a particular enemy count and
ordering; a permutation-invariant encoder over a padded, masked set lets the
same weights handle one fly or six chargers, and lets the curriculum move the
distribution without invalidating what was learned.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from isaac_ai.env import (ACTION_DIMS, DOOR_FEATURES, EGO_FEATURES,
                          ENTITY_FEATURES, GRID_FEATURES, MAX_DOORS,
                          SCALAR_FEATURES)


def _init(layer: nn.Linear, gain: float = np.sqrt(2)) -> nn.Linear:
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, 0.0)
    return layer


class EntitySetEncoder(nn.Module):
    """Embeds each entity independently, then pools over the set.

    Mean and max pooling are concatenated: mean carries "what is around on
    average", max carries "the single most extreme thing present", which is
    what matters for the nearest threat.
    """

    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.embed = nn.Sequential(
            _init(nn.Linear(ENTITY_FEATURES, hidden)),
            nn.ReLU(),
            _init(nn.Linear(hidden, hidden)),
            nn.ReLU(),
        )

    def forward(self, entities: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # entities: (B, N, F), mask: (B, N)
        embedded = self.embed(entities)
        mask = mask.unsqueeze(-1)
        embedded = embedded * mask

        count = mask.sum(dim=1).clamp(min=1.0)
        mean = embedded.sum(dim=1) / count
        # Masked slots must not win the max.
        masked = embedded.masked_fill(mask == 0, float("-inf"))
        maximum = masked.max(dim=1).values
        maximum = torch.nan_to_num(maximum, neginf=0.0)

        return torch.cat([mean, maximum], dim=-1)


class ActorCritic(nn.Module):
    def __init__(self, hidden: int = 128) -> None:
        super().__init__()
        self.entity_encoder = EntitySetEncoder(hidden)
        self.scalar_encoder = nn.Sequential(
            _init(nn.Linear(SCALAR_FEATURES, hidden)),
            nn.ReLU(),
        )
        # Doors keep their slot ordering, so they go through a plain MLP rather
        # than the permutation-invariant path used for entities.
        self.door_encoder = nn.Sequential(
            _init(nn.Linear(MAX_DOORS * DOOR_FEATURES, hidden)),
            nn.ReLU(),
        )
        # The room's obstacle layout. Rocks stop movement and eat tears, spikes
        # damage on contact, and none of it was in the observation at all — a
        # feedforward policy re-decides every tick, so it could not remember
        # bumping into something and would push at an unseen rock forever.
        # Flat MLP, like doors: a tile's position is what it means.
        self.grid_encoder = nn.Sequential(
            _init(nn.Linear(GRID_FEATURES, hidden)),
            nn.ReLU(),
        )
        # The same obstacles centred on the player. The room-absolute grid above
        # puts "solid one tile to my right" at a different index every time the
        # player moves, so nothing could tie it to the player-relative entity
        # offsets — measured on floor-v10 as a 0.019 total-variation response to
        # moving a rock onto the line of fire. Here that fact is always at the
        # same index. The room grid is kept as well: it carries where the player
        # is *within* the room, which an egocentric window by construction
        # cannot.
        self.ego_encoder = nn.Sequential(
            _init(nn.Linear(EGO_FEATURES, hidden)),
            nn.ReLU(),
        )
        # Six branches now. This must stay the LAST concatenated block: a
        # widened layer transfers by keeping the old columns and zeroing the new
        # ones, so appending leaves a resumed policy computing exactly what it
        # did, while inserting anywhere earlier reshuffles every later column.
        self.trunk = nn.Sequential(
            _init(nn.Linear(hidden * 6, hidden)),
            nn.ReLU(),
            _init(nn.Linear(hidden, hidden)),
            nn.ReLU(),
        )
        # One head per action axis: move x, move y, shoot x, shoot y.
        self.heads = nn.ModuleList([
            _init(nn.Linear(hidden, dim), gain=0.01) for dim in ACTION_DIMS
        ])
        self.value = _init(nn.Linear(hidden, 1), gain=1.0)

    def features(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        entity_features = self.entity_encoder(obs["entities"], obs["entity_mask"])
        scalar_features = self.scalar_encoder(obs["scalars"])
        door_features = self.door_encoder(obs["doors"])
        grid_features = self.grid_encoder(obs["grid"])
        ego_features = self.ego_encoder(obs["ego_grid"])
        return self.trunk(torch.cat(
            [entity_features, scalar_features, door_features, grid_features,
             ego_features],
            dim=-1))

    def forward(self, obs: dict[str, torch.Tensor]):
        hidden = self.features(obs)
        logits = [head(hidden) for head in self.heads]
        return logits, self.value(hidden).squeeze(-1)

    def act(self, obs: dict[str, torch.Tensor]):
        logits, value = self(obs)
        distributions = [Categorical(logits=head) for head in logits]
        actions = torch.stack([d.sample() for d in distributions], dim=-1)
        log_prob = sum(d.log_prob(actions[:, i])
                       for i, d in enumerate(distributions))
        return actions, log_prob, value

    def evaluate(self, obs: dict[str, torch.Tensor], actions: torch.Tensor):
        """Returns per-head probabilities as well as entropies.

        Entropy alone cannot see a single dead *action*. floor-v13 answered an
        up-door state with move_y = [0.01 up, 0.61 still, 0.37 down] — entropy
        0.715, which reads perfectly healthy against a 0.5 target and so earned
        no bonus at all, while the agent could not leave a room whose only exit
        was upward. The probabilities are needed to put a floor under each
        action individually.
        """
        logits, value = self(obs)
        distributions = [Categorical(logits=head) for head in logits]
        log_prob = sum(d.log_prob(actions[:, i])
                       for i, d in enumerate(distributions))
        per_head = [d.entropy() for d in distributions]
        entropy = sum(per_head)
        return log_prob, entropy, value, per_head, [d.probs for d in distributions]


def to_tensors(observation: dict[str, np.ndarray],
               device: torch.device) -> dict[str, torch.Tensor]:
    return {key: torch.as_tensor(value, dtype=torch.float32, device=device)
            for key, value in observation.items()}
