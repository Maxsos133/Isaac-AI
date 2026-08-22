"""Offline checks for the policy, curriculum and PPO plumbing.

These run without the game so mistakes in shapes, masking or advantage
computation surface in seconds rather than after a fleet bring-up.
"""

from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaac_ai.curriculum import ROSTER, CombatCurriculum  # noqa: E402
from isaac_ai.env import (ACTION_DIMS, DOOR_CATEGORY_INDEX,  # noqa: E402
                          DOOR_FEATURES, ENTITY_CHEST, ENTITY_CLOSING,
                          ENTITY_CONSUMABLE, ENTITY_FEATURES, ENTITY_FLYING,
                          ENTITY_HOSTILE, ENTITY_PEDESTAL, ENTITY_TANGENTIAL,
                          EGO_FEATURES, GRID_FEATURES, MAX_DOORS, MAX_ENTITIES,
                          SCALAR_FEATURES, compute_reward, decode_action,
                          encode_observation)
from isaac_ai.floor_curriculum import FloorCurriculum  # noqa: E402
from isaac_ai.policy import ActorCritic, to_tensors  # noqa: E402


def fake_observation(batch: int = 5) -> dict[str, torch.Tensor]:
    entities = torch.randn(batch, MAX_ENTITIES, ENTITY_FEATURES)
    mask = torch.zeros(batch, MAX_ENTITIES)
    # Varying entity counts, including an entirely empty set.
    for row in range(batch):
        mask[row, :row] = 1.0
    return {
        "scalars": torch.randn(batch, SCALAR_FEATURES),
        "entities": entities,
        "entity_mask": mask,
        "doors": torch.randn(batch, MAX_DOORS * DOOR_FEATURES),
        "grid": torch.randn(batch, GRID_FEATURES),
        "ego_grid": torch.randn(batch, EGO_FEATURES),
    }


class TestPolicy(unittest.TestCase):
    def test_shapes(self) -> None:
        policy = ActorCritic()
        obs = fake_observation()
        actions, log_prob, value = policy.act(obs)
        self.assertEqual(actions.shape, (5, len(ACTION_DIMS)))
        self.assertEqual(log_prob.shape, (5,))
        self.assertEqual(value.shape, (5,))

    def test_entropy_splits_by_head_group(self) -> None:
        """Summed entropy hides that the shoot heads cannot learn during
        navigation, so they sit at a fixed 2*ln(3) and make movement look far
        less converged than it is."""
        import math
        from isaac_ai.env import MOVE_HEADS, SHOOT_HEADS
        policy = ActorCritic()
        obs = fake_observation()
        actions, _, _ = policy.act(obs)
        _, entropy, _, per_head, _ = policy.evaluate(obs, actions)

        self.assertEqual(len(per_head), len(ACTION_DIMS))
        total = sum(h.mean().item() for h in per_head)
        self.assertAlmostEqual(entropy.mean().item(), total, places=4)

        # A freshly initialised policy is near uniform on every head.
        uniform = math.log(3)
        move = sum(per_head[i].mean().item() for i in MOVE_HEADS)
        shoot = sum(per_head[i].mean().item() for i in SHOOT_HEADS)
        self.assertAlmostEqual(move, 2 * uniform, places=2)
        self.assertAlmostEqual(shoot, 2 * uniform, places=2)

    def test_empty_entity_set_is_finite(self) -> None:
        """An empty room must not produce -inf from the masked max-pool."""
        policy = ActorCritic()
        obs = {
            "scalars": torch.randn(2, SCALAR_FEATURES),
            "entities": torch.randn(2, MAX_ENTITIES, ENTITY_FEATURES),
            "entity_mask": torch.zeros(2, MAX_ENTITIES),
            "doors": torch.randn(2, MAX_DOORS * DOOR_FEATURES),
            "grid": torch.zeros(2, GRID_FEATURES),
            "ego_grid": torch.zeros(2, EGO_FEATURES),
        }
        _, log_prob, value = policy.act(obs)
        self.assertTrue(torch.isfinite(log_prob).all())
        self.assertTrue(torch.isfinite(value).all())

    def test_permutation_invariance(self) -> None:
        """Shuffling entity order must not change the policy output."""
        policy = ActorCritic()
        obs = fake_observation(batch=1)
        obs["entity_mask"][0, :6] = 1.0

        _, value_a = policy(obs)
        order = torch.randperm(MAX_ENTITIES)
        shuffled = {
            "scalars": obs["scalars"],
            "entities": obs["entities"][:, order],
            "entity_mask": obs["entity_mask"][:, order],
            "doors": obs["doors"],
            "grid": obs["grid"],
            "ego_grid": obs["ego_grid"],
        }
        _, value_b = policy(shuffled)
        self.assertTrue(torch.allclose(value_a, value_b, atol=1e-5))

    def test_padding_does_not_leak(self) -> None:
        """Changing masked-out slots must not change the output."""
        policy = ActorCritic()
        obs = fake_observation(batch=1)
        obs["entity_mask"][0, :4] = 1.0
        _, value_a = policy(obs)

        polluted = {k: v.clone() for k, v in obs.items()}
        polluted["entities"][0, 4:] = 999.0
        _, value_b = policy(polluted)
        self.assertTrue(torch.allclose(value_a, value_b, atol=1e-5))


class TestObservationEncoding(unittest.TestCase):
    def raw(self) -> dict:
        return {
            "ready": True,
            "player": {"x": 320, "y": 280, "vx": 0, "vy": 0, "hearts": 6,
                       "max_hearts": 6, "soul_hearts": 0, "bombs": 1, "keys": 0,
                       "coins": 0, "damage": 3.5, "speed": 1.0, "tear_delay": 10,
                       "range": 260, "can_fly": False, "active_item": 0,
                       "is_dead": False},
            "room": {"index": 84, "type": 1, "shape": 1, "clear": False,
                     "top_left_x": 60, "top_left_y": 140,
                     "bottom_right_x": 580, "bottom_right_y": 420,
                     "enemies_alive": 2,
                     "doors": [
                         {"slot": 0, "x": 60, "y": 280, "open": True,
                          "locked": False, "target": 83, "visited": 3,
                          "category": "normal"},
                         {"slot": 2, "x": 580, "y": 280, "open": True,
                          "locked": False, "target": 85, "visited": 0,
                          "category": "curse"},
                     ]},
            "level": {"stage": 1, "stage_type": 0, "curses": 0,
                      "rooms_total": 12, "rooms_visited": 4},
            "entities": [
                {"k": "enemy", "t": 10, "v": 0, "s": 0, "x": 400, "y": 300,
                 "vx": 1, "vy": 0, "hp": 5, "mhp": 10, "boss": False, "d": 82},
            ],
            "events": {},
        }

    def moving_enemy(self, vx: float, vy: float,
                     player_vx: float = 0.0) -> np.ndarray:
        """One enemy to the player's right, with the given velocities."""
        raw = self.raw()
        raw["player"]["vx"] = player_vx
        raw["entities"] = [{"k": "enemy", "t": 10, "v": 0, "s": 0,
                            "x": 400, "y": 280, "vx": vx, "vy": vy,
                            "hp": 5, "mhp": 10, "boss": False, "d": 82}]
        return encode_observation(raw)["entities"][0]

    def grid_room(self, cells: list[int], width: int = 15) -> np.ndarray:
        from isaac_ai.env import encode_grid
        return encode_grid({"grid": cells, "grid_width": width})

    def test_obstacles_reach_the_agent_at_the_right_place(self) -> None:
        """Rocks and spikes were absent from the observation entirely.

        A feedforward policy re-decides every tick, so it cannot remember
        walking into something — an unseen rock is permanently unlearnable and
        it will push at one forever while its tears are eaten by nothing it can
        see. Position carries the meaning here, so the layout must survive.
        """
        from isaac_ai.env import GRID_CLASSES, GRID_HEIGHT, GRID_WIDTH

        cells = [0] * (GRID_WIDTH * GRID_HEIGHT)
        cells[GRID_WIDTH * 2 + 3] = 1   # solid at row 2, column 3
        cells[GRID_WIDTH * 5 + 9] = 2   # hazard at row 5, column 9
        grid = self.grid_room(cells).reshape(GRID_CLASSES, GRID_HEIGHT, GRID_WIDTH)

        self.assertEqual(float(grid[0, 2, 3]), 1.0)
        self.assertEqual(float(grid[1, 5, 9]), 1.0)
        # Each class in its own channel, and nowhere else.
        self.assertEqual(float(grid[1, 2, 3]), 0.0)
        self.assertEqual(float(grid.sum()), 2.0)

    def test_free_floor_and_missing_data_both_read_as_empty(self) -> None:
        """An unreadable room must look like open floor, not like walls."""
        from isaac_ai.env import GRID_FEATURES, GRID_HEIGHT, GRID_WIDTH

        empty = self.grid_room([0] * (GRID_WIDTH * GRID_HEIGHT))
        self.assertEqual(float(empty.sum()), 0.0)
        self.assertEqual(empty.shape, (GRID_FEATURES,))

        from isaac_ai.env import encode_grid
        self.assertEqual(float(encode_grid({}).sum()), 0.0)

    def test_a_larger_room_is_sampled_not_truncated(self) -> None:
        """Bigger rooms must keep their layout rather than lose half of it."""
        from isaac_ai.env import GRID_CLASSES, GRID_HEIGHT, GRID_WIDTH

        wide, tall = GRID_WIDTH * 2, GRID_HEIGHT * 2
        cells = [0] * (wide * tall)
        # A solid tile in the far bottom-right, which truncation would drop.
        cells[(tall - 1) * wide + (wide - 1)] = 1
        grid = self.grid_room(cells, width=wide).reshape(
            GRID_CLASSES, GRID_HEIGHT, GRID_WIDTH)
        self.assertEqual(float(grid[0, GRID_HEIGHT - 1, GRID_WIDTH - 1]), 1.0)

    def test_closing_speed_separates_a_charger_from_scenery(self) -> None:
        """The distinction the agent had no way to see.

        Enemy velocity was room-absolute and the player's own velocity lives in
        the scalar vector, which is concatenated after the entity pooling — so
        "is this thing coming for me" was not merely hard to infer per entity,
        it was absent. Watching the agent play showed it taking contact damage
        from chargers and never singling them out.
        """
        closing = ENTITY_CLOSING

        # Enemy is to the player's right, so moving left closes the gap.
        self.assertGreater(self.moving_enemy(-5, 0)[closing], 0.0)
        self.assertLess(self.moving_enemy(5, 0)[closing], 0.0)
        # Moving perpendicular to the line between them closes nothing.
        self.assertAlmostEqual(float(self.moving_enemy(0, 5)[closing]), 0.0,
                               places=5)

    def test_closing_speed_accounts_for_the_player_moving(self) -> None:
        """A stationary enemy still closes if the player charges it."""
        closing = ENTITY_CLOSING
        self.assertGreater(self.moving_enemy(0, 0, player_vx=5)[closing], 0.0)

    def test_tangential_speed_sees_an_orbiting_enemy(self) -> None:
        tangential = ENTITY_TANGENTIAL
        # Straight at the player: no sideways component.
        self.assertAlmostEqual(float(self.moving_enemy(-5, 0)[tangential]), 0.0,
                               places=5)
        # Across the line of sight: all sideways.
        self.assertNotAlmostEqual(float(self.moving_enemy(0, 5)[tangential]),
                                  0.0, places=5)

    def test_encoding_shapes_and_range(self) -> None:
        encoded = encode_observation(self.raw())
        self.assertEqual(encoded["scalars"].shape, (SCALAR_FEATURES,))
        self.assertEqual(encoded["entities"].shape, (MAX_ENTITIES, ENTITY_FEATURES))
        self.assertEqual(encoded["entity_mask"].sum(), 1.0)
        self.assertTrue(np.isfinite(encoded["scalars"]).all())

    def test_player_centre_maps_near_zero(self) -> None:
        raw = self.raw()
        raw["player"]["x"] = (60 + 580) / 2
        raw["player"]["y"] = (140 + 420) / 2
        encoded = encode_observation(raw)
        self.assertAlmostEqual(float(encoded["scalars"][0]), 0.0, places=5)
        self.assertAlmostEqual(float(encoded["scalars"][1]), 0.0, places=5)

    def test_entity_carries_direction_to_player(self) -> None:
        """The per-entity block must encode where the target is *relative to
        the player*. The encoder pools entities before player state is
        concatenated, so anything omitted here is unrecoverable for aiming."""
        raw = self.raw()
        raw["player"]["x"], raw["player"]["y"] = 300.0, 300.0
        # Enemy directly to the right.
        raw["entities"][0]["x"], raw["entities"][0]["y"] = 400.0, 300.0
        right = encode_observation(raw)["entities"][0]
        self.assertAlmostEqual(float(right[13]), 1.0, places=5)   # unit dx
        self.assertAlmostEqual(float(right[14]), 0.0, places=5)   # unit dy

        # Same enemy, now directly above.
        raw["entities"][0]["x"], raw["entities"][0]["y"] = 300.0, 200.0
        up = encode_observation(raw)["entities"][0]
        self.assertAlmostEqual(float(up[13]), 0.0, places=5)
        self.assertAlmostEqual(float(up[14]), -1.0, places=5)

    def test_relative_features_move_with_the_player(self) -> None:
        """Identical geometry at a different room position must encode the
        same direction — otherwise aim would have to be relearned per location."""
        raw = self.raw()
        raw["player"]["x"], raw["player"]["y"] = 200.0, 200.0
        raw["entities"][0]["x"], raw["entities"][0]["y"] = 260.0, 200.0
        a = encode_observation(raw)["entities"][0]

        raw["player"]["x"], raw["player"]["y"] = 450.0, 380.0
        raw["entities"][0]["x"], raw["entities"][0]["y"] = 510.0, 380.0
        b = encode_observation(raw)["entities"][0]

        for feature in (10, 11, 12, 13, 14):
            self.assertAlmostEqual(float(a[feature]), float(b[feature]), places=5)

    def test_doors_land_in_their_own_slots(self) -> None:
        """Slot index is meaning: slot 0 is always the left door. The block is
        positional, so a door must not be written anywhere else."""
        doors = encode_observation(self.raw())["doors"].reshape(MAX_DOORS, DOOR_FEATURES)
        self.assertEqual(float(doors[0][0]), 1.0)   # left door present
        self.assertEqual(float(doors[2][0]), 1.0)   # right door present
        self.assertEqual(float(doors[1][0]), 0.0)   # no door up
        self.assertEqual(float(doors[3][0]), 0.0)   # no door down

    def test_unvisited_door_is_flagged(self) -> None:
        """The unexplored marker is what separates exploring from pacing."""
        doors = encode_observation(self.raw())["doors"].reshape(MAX_DOORS, DOOR_FEATURES)
        self.assertEqual(float(doors[0][3]), 0.0)   # target visited 3 times
        self.assertEqual(float(doors[2][3]), 1.0)   # target never visited

    def test_door_category_flags(self) -> None:
        """A curse door costs half a heart to walk through and is otherwise
        identical to a normal door. Without this flag that damage is
        unpredictable from anything the agent can see."""
        doors = encode_observation(self.raw())["doors"].reshape(MAX_DOORS, DOOR_FEATURES)
        normal = 7 + DOOR_CATEGORY_INDEX["normal"]
        curse = 7 + DOOR_CATEGORY_INDEX["curse"]
        self.assertEqual(float(doors[0][normal]), 1.0)
        self.assertEqual(float(doors[0][curse]), 0.0)
        self.assertEqual(float(doors[2][curse]), 1.0)
        self.assertEqual(float(doors[2][normal]), 0.0)

    def test_unknown_category_sets_no_flag(self) -> None:
        """An unmapped category must not masquerade as a normal door — a broken
        mapping should be visible, not silently absorbed."""
        raw = self.raw()
        raw["room"]["doors"][0]["category"] = "unknown"
        doors = encode_observation(raw)["doors"].reshape(MAX_DOORS, DOOR_FEATURES)
        self.assertEqual(float(doors[0][7:].sum()), 0.0)
        self.assertEqual(float(doors[0][0]), 1.0)  # still reported as present

    def test_room_transition_pays_nothing_in_the_shared_reward(self) -> None:
        """The transition event fires on every crossing, including back into a
        cleared room. Paying it here let the agent shuttle between two rooms for
        +1.0 a lap, forever. Exploration is scored by the floor env instead."""
        from isaac_ai.config import load_config
        config = load_config()
        raw = self.raw()
        base = {"damage_taken": 0.0, "damage_dealt": 0.0, "kills": 0,
                "room_cleared": False, "new_level": False, "died": False}

        raw["events"] = {**base, "new_room": False}
        without, _ = compute_reward(raw, config)
        raw["events"] = {**base, "new_room": True}
        with_transition, _ = compute_reward(raw, config)
        self.assertAlmostEqual(with_transition, without, places=6)

    def test_decode_action_covers_axes(self) -> None:
        self.assertEqual(decode_action([0, 1, 2, 1]), (-1, 0, 1, 0))
        self.assertEqual(decode_action([2, 2, 0, 0]), (1, 1, -1, -1))


class TestCurriculum(unittest.TestCase):
    def test_starts_easy(self) -> None:
        curriculum = CombatCurriculum(seed=1)
        encounter = curriculum.sample()
        self.assertEqual(encounter.enemy_count, curriculum.min_enemies)
        self.assertLessEqual(len(curriculum.available()), 3)

    def test_recording_alone_never_moves_difficulty(self) -> None:
        """Episodes are frequent; only `advance` may move the ramp."""
        curriculum = CombatCurriculum(seed=1, window=10, adjust_rate=0.1)
        for _ in range(200):
            curriculum.record(True)
        self.assertEqual(curriculum.difficulty, 0.0)

    def test_difficulty_rises_only_on_sustained_success(self) -> None:
        curriculum = CombatCurriculum(seed=1, window=10, adjust_rate=0.1)
        for _ in range(9):
            curriculum.record(True)
        curriculum.advance()
        self.assertEqual(curriculum.difficulty, 0.0)  # window not full yet
        curriculum.record(True)
        curriculum.advance()
        self.assertGreater(curriculum.difficulty, 0.0)

    def test_one_advance_moves_by_one_rate_step(self) -> None:
        curriculum = CombatCurriculum(seed=1, window=10, adjust_rate=0.1)
        for _ in range(10):
            curriculum.record(True)
        curriculum.advance()
        self.assertAlmostEqual(curriculum.difficulty, 0.1, places=6)

    def test_deadband_holds_difficulty_at_target(self) -> None:
        curriculum = CombatCurriculum(seed=1, window=10, adjust_rate=0.1,
                                      target_success=0.6, deadband=0.12)
        # 60% success is exactly on target: the ramp should not move.
        for index in range(10):
            curriculum.record(index < 6)
        curriculum.advance()
        self.assertEqual(curriculum.difficulty, 0.0)

    def test_difficulty_falls_when_losing(self) -> None:
        curriculum = CombatCurriculum(seed=1, window=10, adjust_rate=0.1)
        for _ in range(10):
            curriculum.record(True)
        for _ in range(5):
            curriculum.advance()
        peak = curriculum.difficulty
        self.assertGreater(peak, 0.0)
        for _ in range(10):
            curriculum.record(False)
        curriculum.advance()
        self.assertLess(curriculum.difficulty, peak)

    def test_harder_settings_open_the_roster(self) -> None:
        curriculum = CombatCurriculum(seed=1)
        curriculum.difficulty = 1.0
        self.assertEqual(len(curriculum.available()), len(ROSTER))
        self.assertEqual(curriculum.sample().enemy_count, curriculum.max_enemies)


class TestFloorEpisodeShape(unittest.TestCase):
    def test_target_does_not_cap_rooms_cleared(self) -> None:
        """Reaching the curriculum bar must not end the episode.

        If it did, rooms_cleared could never exceed target_rooms, and an agent
        that stops at the bar would be indistinguishable from one that would
        have kept going — which is exactly the measurement we need.
        """
        source = (Path(__file__).resolve().parents[1]
                  / "src" / "isaac_ai" / "floors.py").read_text(encoding="utf-8")
        # The success flag may be computed, but must not drive termination.
        self.assertNotIn("if reached:\n                terminated", source)
        self.assertIn("reached = self._rooms_cleared[index] >= target", source)


class TestFloorCurriculum(unittest.TestCase):
    def test_bar_starts_at_one_room(self) -> None:
        self.assertEqual(FloorCurriculum().target_rooms(), 1)

    def test_bar_rises_with_difficulty(self) -> None:
        curriculum = FloorCurriculum()
        curriculum.difficulty = 1.0
        self.assertEqual(curriculum.target_rooms(), curriculum.max_rooms)

    def test_recording_alone_never_moves_the_bar(self) -> None:
        curriculum = FloorCurriculum(window=10, adjust_rate=0.1)
        for _ in range(200):
            curriculum.record(True, 3)
        self.assertEqual(curriculum.difficulty, 0.0)

    def test_advance_moves_once_per_call(self) -> None:
        curriculum = FloorCurriculum(window=10, adjust_rate=0.1)
        for _ in range(10):
            curriculum.record(True, 5)
        curriculum.advance()
        self.assertAlmostEqual(curriculum.difficulty, 0.1, places=6)
        self.assertAlmostEqual(curriculum.mean_rooms, 5.0, places=6)


class TestDoorPotential(unittest.TestCase):
    """Potential-based shaping must pull toward unexplored doors only."""

    def raw(self, player_x: float, doors: list[dict]) -> dict:
        return {
            "ready": True,
            "player": {"x": player_x, "y": 280.0},
            "room": {"top_left_x": 60, "top_left_y": 140,
                     "bottom_right_x": 580, "bottom_right_y": 420,
                     "doors": doors},
        }

    def test_closer_to_an_unvisited_door_is_higher_potential(self) -> None:
        from isaac_ai.floors import door_potential
        door = [{"x": 580.0, "y": 280.0, "visited": 0, "locked": False, "slot": 2}]
        far = door_potential(self.raw(100.0, door))
        near = door_potential(self.raw(500.0, door))
        self.assertGreater(near, far)
        self.assertGreaterEqual(far, 0.0)
        self.assertLessEqual(near, 1.0)

    def test_a_secret_door_is_not_encoded_as_a_door_at_all(self) -> None:
        """Removing it from the shaping was not enough.

        floor-v12 stopped steering at secret rooms and the agent kept walking
        into them, because the encoded door still read `present=1, unvisited=1`
        — byte-identical to a sacrifice-room door, which is passable — and the
        policy carried a million steps of "unvisited doors are worth walking
        to". Killing the reward removes the reinforcement but leaves both the
        habit and the perception that justifies it.
        """
        from isaac_ai.env import DOOR_FEATURES, MAX_DOORS, encode_observation

        def encoded(needs_bomb: bool):
            obs = {
                "ready": True,
                "player": {"x": 320.0, "y": 280.0, "vx": 0, "vy": 0,
                           "hearts": 6, "max_hearts": 6, "soul_hearts": 0,
                           "bombs": 0, "keys": 0, "coins": 0, "damage": 3.5,
                           "speed": 1.0, "tear_delay": 10, "range": 260,
                           "can_fly": False, "active_item": 0, "is_dead": False},
                "room": {"index": 84, "type": 1, "shape": 1, "clear": True,
                         "top_left_x": 60, "top_left_y": 140,
                         "bottom_right_x": 580, "bottom_right_y": 420,
                         "enemies_alive": 0,
                         "doors": [{"slot": 1, "x": 580.0, "y": 280.0,
                                    "open": False, "locked": False,
                                    "visited": 0, "needs_bomb": needs_bomb,
                                    "category": "other"}]},
                "level": {"stage": 1, "stage_type": 0, "curses": 0,
                          "rooms_total": 12, "rooms_visited": 4},
                "entities": [], "events": {},
            }
            return encode_observation(obs)["doors"].reshape(
                MAX_DOORS, DOOR_FEATURES)[1]

        self.assertEqual(float(encoded(True).sum()), 0.0,
                         "a bomb-only door must read as plain wall")
        self.assertEqual(float(encoded(False)[0]), 1.0,
                         "an ordinary door must still be present")

    def test_a_secret_room_door_does_not_attract(self) -> None:
        """`locked` is not the same question as "can I get through this".

        A secret room's door is closed and *unlocked*, and opening it needs a
        bomb the floor agent has no action for. Measured over 900 fleet steps:
        secret doors were seen 2450 times, never once open, never locked, and
        the potential steered at one on 1068 observations — 6% of every step
        where it had a target. The agent walks into the wall, confidently,
        until the idle limit ends the attempt and the floor reseeds.
        """
        from isaac_ai.floors import door_potential
        secret = [{"x": 580.0, "y": 280.0, "visited": 0, "locked": False,
                   "needs_bomb": True, "slot": 2}]
        self.assertEqual(door_potential(self.raw(500.0, secret)), 0.0)

        # And it must not shadow a real exit that is further away.
        both = secret + [{"x": 60.0, "y": 280.0, "visited": 0, "locked": False,
                          "needs_bomb": False, "slot": 0}]
        near_real = door_potential(self.raw(100.0, both))
        self.assertGreater(near_real, 0.5,
                           "the reachable door should be what the agent is "
                           "pulled towards")

    def test_an_ordinary_closed_door_still_attracts(self) -> None:
        """Doors shut for a fight reopen on the clear.

        Filtering on `open` instead would collapse the potential the moment a
        fight starts, which is a discontinuity paid on *entering a room* — the
        exact shape that plateaued floor-v1 through v3 at -2.885 a transition.
        """
        from isaac_ai.floors import door_potential
        shut = [{"x": 580.0, "y": 280.0, "visited": 0, "locked": False,
                 "open": False, "slot": 2}]
        self.assertGreater(door_potential(self.raw(500.0, shut)), 0.0)

    def test_visited_and_locked_doors_do_not_attract(self) -> None:
        from isaac_ai.floors import door_potential
        # A door already explored, and one that cannot be opened without a key.
        doors = [{"x": 580.0, "y": 280.0, "visited": 4, "locked": False, "slot": 2},
                 {"x": 60.0, "y": 280.0, "visited": 0, "locked": True, "slot": 0}]
        self.assertEqual(door_potential(self.raw(500.0, doors)), 0.0)

    def test_navigation_is_attracted_to_visited_doors_too(self) -> None:
        """A dead-end room's only door leads back where it came from. The
        navigation task counts that traversal as success, so excluding it would
        leave the agent with no gradient toward the one exit that ends the
        episode."""
        from isaac_ai.floors import door_potential
        doors = [{"x": 580.0, "y": 280.0, "visited": 4, "locked": False, "slot": 2}]
        self.assertEqual(door_potential(self.raw(500.0, doors)), 0.0)
        self.assertGreater(
            door_potential(self.raw(500.0, doors), unvisited_only=False), 0.0)

    def test_one_step_of_shaping_outweighs_the_step_penalty(self) -> None:
        """The signal has to be worth listening to.

        In nav-v1 a step toward a door was worth ~0.004 — the same order as the
        per-step penalty — so the whole policy gradient came out ten times
        smaller than PPO's entropy bonus and the policy drifted to uniform.
        """
        from isaac_ai.config import load_config
        from isaac_ai.floors import door_potential
        config = load_config()

        door = [{"x": 580.0, "y": 280.0, "visited": 0, "locked": False, "slot": 2}]
        # One tick of movement is roughly four pixels.
        before = door_potential(self.raw(300.0, door), unvisited_only=False)
        after = door_potential(self.raw(304.0, door), unvisited_only=False)
        shaping = config.rewards.door_shaping * (after - before)

        self.assertGreater(shaping, 0.0)
        self.assertGreater(shaping, abs(config.rewards.step) * 5)

    def test_death_is_paid_for_even_when_the_event_never_arrives(self) -> None:
        """The -10 was inert, and the deaths counter read 0 while it happened.

        `compute_reward` pays the death penalty on `events["died"]`, which the
        mod sets from MC_POST_GAME_END — and that fires as the game-over screen
        takes over, which is exactly where mod callbacks stop running, so the
        observation carrying it is usually never sent. `is_dead` is visible
        during the death animation and is what actually arrives.
        """
        from isaac_ai.config import load_config
        from isaac_ai.env import compute_reward
        rewards = load_config().rewards

        dead = {"player": {"is_dead": True}, "events": {
            "damage_dealt": 0.0, "damage_taken": 0.0, "kills": 0,
            "room_cleared": False, "new_level": False, "died": False}}
        paid, _ = compute_reward(dead, load_config())
        # compute_reward alone does not notice: this is the bug, pinned.
        self.assertGreater(paid, rewards.death / 2,
                           "compute_reward pays nothing for a dead player")

        # The environments must therefore pay it themselves on is_dead, exactly
        # once, and must not double-pay when the event does arrive.
        with_event = dict(dead)
        with_event["events"] = {**dead["events"], "died": True}
        paid_event, _ = compute_reward(with_event, load_config())
        self.assertAlmostEqual(paid_event - paid, rewards.death, places=5)

    def test_walking_through_a_door_is_never_punished(self) -> None:
        """The bug that plateaued floor-v1 through v3 at 0.25 rooms an episode.

        With a room-local potential, standing on an unvisited door reads 0.990
        and one step later inside the new room reads 0.019 — that door is now
        visited and the next is across the room. At coef 4 the shaping paid
        -3.885 for the step against a +1.00 new_room bonus, so making progress
        cost 2.9 reward. The floor-wide potential cancels the drop against the
        room just gained.
        """
        from isaac_ai.config import load_config
        from isaac_ai.floors import floor_potential
        rewards = load_config().rewards

        def state(x: float, doors: list, visited: int) -> dict:
            return {"ready": True, "player": {"x": x, "y": 280.0},
                    "level": {"rooms_visited": visited, "rooms_total": 12},
                    "room": {"top_left_x": 60.0, "top_left_y": 140.0,
                             "bottom_right_x": 580.0, "bottom_right_y": 420.0,
                             "doors": doors}}

        at_door = state(575.0, [{"x": 60.0, "y": 280.0, "visited": 3, "locked": False},
                                {"x": 580.0, "y": 280.0, "visited": 0, "locked": False}], 4)
        inside = state(70.0, [{"x": 60.0, "y": 280.0, "visited": 1, "locked": False},
                              {"x": 580.0, "y": 280.0, "visited": 0, "locked": False}], 5)

        shaping = rewards.door_shaping * (floor_potential(inside)
                                          - floor_potential(at_door))
        self.assertGreater(shaping + rewards.new_room, 0.0,
                           "entering a new room must not cost reward")
        # And the jump itself must be small: the room gained cancels the local
        # closeness lost, so the transition is nearly free either way.
        self.assertLess(abs(shaping), rewards.new_room)

    def test_combat_gate_costs_nothing_to_enter_and_pays_to_clear(self):
        """The gate must not reintroduce the discontinuity it sits next to.

        `SUPPRESS_SHAPING_IN_COMBAT` drops the local door term while a room has
        live enemies, because `probe_door_pull.py` measured the shaping still
        pulling on 97.1% of combat steps at a mean potential of 0.728. The whole
        risk of that is paying a penalty for walking into a fight — which is the
        exact shape (-2.885 a transition) that plateaued floor-v1 through v3.

        It cannot, and this pins why: the local term is worth at most 1 and
        `rooms_visited` gains exactly 1 on entry, so the step is
        `+1 - closeness_before >= 0`. Clearing then restores the term, which is a
        bonus rather than a cost.
        """
        from isaac_ai.config import load_config
        from isaac_ai.floors import SUPPRESS_SHAPING_IN_COMBAT, floor_potential
        if not SUPPRESS_SHAPING_IN_COMBAT:
            self.skipTest("combat gate disabled")
        rewards = load_config().rewards

        def state(x: float, visited: int, enemies: int) -> dict:
            return {"ready": True, "player": {"x": x, "y": 280.0},
                    "level": {"rooms_visited": visited, "rooms_total": 12},
                    "room": {"top_left_x": 60.0, "top_left_y": 140.0,
                             "bottom_right_x": 580.0, "bottom_right_y": 420.0,
                             "enemies_alive": enemies,
                             "doors": [{"x": 580.0, "y": 280.0,
                                        "visited": 0, "locked": False}]}}

        # Standing on a door in a cleared room, then one step into a room that
        # turns out to be contested: the worst case for the gate, because the
        # local term was near its maximum and is dropped entirely.
        at_door = state(575.0, 4, 0)
        entered_fight = state(70.0, 5, 3)
        step = rewards.door_shaping * (floor_potential(entered_fight)
                                       - floor_potential(at_door))
        self.assertGreaterEqual(
            step + rewards.new_room, 0.0,
            "entering a contested room must not cost reward")

        # Killing the last enemy restores the local term. Nothing else changes,
        # so the whole difference is the gate opening.
        fighting = state(300.0, 5, 2)
        cleared = state(300.0, 5, 0)
        self.assertGreater(
            floor_potential(cleared), floor_potential(fighting),
            "clearing a room must raise the potential, not lower it")

        # And while the fight is on, moving does not change the potential at
        # all — that is the entire point, and the thing that stops navigation
        # competing with combat inside a contested room.
        self.assertEqual(floor_potential(state(100.0, 5, 2)),
                         floor_potential(state(500.0, 5, 2)),
                         "position must not move the potential mid-fight")

    def test_floor_shaping_still_beats_the_step_penalty(self) -> None:
        """Removing the punishment must not cost the pull toward doors.

        Normalising the potential by rooms_total fixes the discontinuity too,
        and shrinks one step of progress to the same order as the step penalty
        — the regime that left nav-v1 maximising randomness instead of learning.
        """
        from isaac_ai.config import load_config
        from isaac_ai.floors import floor_potential
        rewards = load_config().rewards

        def state(x: float) -> dict:
            return {"ready": True, "player": {"x": x, "y": 280.0},
                    "level": {"rooms_visited": 5, "rooms_total": 12},
                    "room": {"top_left_x": 60.0, "top_left_y": 140.0,
                             "bottom_right_x": 580.0, "bottom_right_y": 420.0,
                             "doors": [{"x": 580.0, "y": 280.0,
                                        "visited": 0, "locked": False}]}}

        # One tick of movement is roughly four pixels.
        step = rewards.door_shaping * (floor_potential(state(104.0))
                                       - floor_potential(state(100.0)))
        self.assertGreater(step, abs(rewards.step) * 5)

    def test_locked_doors_never_attract_either_way(self) -> None:
        from isaac_ai.floors import door_potential
        doors = [{"x": 580.0, "y": 280.0, "visited": 0, "locked": True, "slot": 2}]
        self.assertEqual(
            door_potential(self.raw(500.0, doors), unvisited_only=False), 0.0)

    def test_no_doors_is_flat(self) -> None:
        from isaac_ai.floors import door_potential
        self.assertEqual(door_potential(self.raw(300.0, [])), 0.0)
        self.assertEqual(door_potential({}), 0.0)

    def test_potential_is_non_negative(self) -> None:
        """Shaping around a cycle sums to (gamma-1)*sum(potentials). With a
        negative potential that is positive, so a two-room loop paid out on
        every lap. Non-negative potentials make cycling cost, not earn."""
        from isaac_ai.floors import door_potential
        door = [{"x": 580.0, "y": 280.0, "visited": 0, "locked": False, "slot": 2}]
        for x in (60.0, 200.0, 400.0, 580.0):
            self.assertGreaterEqual(door_potential(self.raw(x, door)), 0.0)

    def test_two_room_cycle_does_not_pay(self) -> None:
        """Walk to a door and back; the shaping must not net positive."""
        from isaac_ai.floors import door_potential
        gamma = 0.99
        door = [{"x": 580.0, "y": 280.0, "visited": 0, "locked": False, "slot": 2}]
        near = door_potential(self.raw(500.0, door))
        far = door_potential(self.raw(100.0, door))
        # far -> near -> far, using gamma*phi(s') - phi(s) each leg.
        total = (gamma * near - far) + (gamma * far - near)
        self.assertLessEqual(total, 0.0)


class TestNavigationScoring(unittest.TestCase):
    def test_success_counts_distinct_rooms_not_crossings(self) -> None:
        """Requiring N crossings is satisfiable by walking one door back and
        forth N times — full marks for going nowhere. Success must be keyed on
        distinct rooms reached."""
        source = (Path(__file__).resolve().parents[1]
                  / "src" / "isaac_ai" / "navigation.py").read_text(encoding="utf-8")
        self.assertIn("self._rooms_reached[index] >= self._required[index]", source)
        self.assertNotIn("self._transitions[index] >= self._required[index]", source)

    def test_arrival_reward_is_gated_on_a_new_room(self) -> None:
        source = (Path(__file__).resolve().parents[1]
                  / "src" / "isaac_ai" / "navigation.py").read_text(encoding="utf-8")
        arrival = source.index("reward += self.arrival_reward")
        guard = source.index("room_index not in self._visited[index]")
        self.assertLess(guard, arrival,
                        "the arrival bonus must sit inside the new-room guard")


class TestNavigationPotential(unittest.TestCase):
    """Shaping must not fight the reward.

    Nearest-door shaping pulls back through the door just used, because that is
    the closest one — while only new rooms pay. nav-v3 backtracked 2.34 crossings
    per room reached as a result.
    """

    def obs(self, player_x: float, doors: list[dict]) -> dict:
        return {
            "ready": True,
            "player": {"x": player_x, "y": 280.0},
            "room": {"top_left_x": 60, "top_left_y": 140,
                     "bottom_right_x": 580, "bottom_right_y": 420,
                     "doors": doors},
        }

    def test_prefers_the_door_leading_somewhere_new(self) -> None:
        from isaac_ai.navigation import _mark_doors_for_episode, navigation_potential
        # Came in through the near door on the right; the far one is unexplored.
        doors = [{"x": 560.0, "y": 280.0, "target": 11, "locked": False, "slot": 2},
                 {"x": 80.0, "y": 280.0, "target": 22, "locked": False, "slot": 0}]
        observation = self.obs(500.0, doors)
        _mark_doors_for_episode(observation, {11})
        near_the_used_door = navigation_potential(observation)

        moved = self.obs(200.0, [dict(d) for d in doors])
        _mark_doors_for_episode(moved, {11})
        heading_for_the_new_one = navigation_potential(moved)

        self.assertGreater(heading_for_the_new_one, near_the_used_door)

    def test_dead_end_still_pulls_toward_its_only_door(self) -> None:
        from isaac_ai.navigation import _mark_doors_for_episode, navigation_potential
        doors = [{"x": 560.0, "y": 280.0, "target": 11, "locked": False, "slot": 2}]
        observation = self.obs(500.0, doors)
        _mark_doors_for_episode(observation, {11})
        self.assertGreater(navigation_potential(observation), 0.0)

    def test_marking_is_per_episode_not_per_run(self) -> None:
        from isaac_ai.navigation import _mark_doors_for_episode
        doors = [{"x": 560.0, "y": 280.0, "target": 11, "locked": False,
                  "slot": 2, "visited": 47}]
        observation = self.obs(500.0, doors)
        _mark_doors_for_episode(observation, set())
        # A room the run has seen 47 times is new to THIS episode.
        self.assertEqual(observation["room"]["doors"][0]["visited"], 0)


class TestNavigationCurriculum(unittest.TestCase):
    def test_starts_at_one_traversal(self) -> None:
        from isaac_ai.nav_curriculum import NavigationCurriculum
        self.assertEqual(NavigationCurriculum().required_transitions(), 1)

    def test_dial_actually_changes_the_task(self) -> None:
        """Unlike the floor dial, this one must change what the agent does —
        more traversals per episode is a genuinely longer walk."""
        from isaac_ai.nav_curriculum import NavigationCurriculum
        curriculum = NavigationCurriculum()
        easy = curriculum.required_transitions()
        curriculum.difficulty = 1.0
        self.assertGreater(curriculum.required_transitions(), easy)
        self.assertEqual(curriculum.required_transitions(),
                         curriculum.max_transitions)

    def test_recording_alone_never_moves_the_dial(self) -> None:
        from isaac_ai.nav_curriculum import NavigationCurriculum
        curriculum = NavigationCurriculum(window=10, adjust_rate=0.1)
        for _ in range(200):
            curriculum.record(True, 1)
        self.assertEqual(curriculum.difficulty, 0.0)
        curriculum.advance()
        self.assertGreater(curriculum.difficulty, 0.0)


class TestPixelCapture(unittest.TestCase):
    """Frame handling for the pixel student, without a live window."""

    def config(self, **kwargs):
        from isaac_ai.config import PixelConfig
        return PixelConfig(**{"width": 4, "height": 2, "stack": 3,
                              "grayscale": False, **kwargs})

    def fleet(self, num_envs: int = 2, **kwargs):
        from isaac_ai.capture import FleetCapture
        return FleetCapture([None] * num_envs, self.config(**kwargs))

    def test_input_size_must_divide_the_client_area(self) -> None:
        """An inexact ratio stops the downsample being a box filter."""
        self.config().check_divides(480, 270)  # 120x135 blocks, fine
        with self.assertRaises(ValueError):
            self.config(width=7).check_divides(480, 270)
        with self.assertRaises(ValueError):
            self.config(height=4).check_divides(480, 270)

    def test_downsample_averages_exact_blocks(self) -> None:
        """Over an integer ratio the result must be the block mean."""
        from isaac_ai.capture import downsample

        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        frame[:2, :2] = 100  # one 2x2 block solid, the rest zero
        small = downsample(frame, 2, 2, grayscale=False)
        self.assertEqual(small.shape, (2, 2, 3))
        self.assertEqual(int(small[0, 0, 0]), 100)
        self.assertEqual(int(small[1, 1, 0]), 0)

    def test_grayscale_keeps_a_channel_axis(self) -> None:
        """Dropping to 2-D here would break the transpose into the stack."""
        from isaac_ai.capture import downsample

        small = downsample(np.zeros((4, 4, 3), np.uint8), 2, 2, grayscale=True)
        self.assertEqual(small.shape, (2, 2, 1))

    def test_stack_is_oldest_to_newest(self) -> None:
        """Reversed order trains the student on time running backwards."""
        fleet = self.fleet()
        for value in (1, 2, 3):
            fleet.push(0, np.full((2, 4, 3), value, dtype=np.uint8))

        observation = fleet.observe()
        self.assertEqual(observation.shape, (2, 9, 2, 4))
        # Three RGB frames, oldest in the leading channels.
        self.assertEqual(int(observation[0, 0, 0, 0]), 1)
        self.assertEqual(int(observation[0, 3, 0, 0]), 2)
        self.assertEqual(int(observation[0, 6, 0, 0]), 3)

    def test_stack_drops_the_oldest_frame(self) -> None:
        fleet = self.fleet()
        for value in (1, 2, 3, 4):
            fleet.push(0, np.full((2, 4, 3), value, dtype=np.uint8))

        observation = fleet.observe()
        self.assertEqual(int(observation[0, 0, 0, 0]), 2)
        self.assertEqual(int(observation[0, 6, 0, 0]), 4)

    def test_reset_clears_only_the_finished_episodes(self) -> None:
        """Carrying frames across a reset shows a scene that teleported."""
        fleet = self.fleet()
        fleet.push(0, np.full((2, 4, 3), 5, dtype=np.uint8))
        fleet.push(1, np.full((2, 4, 3), 6, dtype=np.uint8))

        fleet.reset(np.array([True, False]))
        observation = fleet.observe()
        self.assertEqual(int(observation[0].max()), 0)
        self.assertEqual(int(observation[1].max()), 6)

    def test_observe_advances_the_stack_every_call(self) -> None:
        """The stack tracks agent steps, not distinct frames.

        Two observes with no new frame between them therefore push the same
        picture twice, on purpose: skipping it would let the stack's spacing
        drift out of step with the actions taken. A probe that called observe
        an extra time read this as capture failing.
        """
        fleet = self.fleet()
        fleet.push(0, np.full((2, 4, 3), 1, dtype=np.uint8))
        fleet.push(0, np.full((2, 4, 3), 2, dtype=np.uint8))
        fleet.push(0, np.full((2, 4, 3), 2, dtype=np.uint8))

        observation = fleet.observe()
        self.assertEqual(int(observation[0, 3, 0, 0]), 2)
        self.assertEqual(int(observation[0, 6, 0, 0]), 2)

    def test_new_episode_fills_the_stack_instead_of_padding_black(self) -> None:
        """Black padding is a perfect cue that an episode just started.

        It exists nowhere at deployment, so the network can learn to rely on
        something that will never appear again. Repeating the first frame says
        the true thing: nothing has moved yet.
        """
        fleet = self.fleet()
        fleet.push(0, np.full((2, 4, 3), 7, dtype=np.uint8))

        observation = fleet.observe()
        self.assertEqual(int(observation[0].min()), 7)
        self.assertEqual(int(observation[0].max()), 7)

        # Once filled, it must roll normally again rather than re-filling.
        fleet.push(0, np.full((2, 4, 3), 9, dtype=np.uint8))
        observation = fleet.observe()
        self.assertEqual(int(observation[0, 0, 0, 0]), 7)
        self.assertEqual(int(observation[0, 6, 0, 0]), 9)

    def test_reset_restores_fill_behaviour(self) -> None:
        fleet = self.fleet()
        fleet.push(0, np.full((2, 4, 3), 1, dtype=np.uint8))
        fleet.push(0, np.full((2, 4, 3), 2, dtype=np.uint8))

        fleet.reset(np.array([True, False]))
        fleet.push(0, np.full((2, 4, 3), 5, dtype=np.uint8))
        observation = fleet.observe()
        self.assertEqual(int(observation[0].min()), 5)
        self.assertEqual(int(observation[0].max()), 5)

    def test_missing_window_yields_a_stack_not_a_crash(self) -> None:
        """A fleet member without a window must not take the batch down."""
        observation = self.fleet(num_envs=3).observe()
        self.assertEqual(observation.shape, (3, 9, 2, 4))
        self.assertEqual(int(observation.max()), 0)


class TestPixelPolicy(unittest.TestCase):
    """The student, and the asymmetry that is the point of it."""

    SHAPE = (12, 90, 160)

    def build(self):
        from isaac_ai.pixel_policy import PixelActorCritic
        return PixelActorCritic(self.SHAPE, SCALAR_FEATURES,
                                MAX_DOORS * DOOR_FEATURES)

    def frames(self, batch: int = 4):
        return torch.randint(0, 256, (batch,) + self.SHAPE, dtype=torch.uint8)

    def test_shapes(self) -> None:
        model = self.build()
        actions, log_prob, value = model.act(self.frames(), fake_observation(4))
        self.assertEqual(actions.shape, (4, len(ACTION_DIMS)))
        self.assertEqual(log_prob.shape, (4,))
        self.assertEqual(value.shape, (4,))

    def test_flatten_width_follows_the_input_size(self) -> None:
        """Changing [pixels] in config must not need a hand-edited layer."""
        from isaac_ai.pixel_policy import PixelActorCritic

        for shape in ((12, 90, 160), (4, 54, 96), (12, 45, 80)):
            model = PixelActorCritic(shape, SCALAR_FEATURES,
                                     MAX_DOORS * DOOR_FEATURES)
            frames = torch.randint(0, 256, (2,) + shape, dtype=torch.uint8)
            self.assertEqual(model.logits(frames)[0].shape,
                             (2, ACTION_DIMS[0]))

    def test_deployment_path_needs_only_pixels(self) -> None:
        actions = self.build().act_from_pixels(self.frames())
        self.assertEqual(actions.shape, (4, len(ACTION_DIMS)))

    def test_no_privileged_state_reaches_the_actor(self) -> None:
        """The whole design rests on this: the actor must be pixels-only.

        A stray connection from the privileged encoders into the action heads
        would train beautifully and then collapse the day the mod is removed.
        """
        model = self.build()
        model.logits(self.frames())[0].sum().backward()

        leaked = [name for name, param in model.critic.named_parameters()
                  if param.grad is not None]
        self.assertEqual(leaked, [], f"privileged params in the actor: {leaked}")

    def test_critic_reads_the_privileged_state(self) -> None:
        """And the converse — the critic must actually use what it is given."""
        model = self.build()
        state = fake_observation(4)
        before = model.critic(state)
        state["scalars"] = state["scalars"] + 5.0
        self.assertFalse(torch.allclose(before, model.critic(state)))

    def test_distillation_loss_is_zero_on_a_perfect_match(self) -> None:
        from isaac_ai.pixel_policy import distillation_loss

        logits = [torch.randn(6, dim) for dim in ACTION_DIMS]
        loss = distillation_loss(logits, [head.clone() for head in logits])
        self.assertAlmostEqual(float(loss), 0.0, places=5)

    def test_distillation_loss_punishes_disagreement(self) -> None:
        from isaac_ai.pixel_policy import distillation_loss

        teacher = [torch.zeros(6, dim) for dim in ACTION_DIMS]
        confident = [torch.full((6, dim), 0.0) for dim in ACTION_DIMS]
        confident[0][:, 0] = 10.0  # student certain where teacher is uniform
        self.assertGreater(float(distillation_loss(confident, teacher)), 0.1)

    def test_distillation_moves_the_student_towards_the_teacher(self) -> None:
        """Soft labels have to be learnable, not merely well-defined."""
        from isaac_ai.pixel_policy import distillation_loss

        model = self.build()
        frames = self.frames(8)
        teacher = [torch.randn(8, dim) for dim in ACTION_DIMS]

        optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
        first = float(distillation_loss(model.logits(frames), teacher))
        for _ in range(30):
            loss = distillation_loss(model.logits(frames), teacher)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        self.assertLess(float(distillation_loss(model.logits(frames), teacher)),
                        first)


class TestDistillation(unittest.TestCase):
    """The distiller's logic, without a fleet."""

    def distiller(self, **kwargs):
        from isaac_ai.distill import DistillConfig, Distiller

        distiller = Distiller.__new__(Distiller)
        distiller.config = DistillConfig(**kwargs)
        distiller.global_step = 0
        return distiller

    def test_teacher_load_refuses_a_partial_fit(self) -> None:
        """A half-loaded teacher produces confident logits from noise."""
        import tempfile
        from isaac_ai.distill import load_teacher
        from isaac_ai.policy import ActorCritic

        state = ActorCritic().state_dict()
        # Same shape change combat-v3 has: the pre-door scalar encoder.
        state["scalar_encoder.0.weight"] = torch.randn(128, SCALAR_FEATURES - 3)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "policy.pt"
            torch.save({"policy": state}, path)
            with self.assertRaises(ValueError) as caught:
                load_teacher(path, torch.device("cpu"))
        self.assertIn("scalar_encoder.0.weight", str(caught.exception))

    def test_student_share_ramps_between_its_bounds(self) -> None:
        distiller = self.distiller(student_share_start=0.0,
                                   student_share_end=0.8,
                                   student_share_steps=1000)
        self.assertAlmostEqual(distiller.student_share(), 0.0)
        distiller.global_step = 500
        self.assertAlmostEqual(distiller.student_share(), 0.4)
        distiller.global_step = 1000
        self.assertAlmostEqual(distiller.student_share(), 0.8)

    def test_student_share_does_not_overshoot(self) -> None:
        """Past the ramp it must hold, not keep climbing above the end value."""
        distiller = self.distiller(student_share_end=0.8,
                                   student_share_steps=1000)
        distiller.global_step = 10_000
        self.assertAlmostEqual(distiller.student_share(), 0.8)

    def test_driver_assignment_follows_the_share(self) -> None:
        distiller = self.distiller(student_share_start=1.0,
                                   student_share_end=1.0)
        distiller._student_driven = np.zeros(6, dtype=bool)
        distiller._assign_drivers(np.array([True, True, False, False, False, False]))
        # Only the episodes that ended get reassigned.
        self.assertTrue(distiller._student_driven[:2].all())
        self.assertFalse(distiller._student_driven[2:].any())

    def test_update_consumes_a_rollout_and_reduces_kl(self) -> None:
        """Exercises the reshape and per-head split, not just the loss.

        The rollout arrives as (steps, envs, ...) and the teacher's logits as
        one concatenated block; getting either regrouping wrong still runs and
        still produces a falling number, just against the wrong head.
        """
        from isaac_ai.pixel_policy import PixelActorCritic

        shape = (4, 54, 96)
        distiller = self.distiller(epochs=4, minibatch_size=8)
        distiller.device = torch.device("cpu")
        distiller.student = PixelActorCritic(shape, SCALAR_FEATURES,
                                             MAX_DOORS * DOOR_FEATURES)
        distiller.optimizer = torch.optim.Adam(distiller.student.parameters(),
                                               lr=1e-3)

        steps, envs = 4, 3
        frames = torch.randint(0, 256, (steps, envs) + shape, dtype=torch.uint8)
        logits = torch.randn(steps, envs, sum(ACTION_DIMS))

        first = distiller.update(frames.clone(), logits.clone())
        for _ in range(3):
            last = distiller.update(frames.clone(), logits.clone())
        self.assertLess(last["kl"], first["kl"])

    def test_ramp_steps_on_evidence_not_on_update_count(self) -> None:
        """Repeated advance() calls without new episodes must not compound.

        Measured in distill-v2: advance() fired every update while only 10% of
        episodes fed the window, so difficulty ran 0.00 -> 1.00 -> 0.00 at full
        scale. The ramp has to be driven by arriving evidence, not by how often
        the trainer happens to update.
        """
        curriculum = CombatCurriculum()
        for _ in range(curriculum.window):
            curriculum.record(True)

        curriculum.advance()
        after_one = curriculum.difficulty
        self.assertGreater(after_one, 0.0)

        # No new episodes: 50 more calls must change nothing.
        for _ in range(50):
            curriculum.advance()
        self.assertAlmostEqual(curriculum.difficulty, after_one)

        # Fresh evidence unlocks exactly one more step.
        for _ in range(curriculum.refresh):
            curriculum.record(True)
        curriculum.advance()
        curriculum.advance()
        self.assertAlmostEqual(curriculum.difficulty,
                               after_one + curriculum.adjust_rate)

    def test_curriculum_mask_excludes_student_episodes(self) -> None:
        """The student's losses must not drag the teacher's difficulty down.

        Measured in distill-v1 before this existed: difficulty swung between
        0.27 and 0.81 — 2 to 9 enemy types — inside 80k steps, because every
        student loss was recorded as the curriculum's own failure. The target
        the student was fitting moved faster than it could learn.
        """
        from isaac_ai.combat import CombatVecEnv

        env = CombatVecEnv.__new__(CombatVecEnv)
        env.num_envs = 3
        env.curriculum = CombatCurriculum()
        env._failed = [False] * 3

        def record(index: int, success: bool) -> None:
            if (env.curriculum_mask is None
                    or bool(env.curriculum_mask[index])):
                env.curriculum.record(success)

        env.curriculum_mask = np.array([True, False, False])
        record(0, True)   # teacher: counted
        record(1, False)  # student: ignored
        record(2, False)  # student: ignored
        self.assertEqual(env.curriculum.success_rate, 1.0)

        # Unset, plain training must be unaffected.
        env.curriculum = CombatCurriculum()
        env.curriculum_mask = None
        record(0, True)
        record(1, False)
        self.assertAlmostEqual(env.curriculum.success_rate, 0.5)

    def one_hot(self, choices: list[list[int]]) -> list[torch.Tensor]:
        """Logits peaked on the given action per head."""
        return [torch.nn.functional.one_hot(torch.tensor(head), 3).float() * 10.0
                for head in choices]

    def test_agreement_counts_movement_only(self) -> None:
        """Folding in the shoot heads reports wrong movement as half right."""
        distiller = self.distiller()
        # Heads are [move_x, move_y, shoot_x, shoot_y], two samples each.
        student = self.one_hot([[0, 1], [1, 1], [2, 0], [2, 0]])
        teacher = self.one_hot([[0, 2], [1, 2], [0, 0], [0, 0]])
        # Sample 0 matches on both move heads, sample 1 on neither: 2 of 4.
        # The shoot heads disagree everywhere and must not count.
        self.assertAlmostEqual(distiller._agreement(student, teacher), 0.5)

    def test_agreement_compares_modes_not_samples(self) -> None:
        """A student matching the teacher exactly must score 1.0.

        Comparing sampled actions instead caps a flawless student at the
        distribution's collision rate — about 0.5 at combat-v4's entropy —
        which made a run that was 41% of the way there look stalled at chance.
        """
        distiller = self.distiller()
        teacher = [torch.randn(16, 3) for _ in ACTION_DIMS]
        student = [head.clone() for head in teacher]
        self.assertAlmostEqual(distiller._agreement(student, teacher), 1.0)


class TestPixelFineTuning(unittest.TestCase):
    """RL fine-tuning of the pixel student against the privileged critic."""

    SHAPE = (12, 90, 160)

    def model(self):
        from isaac_ai.pixel_policy import PixelActorCritic
        return PixelActorCritic(self.SHAPE, SCALAR_FEATURES,
                                MAX_DOORS * DOOR_FEATURES)

    def test_value_loss_cannot_reach_the_actor(self) -> None:
        """This is what makes critic warm-up safe without freezing anything.

        The warm-up exists because distillation never trains the critic, so its
        advantages start as noise and would wreck a policy that cost a million
        steps. It only works if a value loss has no gradient path into the
        actor — which holds because the critic reads state and the actor reads
        pixels through entirely separate networks.
        """
        model = self.model()
        model.critic(fake_observation(4)).sum().backward()

        actor = [name for name, param in model.named_parameters()
                 if not name.startswith("critic") and param.grad is not None]
        self.assertEqual(actor, [], f"value loss reached the actor: {actor}")

    def test_warmup_flag_flips_after_its_updates(self) -> None:
        from isaac_ai.pixel_ppo import PixelPPOConfig, PixelPPOTrainer

        trainer = PixelPPOTrainer.__new__(PixelPPOTrainer)
        trainer.config = PixelPPOConfig(critic_warmup_updates=3)
        trainer.updates = 0
        self.assertTrue(trainer.warming_up)
        trainer.updates = 2
        self.assertTrue(trainer.warming_up)
        trainer.updates = 3
        self.assertFalse(trainer.warming_up)

    def test_per_axis_entropy_separates_a_dead_head(self) -> None:
        """The summed figure cannot distinguish these two, and it matters.

        combat-v5 through v7 logged shoot_entropy ~1.10, which reads as a policy
        halfway to confident. It was one axis frozen and one abandoned, and the
        agent could not shoot sideways at all for three runs.
        """
        import math

        frozen_plus_uniform = 0.05 + 1.07     # what those runs actually were
        both_half_committed = 0.55 + 0.55     # what the sum made it look like
        self.assertAlmostEqual(frozen_plus_uniform, both_half_committed, places=1)

        # Per axis they are unmistakable: one at the ln(3) ceiling is abandoned.
        self.assertGreater(1.07, math.log(3) - 0.15)
        self.assertLess(0.55, math.log(3) - 0.15)

    def test_critic_seeds_from_the_teacher(self) -> None:
        """The critic is the teacher's network minus its action heads.

        Every tensor matches, value head included, so there is no reason to
        learn a value function from reward when one trained for a million steps
        on the same observation is sitting in memory. Two fine-tuning runs moved
        difficulty 0.02 and 0.00 with a critic starting from noise.
        """
        from isaac_ai.pixel_ppo import PixelPPOConfig, PixelPPOTrainer
        from isaac_ai.policy import ActorCritic

        trainer = PixelPPOTrainer.__new__(PixelPPOTrainer)
        trainer.config = PixelPPOConfig()
        trainer.teacher = ActorCritic()
        trainer.policy = self.model()

        copied, total = trainer.seed_critic()
        self.assertEqual(copied, total)

        teacher_state = trainer.teacher.state_dict()
        for key, value in trainer.policy.critic.state_dict().items():
            self.assertTrue(torch.equal(value, teacher_state[key]), key)

    def test_seeding_the_critic_leaves_the_actor_alone(self) -> None:
        """Seeding must not disturb the distilled policy it starts from."""
        from isaac_ai.pixel_ppo import PixelPPOConfig, PixelPPOTrainer
        from isaac_ai.policy import ActorCritic

        trainer = PixelPPOTrainer.__new__(PixelPPOTrainer)
        trainer.config = PixelPPOConfig()
        trainer.teacher = ActorCritic()
        trainer.policy = self.model()

        before = {k: v.clone() for k, v in trainer.policy.state_dict().items()
                  if not k.startswith("critic")}
        trainer.seed_critic()
        for key, value in before.items():
            self.assertTrue(
                torch.equal(value, trainer.policy.state_dict()[key]), key)

    def test_explained_variance_reads_as_advertised(self) -> None:
        """A critic that predicts the mean must score 0, not look adequate.

        value_loss alone cannot distinguish a bad critic from a volatile task,
        and PPO running on uninformative advantages is indistinguishable from
        "fine-tuning does not help" unless this is measured.
        """
        def explained(values: torch.Tensor, returns: torch.Tensor) -> float:
            variance = returns.var()
            return float(1.0 - (returns - values).var() / variance)

        returns = torch.randn(256) * 3.0 + 5.0
        self.assertAlmostEqual(explained(returns.clone(), returns), 1.0, places=4)
        self.assertAlmostEqual(
            explained(torch.full_like(returns, float(returns.mean())), returns),
            0.0, places=2)
        self.assertLess(explained(-returns, returns), 0.0)

    def test_rollout_holds_frames_as_bytes(self) -> None:
        """float32 frames would be half a gigabyte per rollout for no benefit."""
        from isaac_ai.pixel_ppo import PixelRollout

        buffer = PixelRollout(8, 4, self.SHAPE, torch.device("cpu"))
        self.assertEqual(buffer.frames.dtype, torch.uint8)
        self.assertEqual(tuple(buffer.frames.shape), (8, 4) + self.SHAPE)
        self.assertEqual(tuple(buffer.teacher_logits.shape),
                         (8, 4, sum(ACTION_DIMS)))

    def test_finetuned_student_still_loads_as_a_student(self) -> None:
        """A fine-tune must stay loadable by the same diagnostic as a distil."""
        import tempfile
        from isaac_ai.pixel_ppo import load_student

        model = self.model()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "student.pt"
            torch.save({"student": model.state_dict(),
                        "pixel_shape": self.SHAPE}, path)
            reloaded = load_student(path, torch.device("cpu"))

        frames = torch.randint(0, 256, (2,) + self.SHAPE, dtype=torch.uint8)
        self.assertEqual(reloaded.act_from_pixels(frames).shape,
                         (2, len(ACTION_DIMS)))


class TestAdvantages(unittest.TestCase):
    def test_gae_matches_hand_computation(self) -> None:
        from isaac_ai.ppo import PPOConfig, PPOTrainer

        class StubEnv:
            num_envs = 1

        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.env = StubEnv()
        trainer.device = torch.device("cpu")
        trainer.config = PPOConfig(rollout_steps=3, gamma=0.9, gae_lambda=0.5)

        from isaac_ai.ppo import RolloutBuffer
        trainer.buffer = RolloutBuffer(3, 1, torch.device("cpu"))
        trainer.buffer.rewards = torch.tensor([[1.0], [1.0], [1.0]])
        trainer.buffer.values = torch.tensor([[0.0], [0.0], [0.0]])
        trainer.buffer.dones = torch.tensor([[0.0], [0.0], [0.0]])

        advantages, returns = trainer.advantages(torch.tensor([0.0]))

        # With zero values, delta = r + gamma*0 - 0 = 1 at every step, so
        # gae_t = 1 + gamma*lambda*gae_{t+1}, computed backwards from 0.
        expected_last = 1.0
        expected_mid = 1.0 + 0.9 * 0.5 * expected_last
        expected_first = 1.0 + 0.9 * 0.5 * expected_mid
        self.assertAlmostEqual(float(advantages[2, 0]), expected_last, places=5)
        self.assertAlmostEqual(float(advantages[1, 0]), expected_mid, places=5)
        self.assertAlmostEqual(float(advantages[0, 0]), expected_first, places=5)
        self.assertTrue(torch.allclose(returns, advantages))

    def test_done_truncates_bootstrap(self) -> None:
        from isaac_ai.ppo import PPOConfig, PPOTrainer, RolloutBuffer

        class StubEnv:
            num_envs = 1

        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.env = StubEnv()
        trainer.device = torch.device("cpu")
        trainer.config = PPOConfig(rollout_steps=2, gamma=0.9, gae_lambda=1.0)
        trainer.buffer = RolloutBuffer(2, 1, torch.device("cpu"))
        trainer.buffer.rewards = torch.tensor([[1.0], [1.0]])
        trainer.buffer.values = torch.tensor([[0.0], [0.0]])
        # Episode ends at step 0: future reward must not leak backwards.
        trainer.buffer.dones = torch.tensor([[1.0], [0.0]])

        advantages, _ = trainer.advantages(torch.tensor([0.0]))
        self.assertAlmostEqual(float(advantages[0, 0]), 1.0, places=5)


class TestBlockedMovePenalty(unittest.TestCase):
    """Walking into a wall has to cost more than walking in the open.

    floor-v12 was pinned against geometry on 18.6% of the steps it tried to
    move, against 2.5% for a random walk on the same fleet. A blocked step costs
    the step penalty and nothing else — no position change means no potential
    change — so grinding a wall and drifting uselessly paid exactly the same.
    That absent gradient is why the obstacle grid, and then the egocentric
    version of it, were both ignored for three runs.
    """

    def rates(self, requests, blocked, still):
        from isaac_ai.ppo import PPOTrainer

        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer.env = type("E", (), {"move_requests": requests,
                                     "blocked_moves": blocked,
                                     "still_steps": still})()
        return PPOTrainer._movement_rates(trainer)

    def test_rates_are_reported_for_the_metrics(self):
        rates = self.rates(1000, 186, 50)
        self.assertAlmostEqual(rates["blocked_rate"], 0.186)
        self.assertAlmostEqual(rates["still_rate"], 50 / 1050, places=4)

    def test_no_movement_yet_reports_nothing_rather_than_zero(self):
        """A rate of 0.0 before any step would read as 'measured, perfect'."""
        self.assertEqual(self.rates(0, 0, 0), {})

    def test_standing_still_is_cheaper_than_being_blocked(self):
        """The ordering the penalty has to preserve.

        Standing still is left uncharged on purpose — it is strictly better
        than grinding a wall and earns nothing either way, so it is a refuge,
        not a strategy. But a productive step must still beat both, or the agent
        learns to freeze instead of to steer. `still_rate` is logged so that
        failure is visible if the value is ever set too high.

        The penalty is **off** in the v12 reconstruction, because v12 predates it
        — it arrived in v13, the first run of the decline. So the ordering is
        asserted against whichever configuration is loaded rather than against
        the one that happened to be current when this was written: with the
        penalty off, being blocked costs exactly what idling costs, which is the
        documented defect the penalty exists to fix. Re-enabling it re-arms the
        stricter branch automatically.
        """
        from isaac_ai.config import load_config

        rewards = load_config().rewards
        step = rewards.step
        blocked = step + rewards.blocked_move
        # A step of real progress: door shaping over roughly ten units of a
        # ~520-unit room.
        productive = step + rewards.door_shaping * (10.0 / 520.0)

        if rewards.blocked_move == 0.0:
            self.assertEqual(blocked, step,
                             "with the penalty off, a blocked step must cost "
                             "exactly what idling costs — that is the defect, "
                             "and it must not be masked by some other term")
        else:
            self.assertLess(blocked, step,
                            "being blocked must cost more than idling")
        self.assertGreater(productive, step, "progress must beat idling")
        self.assertGreaterEqual(productive, blocked)
        self.assertLess(abs(rewards.blocked_move), rewards.room_clear,
                        "avoiding walls must never beat clearing a room")


class TestGridProperties(unittest.TestCase):
    """A tile's properties are not mutually exclusive, and pretending they were
    lost real information.

    A spiked rock blocks movement *and* damages on contact. The single-class
    encoding had to pick one, picked solid, and so reported it as an ordinary
    rock — the agent was never told it hurts. Watching v11 play, it walked into
    spikes and fires repeatedly.
    """

    SOLID, HAZARD, PIT, DAMAGING, DESTRUCTIBLE, RETRACTABLE = (
        1 << 0, 1 << 1, 1 << 2, 1 << 3, 1 << 4, 1 << 5)

    def planes(self, mask):
        from isaac_ai.env import (GRID_CLASSES, GRID_HEIGHT, GRID_WIDTH,
                                  encode_grid)
        cells = [0] * (GRID_WIDTH * GRID_HEIGHT)
        cells[4 * GRID_WIDTH + 7] = mask
        flat = encode_grid({"grid": cells, "grid_width": GRID_WIDTH})
        return flat.reshape(GRID_CLASSES, GRID_HEIGHT, GRID_WIDTH)[:, 4, 7]

    @unittest.skipIf(__import__("isaac_ai.env", fromlist=["env"]).GRID_CLASSES < 6,
                     "new property planes not enabled yet (GRID_CLASSES == 3)")
    def test_a_spiked_rock_is_both_solid_and_damaging(self):
        """The case the old encoding could not express at all."""
        cell = self.planes(self.SOLID | self.DAMAGING)
        self.assertEqual(float(cell[0]), 1.0, "should still be solid")
        self.assertEqual(float(cell[3]), 1.0, "should also be damaging")

    def test_the_original_three_planes_keep_their_meaning(self):
        """Planes 0-2 must be untouched or every trained grid weight shifts."""
        self.assertEqual(list(self.planes(self.SOLID))[:3], [1.0, 0.0, 0.0])
        self.assertEqual(list(self.planes(self.HAZARD))[:3], [0.0, 1.0, 0.0])
        self.assertEqual(list(self.planes(self.PIT))[:3], [0.0, 0.0, 1.0])

    @unittest.skipIf(__import__("isaac_ai.env", fromlist=["env"]).GRID_CLASSES < 6,
                     "new property planes not enabled yet (GRID_CLASSES == 3)")
    def test_new_planes_are_appended_after_the_original_three(self):
        """Appending is what makes a resumed policy transfer exactly.

        The planes are the leading dimension of the flattened grid, so a new one
        lands at the end of the vector and a widened encoder keeps every old
        column. Inserting one would shift all 405 later values.
        """
        from isaac_ai.env import GRID_ALL_CLASSES, GRID_CLASSES
        self.assertEqual(GRID_CLASSES, GRID_ALL_CLASSES)
        self.assertEqual(float(self.planes(self.DESTRUCTIBLE)[4]), 1.0)
        self.assertEqual(float(self.planes(self.RETRACTABLE)[5]), 1.0)
        # ...and adding one must not light up any of the original three.
        self.assertEqual(list(self.planes(self.RETRACTABLE))[:3], [0.0, 0.0, 0.0])

    @unittest.skipIf(__import__("isaac_ai.env", fromlist=["env"]).GRID_CLASSES < 6,
                     "new property planes not enabled yet (GRID_CLASSES == 3)")
    def test_a_large_room_merges_properties_rather_than_classes(self):
        """Downsampling must OR the bits, not compare to a single value."""
        from isaac_ai.env import (GRID_CLASSES, GRID_HEIGHT, GRID_WIDTH,
                                  encode_grid)
        wide, tall = GRID_WIDTH * 2, GRID_HEIGHT * 2
        cells = [0] * (wide * tall)
        # Two different tiles inside one output block.
        cells[0 * wide + 0] = self.SOLID
        cells[0 * wide + 1] = self.HAZARD | self.DAMAGING
        flat = encode_grid({"grid": cells, "grid_width": wide})
        cell = flat.reshape(GRID_CLASSES, GRID_HEIGHT, GRID_WIDTH)[:, 0, 0]
        self.assertEqual(float(cell[0]), 1.0)
        self.assertEqual(float(cell[1]), 1.0)
        self.assertEqual(float(cell[3]), 1.0)


class TestObstacleRays(unittest.TestCase):
    """"How far can I go this way", the feature the retreat finding asked for.

    `diagnose_cornered.py` measured blocked retreat at 41.2% of deaths against a
    12.7% baseline — 3.24x — and `probe_input_sensitivity.py` measured the ego
    window at 0.013 jacobian per value against doors' 0.106, so the obstacle
    information was present and not being read. These eight numbers are the same
    facts in the relational, low-dimensional form the policy demonstrably does
    read.
    """

    WIDTH, HEIGHT = 15, 9
    # Player mid-room at row 4, column 7, matching diagnose_grid_use.py.
    ROW, COLUMN = 4, 7
    UP, RIGHT, DOWN, LEFT = 0, 2, 4, 6

    def room(self, solids=(), pits=(), hazards=()):
        cells = [0] * (self.WIDTH * self.HEIGHT)
        for r in range(self.HEIGHT):
            for c in range(self.WIDTH):
                if r in (0, self.HEIGHT - 1) or c in (0, self.WIDTH - 1):
                    cells[r * self.WIDTH + c] = 1 << 0
        for r, c in solids:
            cells[r * self.WIDTH + c] = 1 << 0
        for r, c in pits:
            cells[r * self.WIDTH + c] = 1 << 2
        for r, c in hazards:
            cells[r * self.WIDTH + c] = 1 << 1
        return {"grid": cells, "grid_width": self.WIDTH,
                "top_left_x": 60.0, "top_left_y": 140.0,
                "bottom_right_x": 580.0, "bottom_right_y": 420.0}

    def rays(self, room):
        from isaac_ai.env import encode_obstacle_rays
        return encode_obstacle_rays(
            room, 320.0, 280.0, self.ROW * self.WIDTH + self.COLUMN)

    def test_distance_to_the_wall_is_measured_in_tiles(self):
        from isaac_ai.env import RAY_MAX_TILES
        rays = self.rays(self.room())
        # Interior rows are 1..7, so three tiles of headroom above row 4.
        self.assertAlmostEqual(rays[self.UP], 3 / RAY_MAX_TILES, places=5)
        # Interior columns are 1..13, so six to the right of column 7.
        self.assertAlmostEqual(rays[self.RIGHT], 6 / RAY_MAX_TILES, places=5)

    def test_a_rock_in_the_way_reads_zero_and_only_in_its_direction(self):
        rays = self.rays(self.room(solids=[(self.ROW, self.COLUMN + 1)]))
        self.assertEqual(rays[self.RIGHT], 0.0)
        self.assertGreater(rays[self.LEFT], 0.0, "an unrelated direction must "
                                                 "not be affected")

    def test_a_pit_blocks_a_walking_player(self):
        self.assertEqual(self.rays(
            self.room(pits=[(self.ROW, self.COLUMN - 1)]))[self.LEFT], 0.0)

    def test_a_hazard_does_not_block(self):
        """Spikes hurt and are crossable, and a retreat across them is real.

        Folding hazard into "blocked" would hide the one escape route that is
        merely expensive, which is the opposite of what this feature is for.
        """
        open_room = self.rays(self.room())[self.RIGHT]
        spiked = self.rays(
            self.room(hazards=[(self.ROW, self.COLUMN + 1)]))[self.RIGHT]
        self.assertEqual(spiked, open_room)

    def test_an_unreadable_grid_reads_open_not_trapped(self):
        """A dropped payload must never fabricate a "sealed in" signal.

        `encode_egocentric_grid` makes the same choice for the same reason: an
        all-zero window means open floor, and claiming walls on missing data
        would teach the agent to panic exactly when the mod hiccups.
        """
        from isaac_ai.env import RAY_FEATURES, encode_obstacle_rays
        rays = encode_obstacle_rays({}, 320.0, 280.0, None)
        self.assertEqual(list(rays), [1.0] * RAY_FEATURES)

    def test_rays_are_appended_last_so_a_resume_transfers_exactly(self):
        """Inserting instead of appending scrambles a warm start with no error.

        The rays must occupy the *trailing* columns and the leading 21 must be
        the original scalar vector untouched, or every resumed policy silently
        reads a different feature in each slot.
        """
        import numpy as np

        from isaac_ai.env import (ENABLE_OBSTACLE_RAYS, RAY_FEATURES,
                                  SCALAR_FEATURES, encode_obstacle_rays,
                                  encode_observation)
        if not ENABLE_OBSTACLE_RAYS:
            # The encoder tests above still run and still pin the semantics, so
            # the feature stays covered while it is switched off. Only the
            # *placement* check is meaningless with nothing appended — and
            # `scalars[-0:]` is the whole vector, which would pass for the wrong
            # reason rather than fail honestly.
            self.skipTest("obstacle rays disabled (see ENABLE_OBSTACLE_RAYS)")

        room = self.room(solids=[(self.ROW, self.COLUMN + 1)])
        room.update({"index": 84, "type": 1, "shape": 1, "clear": False,
                     "enemies_alive": 1, "doors": []})
        obs = {"ready": True,
               "player": {"x": 320.0, "y": 280.0, "vx": 0, "vy": 0, "hearts": 6,
                          "max_hearts": 6, "soul_hearts": 0, "bombs": 1,
                          "keys": 0, "coins": 0, "damage": 3.5, "speed": 1.0,
                          "tear_delay": 10, "range": 260, "can_fly": False,
                          "grid_index": self.ROW * self.WIDTH + self.COLUMN,
                          "is_dead": False},
               "room": room,
               "level": {"stage": 1, "rooms_total": 12, "rooms_visited": 4},
               "entities": [], "events": {}}

        scalars = encode_observation(obs)["scalars"]
        self.assertEqual(len(scalars), SCALAR_FEATURES)
        expected = encode_obstacle_rays(
            room, 320.0, 280.0, obs["player"]["grid_index"])
        np.testing.assert_allclose(scalars[-RAY_FEATURES:], expected)
        # The first scalar is the normalised player x, exactly as before the
        # rays existed — a cheap guard that nothing shifted. The player is at the
        # room's centre and the normalisation is [-1, 1], so this is 0.0.
        self.assertAlmostEqual(float(scalars[0]), 0.0, places=5)

    def test_scalar_length_matches_the_declared_size(self):
        from isaac_ai.env import SCALAR_FEATURES, SCALAR_RAY_FEATURES

        self.assertEqual(SCALAR_FEATURES, 21 + SCALAR_RAY_FEATURES)


class TestEgocentricGrid(unittest.TestCase):
    """The same obstacle must land at the same index wherever the player is.

    The room-absolute grid puts "solid one tile to my right" at a different
    index every time the player moves, so nothing could tie it to the
    player-relative entity offsets. Measured on floor-v10: moving a rock onto
    the line of fire changed the policy by a total variation of 0.019.
    """

    def room(self, cells, width, px, py, height=9):
        """A room whose extents describe the *playable* area, as the mod's do.

        `GetTopLeftPos`/`GetBottomRightPos` are the centres of the first and
        last walkable tiles, inside the wall ring, while `GetGridWidth` counts
        the ring too. Scaling one across the other put the player inside a wall
        on 27% of 18,000 live observations, so the fixtures have to describe the
        real relationship or they would pass on a mapping the game rejects.
        """
        from isaac_ai.env import encode_egocentric_grid
        return encode_egocentric_grid(
            {"top_left_x": 15.0, "top_left_y": 15.0,
             "bottom_right_x": (width - 2 + 0.5) * 10.0,
             "bottom_right_y": (height - 2 + 0.5) * 10.0,
             "grid": cells, "grid_width": width}, px, py)

    def at_tile(self, row, col):
        """Position of the centre of a grid tile, at 10 units per tile."""
        return (col + 0.5) * 10.0, (row + 0.5) * 10.0

    def test_player_tile_is_never_the_wall_ring(self):
        """The invariant that caught the playable-area mismatch.

        A player standing at the extreme left of the walkable area belongs in
        interior column 1, not in the wall at column 0.
        """
        from isaac_ai.env import EGO_RADIUS
        width, height = 15, 9
        cells = [0] * (width * height)
        for x in range(width):
            cells[x] = cells[(height - 1) * width + x] = 1
        for y in range(height):
            cells[y * width] = cells[y * width + width - 1] = 1
        for px, py in ((15.0, 15.0), (135.0, 75.0), (15.0, 75.0), (135.0, 15.0)):
            grid = self.window(self.room(cells, width, px, py))
            self.assertEqual(
                float(grid[0, EGO_RADIUS, EGO_RADIUS]), 0.0,
                f"player at the playable corner {(px, py)} reads as inside a wall")

    def window(self, flat):
        from isaac_ai.env import EGO_SIZE
        from isaac_ai.env import GRID_CLASSES
        return flat.reshape(GRID_CLASSES, EGO_SIZE, EGO_SIZE)

    def test_same_obstacle_same_index_from_different_positions(self):
        """The property the room-absolute encoding could not provide."""
        from isaac_ai.env import EGO_RADIUS
        width, height = 15, 9
        # A solid tile immediately right of the player, at two different
        # places in the room.
        for pcol, prow in ((3, 4), (10, 2)):
            cells = [0] * (width * height)
            cells[prow * width + (pcol + 1)] = 1
            px, py = self.at_tile(prow, pcol)
            flat = self.room(cells, width, px, py)
            grid = self.window(flat)
            self.assertEqual(
                float(grid[0, EGO_RADIUS, EGO_RADIUS + 1]), 1.0,
                f"solid-to-the-right not at a fixed index from {(prow, pcol)}")

    def test_off_map_reads_as_wall_not_empty(self):
        """Zero-filling would say the agent can walk out through the edge."""
        from isaac_ai.env import EGO_RADIUS
        width, height = 15, 9
        cells = [0] * (width * height)
        flat = self.room(cells, width, *self.at_tile(1, 1))  # playable corner
        grid = self.window(flat)
        self.assertEqual(float(grid[0, 0, 0]), 1.0)
        self.assertEqual(float(grid[0, EGO_RADIUS, EGO_RADIUS]), 0.0)

    def test_classes_stay_separate(self):
        from isaac_ai.env import EGO_RADIUS
        width, height = 15, 9
        cells = [0] * (width * height)
        cells[4 * width + 8] = 2                          # hazard right of player
        flat = self.room(cells, width, *self.at_tile(4, 7))
        grid = self.window(flat)
        self.assertEqual(float(grid[1, EGO_RADIUS, EGO_RADIUS + 1]), 1.0)
        self.assertEqual(float(grid[0, EGO_RADIUS, EGO_RADIUS + 1]), 0.0)

    def test_unreadable_grid_is_empty_not_sealed(self):
        """A broken payload must not look like a room with no exits."""
        flat = self.room([], 0, *self.at_tile(4, 7))
        self.assertEqual(float(flat.sum()), 0.0)


class TestEpisodeEndReasons(unittest.TestCase):
    """"Episode over" covers three unrelated failures; the summary must split them.

    Dying is a combat problem, the step cap is a pacing problem, and the idle
    limit is usually the agent pacing between rooms it has already emptied —
    where `door_potential` is flat by construction and nothing tells it which
    way to walk. Aggregated wrong, all three read as one number.
    """

    def summarize(self, extras):
        from isaac_ai.ppo import PPOConfig, PPOTrainer

        trainer = PPOTrainer.__new__(PPOTrainer)
        trainer._episode_extras = extras
        return PPOTrainer._episode_summary(trainer)

    def episode(self, reason, length=100, stranded=False, rooms_seen=1,
                enemies_alive=0):
        return {"r": 0.0, "l": length, "success": False, "rooms_cleared": 0,
                "rooms_seen": rooms_seen, "transitions": 1,
                "backtrack_ratio": 1.0, "descended": 0,
                "reason": reason, "stranded": stranded,
                "enemies_alive": enemies_alive}

    def test_reasons_are_reported_as_shares(self):
        summary = self.summarize([self.episode("died"), self.episode("died"),
                                  self.episode("idle"), self.episode("timeout")])
        self.assertAlmostEqual(summary["ended_died"], 0.5)
        self.assertAlmostEqual(summary["ended_idle"], 0.25)
        self.assertAlmostEqual(summary["ended_timeout"], 0.25)

    def test_idle_length_is_measured_over_idle_episodes_only(self):
        """The throughput number: how much game time a stall actually burns."""
        summary = self.summarize([self.episode("died", length=50),
                                  self.episode("idle", length=500),
                                  self.episode("idle", length=520)])
        self.assertEqual(summary["idle_episode_steps"], 510)
        self.assertEqual(summary["episode_steps"], 356)

    def test_stranded_counts_only_episodes_that_gave_up(self):
        """A death in an exhausted room is not evidence about navigation."""
        summary = self.summarize([
            self.episode("died", stranded=True),      # must not count
            self.episode("idle", stranded=True),
            self.episode("timeout", stranded=False),
        ])
        self.assertAlmostEqual(summary["stranded"], 0.5)

    def test_every_reason_is_logged_so_shares_sum_to_one(self):
        """Omitting absent reasons makes cross-update averaging silently wrong.

        Averaging each key over only the updates where it appeared made the
        shares sum to 1.047 on floor-v9b. If the block exists at all then
        episodes ended, so 0.0 already means "measured, none of them".
        """
        summary = self.summarize([self.episode("died"), self.episode("idle")])
        total = sum(summary[f"ended_{n}"]
                    for n in ("died", "idle", "timeout", "dropped"))
        self.assertAlmostEqual(total, 1.0, places=6)
        self.assertEqual(summary["ended_timeout"], 0.0)

    def test_exhausted_excludes_a_stall_inside_a_locked_fight(self):
        """Doors lock during combat and door_potential skips locked doors.

        So `stranded` alone also counts an attempt that stalled mid-fight, which
        is a combat problem, not the flat-potential navigation problem a
        non-local potential would fix.
        """
        summary = self.summarize([
            self.episode("idle", stranded=True, enemies_alive=3),
            self.episode("idle", stranded=True, enemies_alive=0),
        ])
        self.assertAlmostEqual(summary["stranded"], 1.0)
        self.assertAlmostEqual(summary["exhausted"], 0.5)


class TestEntitySemanticFlags(unittest.TestCase):
    """A chest, a coin and an item pedestal must stop being the same vector.

    The mod sent Type/Variant/SubType from the beginning and nothing read them,
    so every pickup encoded identically apart from position — including a spiked
    chest and a mimic, both of which damage you, against a normal chest.
    """

    def encode_entity(self, **flags):
        obs = {
            "ready": True,
            "player": {"x": 320.0, "y": 280.0, "vx": 0, "vy": 0, "hearts": 6,
                       "max_hearts": 6, "soul_hearts": 0, "bombs": 1, "keys": 0,
                       "coins": 0, "damage": 3.5, "speed": 1.0,
                       "tear_delay": 10, "range": 260, "can_fly": False,
                       "active_item": 0, "is_dead": False},
            "room": {"index": 84, "type": 1, "shape": 1, "clear": False,
                     "top_left_x": 60, "top_left_y": 140,
                     "bottom_right_x": 580, "bottom_right_y": 420,
                     "enemies_alive": 0, "doors": []},
            "level": {"stage": 1, "stage_type": 0, "curses": 0,
                      "rooms_total": 12, "rooms_visited": 4},
            "entities": [dict({"k": "pickup", "t": 5, "v": 50, "s": 0,
                               "x": 400.0, "y": 300.0, "vx": 0, "vy": 0,
                               "hp": 0, "mhp": 0, "boss": False, "d": 89.4},
                              **flags)],
            "events": {},
        }
        return encode_observation(obs)["entities"][0]

    def test_a_chest_and_a_coin_differ(self):
        chest = self.encode_entity(chest=True)
        coin = self.encode_entity(consumable=True)
        self.assertEqual(float(chest[ENTITY_CHEST]), 1.0)
        self.assertEqual(float(chest[ENTITY_CONSUMABLE]), 0.0)
        self.assertEqual(float(coin[ENTITY_CONSUMABLE]), 1.0)
        self.assertEqual(float(coin[ENTITY_CHEST]), 0.0)
        self.assertFalse(np.array_equal(chest, coin))

    def test_a_spiked_chest_is_distinguishable_from_a_normal_one(self):
        """The flag that actually earns its place."""
        normal = self.encode_entity(chest=True)
        spiked = self.encode_entity(chest=True, hostile=True)
        self.assertEqual(float(normal[ENTITY_HOSTILE]), 0.0)
        self.assertEqual(float(spiked[ENTITY_HOSTILE]), 1.0)
        # Still a chest: the flags are not mutually exclusive.
        self.assertEqual(float(spiked[ENTITY_CHEST]), 1.0)

    def test_pedestal_and_flying_are_separate(self):
        pedestal = self.encode_entity(pedestal=True)
        flyer = self.encode_entity(flying=True)
        self.assertEqual(float(pedestal[ENTITY_PEDESTAL]), 1.0)
        self.assertEqual(float(flyer[ENTITY_FLYING]), 1.0)
        self.assertEqual(float(pedestal[ENTITY_FLYING]), 0.0)

    def test_absent_flags_encode_as_zero_not_a_default(self):
        """An unmapped entity must look like nothing, not like something."""
        bare = self.encode_entity()
        for index in (ENTITY_CONSUMABLE, ENTITY_PEDESTAL, ENTITY_CHEST,
                      ENTITY_HOSTILE, ENTITY_FLYING):
            self.assertEqual(float(bare[index]), 0.0)

    def test_flags_are_appended_after_the_kinematic_features(self):
        """`_warm_start` transfers a widened layer by zeroing the new columns.

        That is exact only while the new features sit at the end. If one is ever
        inserted earlier, every later column shifts and a resumed policy quietly
        computes something else — no error, just a different network.
        """
        self.assertTrue(ENTITY_CONSUMABLE > ENTITY_TANGENTIAL > ENTITY_CLOSING)
        self.assertEqual(ENTITY_FLYING, ENTITY_FEATURES - 1)
        # The kinematic block must keep its meaning at its original offsets.
        moving = self.encode_entity()
        self.assertEqual(len(moving), ENTITY_FEATURES)


class TestMinActionProbability(unittest.TestCase):
    """Entropy cannot see one action inside a head die.

    floor-v13 answered an up-door state with move_y = [0.01 up, 0.61 still,
    0.37 down]. Entropy 0.715, above the 0.5 target, so the clamped bonus gave
    **zero** gradient and nothing pushed P(up) back up — while the agent could
    not leave a room whose only exit was upward and the logged per-axis entropy
    read a healthy 0.694 the whole time.
    """

    def shortfall(self, probs, floor):
        from isaac_ai.ppo import PPOConfig

        config = PPOConfig(min_action_prob=floor)
        tensor = torch.tensor([probs], requires_grad=True)
        if config.min_action_prob <= 0:
            return 0.0, torch.zeros(1)
        loss = torch.relu(config.min_action_prob - tensor).sum(-1).mean()
        loss.backward()
        return float(loss), tensor.grad[0]

    def test_the_dead_action_state_is_penalised(self):
        """The exact distribution that stranded floor-v13."""
        loss, grad = self.shortfall([0.01, 0.61, 0.37], floor=0.05)
        self.assertAlmostEqual(loss, 0.04, places=6)
        self.assertLess(float(grad[0]), 0.0, "P(up) must be pushed up")
        self.assertEqual(float(grad[1]), 0.0, "healthy actions untouched")

    def test_entropy_alone_would_have_said_it_was_fine(self):
        """Why the floor is needed at all, stated as an assertion."""
        probs = torch.tensor([0.01, 0.61, 0.37])
        entropy = float(-(probs * probs.log()).sum())
        self.assertGreater(entropy, 0.5,
                           "this is above the entropy target, so the clamped "
                           "bonus contributes nothing and P(up) is unprotected")
        self.assertLess(float(probs.min()), 0.02)

    def test_a_confident_but_alive_head_is_left_alone(self):
        """The policy must stay free to commit where it should."""
        loss, _ = self.shortfall([0.06, 0.88, 0.06], floor=0.05)
        self.assertEqual(loss, 0.0)

    def test_floor_bounds_entropy_from_below(self):
        """So it subsumes most of what entropy_target was doing."""
        probs = torch.tensor([0.05, 0.05, 0.90])
        self.assertGreater(float(-(probs * probs.log()).sum()), 0.35)

    def test_default_is_off(self):
        from isaac_ai.ppo import PPOConfig

        self.assertEqual(PPOConfig().min_action_prob, 0.0)


class TestEntropyTarget(unittest.TestCase):
    """A head at the ceiling must stop being pulled towards uniform.

    The bonus is applied to the summed entropy of all four heads, so a head
    receiving no learning gradient is pushed to ln(3)=1.099 at full strength
    forever — the documented cause of every dead axis here. With a target set,
    a head at or above it contributes a constant and therefore no gradient.
    """

    def _entropy_grad(self, logits_value, target):
        """d(entropy objective)/d(logits) for one head at a chosen sharpness."""
        from isaac_ai.ppo import PPOConfig

        config = PPOConfig(entropy_target=target)
        logits = torch.tensor([[logits_value, 0.0, 0.0]], requires_grad=True)
        probs = torch.softmax(logits, dim=-1)
        head = -(probs * torch.log(probs + 1e-9)).sum(-1)
        if config.entropy_target > 0:
            objective = head.clamp(max=config.entropy_target).sum(0).mean()
        else:
            objective = head.mean()
        objective.backward()
        return float(logits.grad.abs().sum()), float(head)

    def test_uniform_head_gets_no_gradient_under_target(self):
        grad, entropy = self._entropy_grad(0.0, target=0.5)
        self.assertAlmostEqual(entropy, math.log(3), places=4)
        self.assertAlmostEqual(grad, 0.0, places=8)

    def test_uniform_head_is_pushed_without_a_target(self):
        """Without a target the same head is still being acted on."""
        # At exactly uniform the gradient is zero by symmetry, so perturb it:
        # slightly-committed is where the pull back to uniform is visible.
        grad, entropy = self._entropy_grad(0.8, target=0.0)
        self.assertLess(entropy, math.log(3))
        self.assertGreater(grad, 1e-3)

    def test_collapsed_head_still_gets_pushed_back_up(self):
        """The target is a floor, not an off switch."""
        grad, entropy = self._entropy_grad(6.0, target=0.5)
        self.assertLess(entropy, 0.5)
        self.assertGreater(grad, 1e-3)

    def test_default_is_off(self):
        from isaac_ai.ppo import PPOConfig

        self.assertEqual(PPOConfig().entropy_target, 0.0)


class TestBridgeFailureIsRecoverable(unittest.TestCase):
    """Every way an instance can die must surface as BridgeError.

    `_receive_all` catches BridgeError and marks the instance failed so the run
    continues on the survivors. Anything else propagates out of the trainer and
    takes the whole fleet down with it, which is what happened to floor-v7 at
    57% of 1M steps: one isaac-ng.exe hit an access violation, the kernel sent
    an RST, and `recv_into` raised ConnectionResetError — an OSError, not a
    BridgeError — straight past the handler.

    A clean shutdown was always handled, which is exactly why this survived
    thirty-odd runs: closing normally ends the stream and readline returns b"".
    """

    def _bridge_and_peer(self, port: int):
        import socket

        from isaac_ai.bridge import InstanceBridge

        bridge = InstanceBridge(port=port, index=0)
        peer = socket.create_connection(("127.0.0.1", port))
        peer.sendall(b'{"t":"hello"}\n')
        self.assertTrue(bridge.poll_accept(5.0))
        return bridge, peer

    def test_connection_reset_becomes_bridge_error(self):
        """A crashed game drops the socket with an RST, not a FIN."""
        import socket
        import struct

        from isaac_ai.bridge import BridgeError

        bridge, peer = self._bridge_and_peer(9897)
        try:
            # SO_LINGER 0 makes close() send RST — what the kernel does for a
            # process that died rather than exited.
            peer.setsockopt(socket.SOL_SOCKET, socket.SO_LINGER,
                            struct.pack("ii", 1, 0))
            peer.close()
            with self.assertRaises(BridgeError):
                bridge.receive()
        finally:
            bridge.close()

    def test_clean_close_becomes_bridge_error(self):
        """The orderly path has to keep working too."""
        from isaac_ai.bridge import BridgeError

        bridge, peer = self._bridge_and_peer(9898)
        try:
            peer.close()
            with self.assertRaises(BridgeError):
                bridge.receive()
        finally:
            bridge.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
