"""Which part of the observation is the policy actually listening to?

floor-v22 tested whether the door *reward* was drowning combat and the answer was
no: gating door shaping off during fights changed `ended_idle` by 0.007 and left
the slope of `rooms_cleared` identical. What it did show is that removing doors
from the **observation** changes behaviour a great deal — v21 fired correctly
28/32 with the door block blanked against 12/32 with it present — and a shaping
change cannot reach that.

So this measures input attribution directly, with no game and no training run.

Two independent methods, because either alone is arguable:

  swap    take two realistic states that differ in enemy position *and* in which
          door is unvisited, then rebuild state A with exactly one input block
          replaced by B's. The total variation in the action distribution is
          that block's influence, measured with values the encoder actually
          produces rather than synthetic noise.

  jacobian  norm of d(logits)/d(block), reported per input and scaled by the
          block's own magnitude, so a 104-value door block and a 704-value
          entity block are comparable.

They answer slightly different questions -- swap is "how much does this fact
matter", jacobian is "how hard is the policy leaning on these numbers" -- and a
disagreement between them is itself informative, so both are printed.

`MAX_ENTITIES` is 32 slots of 22 features but a real room fills only a few, so
the entity block is reported both raw and per *occupied* slot. Comparing a
mostly-empty 704-value block against a mostly-full 104-value one without saying
so is how you manufacture the conclusion you wanted.

    .venv/Scripts/python.exe scripts/probe_input_sensitivity.py runs/floor-v22/policy.pt
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from diagnose_shoot_axis import observation  # noqa: E402
from isaac_ai.env import encode_observation  # noqa: E402
from isaac_ai.policy import ActorCritic, to_tensors  # noqa: E402

BLOCKS = ("entities", "doors", "scalars", "grid", "ego_grid")
HEADS = ("move_x", "move_y", "shoot_x", "shoot_y")


def with_doors(raw: dict, unvisited_slot: int) -> dict:
    """Same room, but a different door is the unexplored one."""
    out = copy.deepcopy(raw)
    for door in out["room"]["doors"]:
        door["visited"] = 0 if door.get("slot") == unvisited_slot else 3
    return out


def tensors(raw: dict) -> dict:
    encoded = encode_observation(raw)
    return {k: v[None] for k, v in
            to_tensors(encoded, torch.device("cpu")).items()}


def distribution(net: ActorCritic, obs: dict) -> list[np.ndarray]:
    with torch.no_grad():
        logits, _ = net(obs)
    return [torch.softmax(head[0], -1).numpy() for head in logits]


def total_variation(a: list[np.ndarray], b: list[np.ndarray]) -> float:
    return float(np.mean([0.5 * np.abs(x - y).sum() for x, y in zip(a, b)]))


def main() -> None:
    sys.stdout.reconfigure(line_buffering=True)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    args = parser.parse_args()

    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    net = ActorCritic()
    net.load_state_dict(checkpoint["policy"])
    net.eval()
    print(f"{args.checkpoint}  ({checkpoint.get('global_step', 0):,} steps)\n")

    slots = sorted({d.get("slot") for d in observation(220, 0)["room"]["doors"]})
    # A and B differ in both things at once, so every block has something to
    # contribute and none is being fed an identity swap.
    raw_a = with_doors(observation(220, 0), slots[0])
    raw_b = with_doors(observation(-220, 0), slots[-1])
    a, b = tensors(raw_a), tensors(raw_b)

    base = distribution(net, a)
    both = total_variation(base, distribution(net, b))

    occupied = int(a["entity_mask"][0].sum().item())
    print(f"  state A: enemy right, door slot {slots[0]} unvisited")
    print(f"  state B: enemy left,  door slot {slots[-1]} unvisited")
    print(f"  {occupied} entity slot(s) occupied of {a['entities'].shape[1]}")
    print(f"  swapping everything at once: TV {both:.4f}\n")

    print(f"  {'block':<11}{'values':>8}{'swap TV':>10}{'share':>8}"
          f"{'jacobian':>11}{'per value':>11}")

    swaps: dict[str, float] = {}
    for name in BLOCKS:
        hybrid = dict(a)
        hybrid[name] = b[name]
        if name == "entities":
            # The mask selects which slots are read at all; swapping the values
            # without it would leave B's enemy in a slot A says is empty.
            hybrid["entity_mask"] = b["entity_mask"]
            hybrid["entity_types"] = b.get("entity_types", a.get("entity_types"))
        swaps[name] = total_variation(base, distribution(net, hybrid))

    jac: dict[str, float] = {}
    for name in BLOCKS:
        inputs = {k: v.clone() for k, v in a.items()}
        inputs[name].requires_grad_(True)
        logits, _ = net(inputs)
        # Sum of squared logits is a scale-free stand-in for "how much the head
        # outputs move"; its gradient norm is the sensitivity to this block.
        objective = sum((head ** 2).sum() for head in logits)
        objective.backward()
        grad = inputs[name].grad
        jac[name] = float(grad.norm().item()) if grad is not None else 0.0

    total_swap = sum(swaps.values()) or 1.0
    for name in BLOCKS:
        size = int(np.prod(a[name].shape[1:]))
        print(f"  {name:<11}{size:>8}{swaps[name]:>10.4f}"
              f"{swaps[name] / total_swap:>7.0%}"
              f"{jac[name]:>11.3f}{jac[name] / size:>11.5f}")

    ent_size = int(np.prod(a["entities"].shape[1:]))
    per_slot = ent_size // a["entities"].shape[1]
    print(f"\n  entity block is {ent_size} values but only {occupied} slot(s) "
          f"({occupied * per_slot} values) carry anything;")
    print(f"  per *occupied* value that is jacobian "
          f"{jac['entities'] / max(occupied * per_slot, 1):.5f}, against doors' "
          f"{jac['doors'] / int(np.prod(a['doors'].shape[1:])):.5f}")

    print(f"\n  {'head':<10}{'doors swapped':>16}{'entities swapped':>18}")
    for index, head in enumerate(HEADS):
        d = dict(a)
        d["doors"] = b["doors"]
        e = dict(a)
        e["entities"], e["entity_mask"] = b["entities"], b["entity_mask"]
        if "entity_types" in b:
            e["entity_types"] = b["entity_types"]
        dd = distribution(net, d)[index]
        ee = distribution(net, e)[index]
        print(f"  {head:<10}{0.5 * np.abs(dd - base[index]).sum():>16.4f}"
              f"{0.5 * np.abs(ee - base[index]).sum():>18.4f}")

    print("\n  reading: `shoot_x` and `shoot_y` should move more when the enemy "
          "moves than\n  when a door does. If they do not, the shoot heads are "
          "keyed to navigation\n  state, and no reward change reaches that — it "
          "is an encoder problem.")


if __name__ == "__main__":
    main()
