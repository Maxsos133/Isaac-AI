"""Command line entry points."""

from __future__ import annotations

import argparse
import time
from dataclasses import replace

import numpy as np

from isaac_ai import launcher
from isaac_ai.config import load_config
from isaac_ai.env import ACTION_DIMS, IsaacVecEnv


def cmd_smoke(args: argparse.Namespace) -> int:
    """Bring up a fleet, drive it with random actions, and report throughput.

    This is the Phase 1 exit criterion: N instances stepping in lockstep
    through a real environment loop with episode resets.
    """
    config = load_config()
    # Short episodes so the smoke test actually exercises the reset path;
    # a real run in an empty start room would never terminate on its own.
    config = replace(config, env=replace(config.env,
                                         max_episode_steps=args.episode_steps))
    fleet = launcher.bring_up(config, count=args.instances)
    try:
        return _run_smoke(fleet, config, args)
    finally:
        fleet.shutdown()


def _run_smoke(fleet: launcher.Fleet, config, args: argparse.Namespace) -> int:
    if fleet.ready_count == 0:
        print("no instances reached a run")
        return 1

    env = IsaacVecEnv(fleet.bridges(), config)
    print(f"\nresetting {env.num_envs} instance(s)...")
    env.reset()

    rng = np.random.default_rng(0)
    started = time.perf_counter()
    episodes = 0
    total_return = 0.0

    print(f"stepping for {args.steps} agent steps...\n")
    for step in range(args.steps):
        actions = np.stack([
            rng.integers(0, dim, size=env.num_envs) for dim in ACTION_DIMS
        ], axis=1)
        _, rewards, terminated, truncated, infos = env.step(actions)

        done = terminated | truncated
        for info in infos:
            if "episode" in info:
                episodes += 1
                total_return += info["episode"]["r"]
        if done.any():
            env.reset_done(done)

        if (step + 1) % 100 == 0:
            elapsed = time.perf_counter() - started
            ticks = (step + 1) * env.num_envs * env.action_repeat
            print(f"  step {step + 1:5d} | {ticks / elapsed:7.1f} game ticks/s "
                  f"| {(step + 1) * env.num_envs / elapsed:6.1f} agent steps/s "
                  f"| episodes {episodes} | alive {env.alive_count}")

    elapsed = time.perf_counter() - started
    agent_steps = args.steps * env.num_envs
    print(f"\n{agent_steps} agent steps in {elapsed:.1f}s")
    print(f"  {agent_steps / elapsed:.1f} agent steps/s")
    print(f"  {agent_steps * env.action_repeat / elapsed:.1f} game ticks/s")
    print(f"  projected: {agent_steps / elapsed * 3600:,.0f} agent steps/hour")
    if episodes:
        print(f"  {episodes} episodes, mean return {total_return / episodes:.2f}")
    return 0


def cmd_train(args: argparse.Namespace) -> int:
    """Train the state-based combat teacher."""
    import torch

    from isaac_ai.combat import CombatVecEnv
    from isaac_ai.curriculum import CombatCurriculum
    from isaac_ai.ppo import PPOConfig, PPOTrainer

    config = load_config()
    fleet = launcher.bring_up(config, count=args.instances)
    try:
        if fleet.ready_count == 0:
            print("no instances reached a run")
            return 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        curriculum = CombatCurriculum(seed=args.seed)
        env = CombatVecEnv(fleet.bridges(), config, curriculum,
                           max_encounter_steps=args.encounter_steps)

        run_dir = config.root / "runs" / args.run_name
        # `entropy_target` and `min_action_prob` were on `train-floor` only, so
        # every combat run this project ever trained was trained without them.
        # That is very likely why combat-v5 through v7 could not shoot
        # horizontally at all and it went unnoticed for three runs: the entropy
        # bonus applies to the *summed* entropy of four heads, so a head with no
        # learning gradient is pushed to the ln(3)=1.099 uniform ceiling at full
        # strength forever, and the summed metric cannot show it. Adding the
        # ceiling on floors took `shoot_x` from 1.084 to 0.712 and produced the
        # first policy here with no dead axes.
        trainer = PPOTrainer(env,
                             PPOConfig(rollout_steps=args.rollout_steps,
                                       entropy_target=args.entropy_target,
                                       min_action_prob=args.min_action_prob),
                             device, run_dir)

        if args.resume:
            _warm_start(trainer, config, args.resume, device)
            # combat-v5 rose monotonically for a million steps and never
            # plateaued, so the useful thing to do with a teacher is usually to
            # continue it rather than start again. Difficulty is not stored in
            # the checkpoint, so it has to be handed back explicitly or the
            # curriculum re-climbs from zero and wastes the head start.
            curriculum.difficulty = args.start_difficulty
            print(f"resuming curriculum at difficulty {args.start_difficulty}")

        print(f"\ntraining on {env.num_envs} instance(s), device={device}")
        print(f"run directory: {run_dir}\n")
        trainer.train(total_steps=args.steps)

        print(f"\nsaved policy to {trainer.save()}")
        return 0
    finally:
        fleet.shutdown()


def _warm_start(trainer, config, resume: str, device) -> None:
    """Behaviour-preserving transfer from an earlier checkpoint.

    Copying only same-shape tensors is not enough. When an extra encoder is
    concatenated on, the trunk's first layer grows, and reinitialising it severs
    the path from the transferred features to the action heads — so the policy
    starts random despite most tensors having "transferred". floor-v1 did
    exactly that and began at entropy 4.25 instead of the ~2.0 it had learned.

    Widened layers instead keep their old weights in the leading columns and get
    zeros for the new inputs, so at initialisation the network computes exactly
    what the checkpoint did and the new features earn influence by gradient.
    This relies on new features having been *appended*, never inserted.
    """
    import torch

    checkpoint = config.root / resume
    source = torch.load(checkpoint, map_location=device)["policy"]
    target = trainer.policy.state_dict()

    exact, widened, fresh = [], [], []
    for name, new_tensor in target.items():
        old = source.get(name)
        if old is None:
            fresh.append(name)
            continue
        if old.shape == new_tensor.shape:
            new_tensor.copy_(old)
            exact.append(name)
        elif (old.dim() == new_tensor.dim() == 2
              and old.shape[0] == new_tensor.shape[0]
              and old.shape[1] < new_tensor.shape[1]):
            new_tensor.zero_()
            new_tensor[:, :old.shape[1]].copy_(old)
            widened.append(f"{name} ({old.shape[1]}->{new_tensor.shape[1]})")
        else:
            fresh.append(name)

    trainer.policy.load_state_dict(target)
    print(f"warm start from {checkpoint.name}: {len(exact)} copied, "
          f"{len(widened)} widened, {len(fresh)} fresh")
    for name in widened:
        print(f"  widened (new inputs zeroed): {name}")
    for name in fresh:
        print(f"  freshly initialised: {name}")


def cmd_train_floor(args: argparse.Namespace) -> int:
    """Train floor progression: real rooms, one health pool, find the way on."""
    import torch

    from isaac_ai.floor_curriculum import FloorCurriculum
    from isaac_ai.floors import FloorVecEnv
    from isaac_ai.ppo import PPOConfig, PPOTrainer

    config = load_config()
    fleet = launcher.bring_up(config, count=args.instances)
    try:
        if fleet.ready_count == 0:
            print("no instances reached a run")
            return 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        curriculum = FloorCurriculum(difficulty=args.start_difficulty)
        env = FloorVecEnv(fleet.bridges(), config, curriculum,
                          max_floor_steps=args.floor_steps)

        run_dir = config.root / "runs" / args.run_name
        trainer = PPOTrainer(env,
                             PPOConfig(rollout_steps=args.rollout_steps,
                                       entropy_target=args.entropy_target,
                                       min_action_prob=args.min_action_prob),
                             device, run_dir,
                             recover=(_make_recover(config, fleet, env)
                                      if args.relaunch_crashed else None))

        if args.resume:
            _warm_start(trainer, config, args.resume, device)

        print(f"\ntraining on {env.num_envs} instance(s), device={device}")
        print(f"run directory: {run_dir}\n")
        trainer.train(total_steps=args.steps)
        print(f"\nsaved policy to {trainer.save()}")
        return 0
    finally:
        fleet.shutdown()


def _make_recover(config, fleet, env):
    """A closure the trainer can call to bring crashed instances back.

    Passed as a callback rather than handing `PPOTrainer` the fleet, so nothing
    in the learner has to know about processes, windows or SPACE taps.

    Off by default. The failure it fixes costs one instance an hour; the failure
    it could introduce is a wedged fleet, because run entry needs the foreground
    and a blocked Isaac stops pumping Win32 messages. That trade is only worth
    taking deliberately, and only once it has been watched working.
    """
    def recover() -> int:
        restored = 0
        for index in env.failed_indices():
            bridge = env.bridges[index]
            if launcher.relaunch(config, fleet, bridge,
                                 keep_alive=env.healthy_bridges(exclude=index)):
                if env.restore(index):
                    restored += 1
        if restored:
            print(f"  fleet back to {env.alive_count}/{env.num_envs} instances")
        return restored

    return recover


def cmd_distill(args: argparse.Namespace) -> int:
    """Distil a state-based teacher into the pixels-only student."""
    import torch

    from isaac_ai.capture import FleetCapture
    from isaac_ai.combat import CombatVecEnv
    from isaac_ai.curriculum import CombatCurriculum
    from isaac_ai.distill import Distiller, DistillConfig, load_teacher

    config = load_config()
    # Fail on a teacher that cannot be loaded before spending several minutes
    # bringing a fleet up.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    teacher = load_teacher(config.root / args.teacher, device)

    fleet = launcher.bring_up(config, count=args.instances)
    capture = None
    try:
        if fleet.ready_count == 0:
            print("no instances reached a run")
            return 1

        curriculum = CombatCurriculum(max_enemies=config.combat.max_enemies)
        # Start where the teacher is already competent rather than at zero.
        # The ramp only steps once per batch of fresh teacher episodes, and at
        # a 0.9 student share those arrive slowly enough that a whole run has
        # roughly 47 steps of travel in it — climbing from 0 to the teacher's
        # level would spend 40% of that budget before the demonstrations are
        # even worth imitating.
        curriculum.difficulty = args.start_difficulty
        env = CombatVecEnv(fleet.bridges(), config, curriculum)
        capture = FleetCapture(
            [i.hwnd for i in fleet.instances if i.ready and not i.failed],
            config.pixels)

        run_dir = config.root / "runs" / args.run_name
        distiller = Distiller(
            env, capture, teacher,
            DistillConfig(rollout_steps=args.rollout_steps,
                          student_share_end=args.student_share,
                          student_share_steps=args.student_share_steps,
                          student_greedy=args.student_greedy),
            device, run_dir)

        print(f"\nteacher: {args.teacher}")
        print(f"student sees {capture.shape} "
              f"({config.pixels.stack} x "
              f"{'grey' if config.pixels.grayscale else 'RGB'} "
              f"{config.pixels.width}x{config.pixels.height})")
        print(f"distilling on {env.num_envs} instance(s), device={device}")
        print(f"run directory: {run_dir}\n")
        distiller.train(total_steps=args.steps)
        print(f"\nsaved student to {distiller.save()}")
        return 0
    finally:
        if capture is not None:
            capture.stop()
        fleet.shutdown()


def cmd_finetune(args: argparse.Namespace) -> int:
    """Push the pixel student past its teacher with RL and a privileged critic."""
    import torch

    from isaac_ai.capture import FleetCapture
    from isaac_ai.combat import CombatVecEnv
    from isaac_ai.curriculum import CombatCurriculum
    from isaac_ai.distill import load_teacher
    from isaac_ai.pixel_ppo import PixelPPOConfig, PixelPPOTrainer, load_student

    config = load_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Both before bring-up: a bad path here should not cost several minutes.
    student = load_student(config.root / args.student, device)
    teacher = load_teacher(config.root / args.teacher, device)

    fleet = launcher.bring_up(config, count=args.instances)
    capture = None
    try:
        if fleet.ready_count == 0:
            print("no instances reached a run")
            return 1

        curriculum = CombatCurriculum(max_enemies=config.combat.max_enemies)
        curriculum.difficulty = args.start_difficulty
        env = CombatVecEnv(fleet.bridges(), config, curriculum)
        capture = FleetCapture(
            [i.hwnd for i in fleet.instances if i.ready and not i.failed],
            config.pixels)

        run_dir = config.root / "runs" / args.run_name
        trainer = PixelPPOTrainer(
            env, capture, student, teacher,
            PixelPPOConfig(rollout_steps=args.rollout_steps,
                           teacher_kl_coef=args.teacher_kl,
                           critic_warmup_updates=args.critic_warmup),
            device, run_dir)

        print(f"\nstudent: {args.student}")
        print(f"teacher (anchor only): {args.teacher}")
        print(f"critic warm-up: {args.critic_warmup} updates before the actor moves")
        print(f"fine-tuning on {env.num_envs} instance(s), device={device}")
        print(f"run directory: {run_dir}\n")
        trainer.train(total_steps=args.steps)
        print(f"\nsaved student to {trainer.save()}")
        return 0
    finally:
        if capture is not None:
            capture.stop()
        fleet.shutdown()


def cmd_train_nav(args: argparse.Namespace) -> int:
    """Train navigation in isolation: walk through a door."""
    import torch

    from isaac_ai.nav_curriculum import NavigationCurriculum
    from isaac_ai.navigation import NavigationVecEnv
    from isaac_ai.ppo import PPOConfig, PPOTrainer

    config = load_config()
    fleet = launcher.bring_up(config, count=args.instances)
    try:
        if fleet.ready_count == 0:
            print("no instances reached a run")
            return 1

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        curriculum = NavigationCurriculum()
        env = NavigationVecEnv(fleet.bridges(), config, curriculum,
                               max_nav_steps=args.nav_steps)

        run_dir = config.root / "runs" / args.run_name
        trainer = PPOTrainer(
            env,
            PPOConfig(rollout_steps=args.rollout_steps,
                      entropy_coef=args.entropy_coef),
            device, run_dir)
        if args.resume:
            _warm_start(trainer, config, args.resume, device)

        print(f"\ntraining on {env.num_envs} instance(s), device={device}")
        print(f"run directory: {run_dir}\n")
        trainer.train(total_steps=args.steps)
        print(f"\nsaved policy to {trainer.save()}")
        return 0
    finally:
        fleet.shutdown()


def cmd_seed(args: argparse.Namespace) -> int:
    config = load_config()
    launcher.seed_save(config)
    print(f"seeded savedata from {config.save.snapshot}")
    return 0


def cmd_deploy(args: argparse.Namespace) -> int:
    config = load_config()
    target = launcher.deploy_mod(config)
    print(f"deployed mod to {target}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(prog="isaac_ai")
    sub = parser.add_subparsers(dest="command", required=True)

    smoke = sub.add_parser("smoke", help="bring up a fleet and step it randomly")
    smoke.add_argument("--instances", type=int, default=None)
    smoke.add_argument("--steps", type=int, default=300)
    smoke.add_argument("--episode-steps", type=int, default=100,
                       help="truncate episodes early so resets get exercised")
    smoke.set_defaults(func=cmd_smoke)

    train = sub.add_parser("train", help="train the combat teacher with PPO")
    train.add_argument("--instances", type=int, default=None)
    train.add_argument("--steps", type=int, default=1_000_000)
    train.add_argument("--rollout-steps", type=int, default=128)
    train.add_argument("--encounter-steps", type=int, default=450)
    train.add_argument("--run-name", type=str, default="combat-v1")
    train.add_argument("--seed", type=int, default=0)
    train.add_argument("--resume", default=None,
                       help="continue from a checkpoint instead of starting "
                            "fresh, e.g. runs/combat-v5/policy.pt")
    train.add_argument("--start-difficulty", type=float, default=0.0,
                       help="curriculum difficulty to begin at; pair with "
                            "--resume or the ramp re-climbs from zero")
    # Same two flags `train-floor` has, and for a sharper reason here: combat is
    # where the dead-axis failure was first found and where it did the most
    # damage. combat-v5 through v7 could not shoot horizontally at all, for three
    # consecutive runs, because nothing stopped an ungradiented head being
    # dragged to uniform — and every combat teacher this project has was trained
    # before either of these existed.
    train.add_argument("--entropy-target", type=float, default=0.0,
                       help="per-head entropy ceiling in nats; above it a head "
                            "earns no bonus, so nothing pulls an ungradiented "
                            "head to the ln(3)=1.099 uniform ceiling. "
                            "0 = off (what every combat run so far used)")
    train.add_argument("--min-action-prob", type=float, default=0.0,
                       help="floor under each individual action's probability. "
                            "entropy-target only stops a whole head collapsing; "
                            "it cannot see one action inside a head die, which "
                            "is how floor-v13 ended up unable to move up at all "
                            "while its move_y entropy read a healthy 0.694")
    train.set_defaults(func=cmd_train)

    floor = sub.add_parser("train-floor", help="train floor progression with PPO")
    floor.add_argument("--instances", type=int, default=None)
    floor.add_argument("--steps", type=int, default=1_000_000)
    floor.add_argument("--rollout-steps", type=int, default=128)
    floor.add_argument("--floor-steps", type=int, default=3000,
                       help="hard cap on one floor attempt, in agent steps")
    floor.add_argument("--run-name", type=str, default="floor-v1")
    floor.add_argument("--resume", type=str, default=None,
                       help="checkpoint to warm-start from, e.g. runs/combat-v3/policy.pt")
    floor.add_argument("--min-action-prob", type=float, default=0.0,
                       help="floor under each individual action's probability. "
                            "entropy-target only stops a whole head collapsing; "
                            "it cannot see one action inside a head die, which "
                            "is how floor-v13 ended up unable to move up at all "
                            "while its move_y entropy read a healthy 0.694")
    floor.add_argument("--entropy-target", type=float, default=0.0,
                       help="per-head entropy ceiling in nats; above it a head "
                            "earns no bonus, so nothing pulls an ungradiented "
                            "head to the ln(3)=1.099 uniform ceiling. "
                            "0 = off (original behaviour)")
    # `train` has had this since the combat curriculum existed; `train-floor`
    # never got it, so the floor dial has re-climbed from 0.000 on every resume
    # in this lineage's history — v19b, v20, v21, v22, v23, v24, v25, v26.
    # floor-v25 reached 0.090 and v26 started again from zero. It has cost
    # little so far only because `target_rooms` maps anything under ~0.13 to 1,
    # so the task did not actually change; it starts mattering the moment the
    # dial is high enough to move the target.
    floor.add_argument("--relaunch-crashed", action="store_true",
                       help="bring instances back after a game crash instead of "
                            "finishing the run short. floor-v27b lost 2 of 20 "
                            "over 5M steps. Off by default: run entry needs the "
                            "foreground, and a bug there wedges the whole fleet "
                            "rather than costing one instance")
    floor.add_argument("--start-difficulty", type=float, default=0.0,
                       help="curriculum difficulty to begin at; pair with "
                            "--resume or the ramp re-climbs from zero. "
                            "Difficulty is not stored in the checkpoint")
    floor.set_defaults(func=cmd_train_floor)

    nav = sub.add_parser("train-nav", help="train navigation in isolation")
    nav.add_argument("--instances", type=int, default=None)
    nav.add_argument("--steps", type=int, default=400_000)
    nav.add_argument("--rollout-steps", type=int, default=128)
    nav.add_argument("--nav-steps", type=int, default=150,
                     help="cap on one traversal attempt, in agent steps")
    nav.add_argument("--run-name", type=str, default="nav-v1")
    nav.add_argument("--resume", type=str, default=None)
    nav.add_argument("--entropy-coef", type=float, default=0.002,
                     help="lower than combat's 0.01: in nav-v1 the entropy "
                          "bonus was ~10x the policy gradient and the policy "
                          "drifted to uniform")
    nav.set_defaults(func=cmd_train_nav)

    distill = sub.add_parser("distill",
                             help="distil a teacher into the pixel student")
    distill.add_argument("--instances", type=int, default=None)
    distill.add_argument("--steps", type=int, default=1_000_000)
    distill.add_argument("--rollout-steps", type=int, default=64)
    distill.add_argument("--run-name", default="distill-v1")
    distill.add_argument("--teacher", default="runs/combat-v4/policy.pt",
                         help="state-based policy to imitate")
    distill.add_argument("--student-share", type=float, default=0.9,
                         help="share of episodes the student ends up driving")
    distill.add_argument("--student-share-steps", type=int, default=300_000,
                         help="steps over which that share ramps up")
    distill.add_argument("--student-greedy", action="store_true",
                         help="let the student drive with its most likely "
                              "action instead of sampling; helps at low "
                              "difficulty but not at the range training runs in")
    distill.add_argument("--start-difficulty", type=float, default=0.55,
                         help="curriculum difficulty to begin at; default is "
                              "roughly where combat-v4 ended (0.57)")
    distill.set_defaults(func=cmd_distill)

    finetune = sub.add_parser("finetune",
                              help="RL fine-tune the pixel student past its teacher")
    finetune.add_argument("--instances", type=int, default=None)
    finetune.add_argument("--steps", type=int, default=1_000_000)
    finetune.add_argument("--rollout-steps", type=int, default=64)
    finetune.add_argument("--run-name", default="finetune-v1")
    finetune.add_argument("--student", default="runs/distill-v5/student.pt")
    finetune.add_argument("--teacher", default="runs/combat-v6/policy.pt",
                          help="used only as a KL anchor, never to act")
    finetune.add_argument("--teacher-kl", type=float, default=0.05,
                          help="pull back towards the teacher; 0 disables")
    finetune.add_argument("--critic-warmup", type=int, default=25,
                          help="updates training only the critic first")
    finetune.add_argument("--start-difficulty", type=float, default=0.42,
                          help="where every distilled student's curriculum has "
                               "settled, regardless of teacher or resolution")
    finetune.set_defaults(func=cmd_finetune)

    seed = sub.add_parser("seed", help="restore the pristine save snapshot")
    seed.set_defaults(func=cmd_seed)

    deploy = sub.add_parser("deploy", help="copy the bridge mod into the game")
    deploy.set_defaults(func=cmd_deploy)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
