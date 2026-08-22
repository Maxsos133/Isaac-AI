"""Typed configuration loaded from config.toml."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class GameConfig:
    executable: Path
    mods_dir: Path
    savedata_dir: Path
    app_id: int


@dataclass(frozen=True)
class InstancesConfig:
    count: int
    base_port: int
    window_width: int
    window_height: int
    grid_columns: int
    origin_x: int
    origin_y: int

    def port_for(self, index: int) -> int:
        return self.base_port + index

    def window_rect(self, index: int) -> tuple[int, int, int, int]:
        column = index % self.grid_columns
        row = index // self.grid_columns
        x = self.origin_x + column * self.window_width
        y = self.origin_y + row * self.window_height
        return x, y, self.window_width, self.window_height


@dataclass(frozen=True)
class SaveConfig:
    snapshot: Path


@dataclass(frozen=True)
class EnvConfig:
    action_repeat: int
    max_episode_steps: int
    startup_timeout_seconds: float


@dataclass(frozen=True)
class RewardConfig:
    damage_dealt: float
    damage_taken: float
    kill: float
    room_clear: float
    new_level: float
    death: float
    step: float
    new_room: float = 0.0
    door_shaping: float = 0.0
    navigation_arrival: float = 10.0
    # Combat overrides `step`. The shared value has to stay small enough for
    # door shaping to dominate it in the floor and navigation tasks; combat has
    # no shaping term, so it can price idling on its own terms.
    combat_step: float = -0.005
    # Floors override `death`, and only floors actually charge it: combat counts
    # deaths but still pays through `compute_reward`'s `events["died"]`, which
    # the mod almost never delivers. Kept separate so tuning the floor penalty
    # cannot silently redefine the environment every combat run was measured in.
    floor_death: float = -10.00
    # Floors only: charged when the agent asks to move and does not travel.
    # Measured on floor-v12 against a random-action control on the same fleet:
    # the trained policy was pinned against geometry on **18.6%** of the steps
    # it tried to move, against **2.5%** for a random walk — roughly seven times
    # as often. A blocked step costs the -0.002 step penalty and nothing else,
    # because no position change means no potential change, so walking into a
    # wall and walking uselessly in the open pay exactly the same. That is why
    # the obstacle grid and then the egocentric version of it both went unused:
    # the information was there and nothing made it worth reading.
    # 0.0 disables it.
    blocked_move: float = 0.0


@dataclass(frozen=True)
class CombatConfig:
    max_enemies: int = 10


@dataclass(frozen=True)
class PixelConfig:
    """The student's observation. Changing any of it invalidates a trained one."""

    width: int = 160
    height: int = 90
    stack: int = 4
    grayscale: bool = False

    def check_divides(self, client_width: int, client_height: int) -> None:
        """Refuse a size the client area does not divide exactly.

        An inexact ratio makes the downsample a resampling filter rather than a
        box filter, which puts scale-dependent artefacts into every frame and
        quietly changes the aspect. Window geometry is chosen so this holds;
        failing loudly here is what keeps it true.
        """
        if client_width % self.width or client_height % self.height:
            raise ValueError(
                f"pixel input {self.width}x{self.height} does not divide the "
                f"{client_width}x{client_height} client area exactly — adjust "
                f"[pixels] or the window size in [instances]")


@dataclass(frozen=True)
class AppConfig:
    root: Path
    game: GameConfig
    instances: InstancesConfig
    save: SaveConfig
    env: EnvConfig
    rewards: RewardConfig
    combat: CombatConfig
    pixels: PixelConfig


def _resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (root / path)


def load_config(path: str | Path | None = None) -> AppConfig:
    root = Path(__file__).resolve().parents[2]
    config_path = Path(path) if path else (root / "config.toml")
    with config_path.open("rb") as handle:
        raw = tomllib.load(handle)

    game_raw = raw["game"]
    save_raw = raw["save"]

    return AppConfig(
        root=root,
        game=GameConfig(
            executable=Path(game_raw["executable"]),
            mods_dir=Path(game_raw["mods_dir"]),
            savedata_dir=Path(game_raw["savedata_dir"]),
            app_id=int(game_raw["app_id"]),
        ),
        instances=InstancesConfig(**raw["instances"]),
        save=SaveConfig(snapshot=_resolve(root, save_raw["snapshot"])),
        env=EnvConfig(**raw["env"]),
        rewards=RewardConfig(**raw["rewards"]),
        combat=CombatConfig(**raw.get("combat", {})),
        pixels=PixelConfig(**raw.get("pixels", {})),
    )
