"""Configuration for the car navigation environment.

One flat dataclass rather than nested ones, so overriding a single knob from a
training script stays a one-liner:

    cfg = CarNavConfig(image_size=64, num_targets=3)
"""

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Tuple

ASSETS = Path(__file__).resolve().parent.parent / "assets"


@dataclass
class CarNavConfig:
    # ---- map -------------------------------------------------------------
    map_path: str = str(ASSETS / "city_map.png")
    # A pixel counts as drivable road when every channel is bright AND the
    # channel spread is small, i.e. it is white-ish rather than a saturated
    # colour. Buildings (orange) and water (blue) both fail this test.
    road_min_channel: int = 200
    road_max_spread: int = 30

    # ---- car physics -----------------------------------------------------
    # 10x16 (T3D's original size) leaves 0.0% of the road drivable under
    # footprint collision - the whole rectangle needs 17px of clearance on a
    # map whose median road is ~14px wide. Shrunk to 10x6: half-diagonal
    # 5.8px -> 6px clearance -> 12.8% of road survives ("tight" but
    # workable, per the erosion table in README.md). Deliberately harder
    # than 8x5 (22.5%, "workable") - chosen when switching to footprint
    # collision on 2026-09-02; see notes/journal.md for the decision.
    car_length: float = 10.0            # along heading, pixels
    car_width: float = 6.0              # across heading, pixels
    max_steering_deg: float = 8.0       # per step, applied to heading
    min_speed: float = 1.5              # pixels per step
    max_speed: float = 2.5

    # ---- collision -------------------------------------------------------
    # "center"   : only the car's centre pixel must be on road (what T3D did).
    # "footprint": the car's whole rectangle must be on road. Stricter and
    #              more physically honest - the car no longer visibly clips
    #              buildings. Needs the smaller car size above to be viable
    #              at all on this map; see README.md's erosion table.
    collision_mode: str = "footprint"

    # ---- ray sensors -----------------------------------------------------
    # Kept for optional low-dim input. NOTE the default distance is 10, not
    # T3D's 20: median road width on city_map.png is ~14px, so 20px rays
    # overshoot the road and read "obstacle" even when the car is safely
    # centred (23% of on-road poses read 0/7). See README.
    use_sensors: bool = True
    sensor_angles_deg: Tuple[float, ...] = (-45, -30, -15, 0, 15, 30, 45)
    sensor_distance: float = 10.0

    # ---- observation -----------------------------------------------------
    use_image: bool = True
    image_size: int = 64                # output is image_size x image_size x 3
    crop_world_px: float = 96.0         # world extent the crop covers
    egocentric_rotation: bool = True    # rotate so the car always faces "up"
    draw_target_in_image: bool = True   # stamp the goal marker when in frame
    use_vector: bool = True             # dict obs gets a "vector" entry too

    # ---- targets ---------------------------------------------------------
    num_targets: int = 1
    target_radius: float = 20.0         # distance at which a target counts
    target_min_dist: float = 120.0      # sampled range from the car at reset
    target_max_dist: float = 300.0

    # ---- episode ---------------------------------------------------------
    max_episode_steps: int = 1000
    random_start: bool = True
    random_heading: bool = True
    # Minimum clear radius (px) required around a sampled spawn point, so the
    # car does not begin already clipping a building.
    spawn_clearance: float = 8.0

    # ---- reward ----------------------------------------------------------
    # Same shape as the T3D reward so behaviour stays recognisable. DreamerV3
    # symlog-transforms rewards internally, so the absolute scale matters much
    # less to it than it did to TD3.
    reward_step: float = -0.1
    reward_crash: float = -100.0
    reward_target: float = 100.0
    # Kept below reward_step / len(sensor_angles_deg) (0.1/7 ~ 0.014), so that
    # surviving with every sensor clear (7 * this, per step) never outearns
    # the per-step existence cost on its own. Without that, a policy that
    # never approaches the goal can still rack up unbounded reward just by
    # loitering in open space for longer - which is exactly what happened at
    # the old value of 2.5 (7 * 2.5 = 17.5/step, dwarfing reward_target's
    # one-time +100 within ~6 steps of survival). See
    # notes/journal.md, 2026-09-01, for the training run that found this.
    reward_per_clear_sensor: float = 0.01
    reward_progress_scale: float = 15.0
    reward_alignment_scale: float = 3.0

    def __post_init__(self):
        if self.collision_mode not in ("center", "footprint"):
            raise ValueError(
                f"collision_mode must be 'center' or 'footprint', got {self.collision_mode!r}")
        if not (self.use_image or self.use_vector):
            raise ValueError("at least one of use_image / use_vector must be True")
        if self.min_speed > self.max_speed:
            raise ValueError("min_speed must not exceed max_speed")
        if self.target_min_dist > self.target_max_dist:
            raise ValueError("target_min_dist must not exceed target_max_dist")
        if not Path(self.map_path).exists():
            raise FileNotFoundError(f"map not found: {self.map_path}")

    def to_dict(self):
        return asdict(self)
