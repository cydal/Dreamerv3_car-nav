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
    # Continuous ray-marched clearance per angle (see
    # observations.sensor_distances), not a single fixed-distance binary
    # read. That distinction is what lets max_range be long (real lookahead)
    # without the old overshoot problem: T3D's original 20px single-endpoint
    # read a 14px-wide corridor's own far wall as "obstacle" even when the
    # car was centred (23% of on-road poses read 0/7 at that setting) -
    # marching to the first actual obstacle instead of sampling one point
    # doesn't have that failure mode, so range can go well past 20px.
    use_sensors: bool = True
    sensor_angles_deg: Tuple[float, ...] = (-60, -45, -30, -15, 0, 15, 30, 45, 60)
    sensor_max_range: float = 45.0      # marched out this far per angle, px
    sensor_step_px: float = 1.5         # raymarch resolution, px

    # ---- observation -----------------------------------------------------
    use_image: bool = True
    image_size: int = 64                # output is image_size x image_size x 3
    # Raised 96->150 (2026-09-02): the car only saw ~96px of world around
    # itself, which showed an approaching turn very late. Same 64x64 output
    # resolution, larger extent per pixel - trades detail for lookahead
    # range. Untested whether this matters given the vector obs already
    # carries unbounded-range bearing/distance; goes in alongside the
    # reward-shaping fix below since both target the same "fails at turns"
    # symptom from different angles.
    crop_world_px: float = 150.0        # world extent the crop covers
    egocentric_rotation: bool = True    # rotate so the car always faces "up"
    draw_target_in_image: bool = True   # stamp the goal marker when in frame
    use_vector: bool = True             # dict obs gets a "vector" entry too

    # ---- road distance -----------------------------------------------------
    # Added 2026-09-03: Euclidean distance/bearing (goal_features, below) is a
    # bad reward/observation signal on a curved road network - a road bending
    # away from the straight line to the target makes the only viable move
    # look worse. Road distance (BFS shortest path over drivable pixels)
    # can't have that problem: a correct turn always reduces it. See
    # CityMap.road_distance_field and notes/journal.md for the measurements
    # behind the stride/cost tradeoff.
    use_road_distance: bool = False     # off by default; opt in per task
    road_distance_stride: int = 4       # downsample factor for the BFS grid
    road_distance_norm: float = 500.0   # normalizer for the vector feature

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
    # 2026-09-04: crash rate plateaued at ~77% even after the road-distance
    # reward fix closed the far-target gap - the remaining failure mode
    # looked like collision avoidance itself, not navigation. Replaced the
    # old binary "count of clear sensors" bonus with three continuous terms
    # built on sensor_distances' graded per-angle clearance:
    #   clearance - same spirit as the old per-clear-sensor bonus (reward
    #     open space a little), but continuous. Capped well below
    #     |reward_step| (0.05 < 0.1) for the same reason as before: a policy
    #     that never approaches the goal must not be able to out-earn the
    #     per-step existence cost just by loitering somewhere open. See
    #     notes/journal.md, 2026-09-01, for the run that found this the hard
    #     way at the old value (7 * 2.5 = 17.5/step).
    #   danger - a dense penalty that switches on once the *nearest* sensor
    #     reading drops inside danger_margin_px, scaling up to
    #     -reward_danger_scale right at the wall. This is the actual new
    #     signal: previously the only feedback about a near-miss was the
    #     terminal -100 crash itself, so the value function had to learn
    #     collision risk purely from that rare terminal, several steps after
    #     the point where avoiding it was still possible. This fires while
    #     there's still time to react.
    #   caution - a small bonus for slowing down specifically while inside
    #     caution_threshold of a wall, i.e. an explicit "brake near
    #     obstacles" incentive rather than hoping speed control emerges
    #     indirectly from the danger penalty. Only active near a wall (not
    #     everywhere), and capped at 0.05 - half of |reward_step| - so
    #     hugging a wall at minimum speed can't be a profitable substitute
    #     for making progress (min_speed itself is also a floor > 0, so the
    #     car can never fully stop to camp on this bonus).
    reward_clearance_scale: float = 0.05
    danger_margin_px: float = 8.0
    reward_danger_scale: float = 0.5
    caution_threshold: float = 0.4
    reward_caution_scale: float = 0.05
    # progress/alignment history:
    #   15.0 / 3.0  -> original
    #   40.0 / 8.0  -> raised 2026-09-02 morning: the first footprint run
    #                  showed both barely above zero per step even once
    #                  they weren't competing against the sensor term.
    #   40.0 / 3.0  -> alignment lowered back down 2026-09-02 evening: both
    #                  terms are straight-line proxies (progress rewards
    #                  closing Euclidean distance, alignment rewards facing
    #                  the target's straight-line bearing), which on a
    #                  curved/radial road network actively fights correct
    #                  turns - the road curving away from the straight line
    #                  is exactly when taking it correctly scores worse on
    #                  both terms. alignment is the more instant-by-instant
    #                  pressure (rewards facing the target *this step*), so
    #                  it fights a turn harder than progress does (which at
    #                  least averages net movement over time and can
    #                  tolerate a temporary detour). Lowering it back down
    #                  is the smaller, safer half of addressing the "only
    #                  reaches target when directly ahead, fails at turns"
    #                  symptom - progress stays raised since it's less
    #                  directly at odds with turning. See notes/journal.md.
    reward_progress_scale: float = 40.0
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
