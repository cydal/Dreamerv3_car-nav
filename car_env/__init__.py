"""Standalone car navigation environment, extracted from the T3D project.

Contains no learning algorithm. Depends only on numpy + pillow, so it imports
cleanly in a training process with no Qt and no display.

    from car_env import CarNavEnv, CarNavConfig

    env = CarNavEnv(CarNavConfig(num_targets=2))
    obs, info = env.reset(seed=0)
    obs, reward, terminated, truncated, info = env.step([0.0, 1.0])
"""

from .citymap import CityMap
from .config import CarNavConfig
from .env import CarNavEnv
from .observations import EgocentricRenderer, goal_features, sensor_distances

__all__ = [
    "CarNavEnv",
    "CarNavConfig",
    "CityMap",
    "EgocentricRenderer",
    "goal_features",
    "sensor_distances",
]

__version__ = "0.1.0"
