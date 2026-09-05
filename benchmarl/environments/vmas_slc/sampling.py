from dataclasses import dataclass, MISSING

from benchmarl.environments.vmas_slc._config_base import SlcConfigBase


@dataclass
class TaskConfig(SlcConfigBase):
    max_steps: int = MISSING
    n_agents: int = MISSING
    shared_rew: bool = MISSING
    n_gaussians: int = MISSING
    lidar_range: float = MISSING
    cov: float = MISSING
    collisions: bool = MISSING
    spawn_same_pos: bool = MISSING
