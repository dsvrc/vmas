#  Hydra schema for the ``vmas_ns/navigation_pcw`` task.

from dataclasses import dataclass, MISSING
from typing import Optional


@dataclass
class TaskConfig:
    # ---- stock VMAS navigation (semantics unchanged) ----------------------
    max_steps: int = MISSING
    n_agents: int = MISSING
    collisions: bool = MISSING
    agents_with_same_goal: int = MISSING
    observe_all_goals: bool = MISSING
    shared_rew: bool = MISSING
    split_goals: bool = MISSING
    lidar_range: float = MISSING
    agent_radius: float = MISSING

    # ---- the non-stationarity: ONE severity dial --------------------------
    ns_severity: float = MISSING

    # ---- fixed structural constants (calibrate once, then leave alone) ----
    ns_gain: float = MISSING
    ns_rho: float = MISSING
    ns_driver_period: int = MISSING
    ns_phase_spread: bool = MISSING

    # ---- Phase-1 only -----------------------------------------------------
    ns_freeze_driver: Optional[float] = MISSING

    # ---- PACT (the method) ------------------------------------------------
    pact_enabled: bool = MISSING
    pact_oracle: bool = MISSING
    pact_beta_max: Optional[float] = MISSING
    pact_beta_mode: str = MISSING
    pact_beta_ema: float = MISSING
    pact_obs_features: bool = MISSING
    pact_gate: bool = MISSING
    pact_gate_tol: float = MISSING
