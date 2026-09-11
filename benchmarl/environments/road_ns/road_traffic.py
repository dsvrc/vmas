from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    # --- the stock task ----------------------------------------------------
    max_steps: int = MISSING
    n_agents: int = MISSING
    map_type: str = MISSING
    is_partial_observation: bool = MISSING
    n_nearing_agents_observed: int = MISSING
    is_observe_vertices: bool = MISSING
    is_add_noise: bool = MISSING

    # --- the severity dial (TASK physics: reaches EVERY arm) ---------------
    ns_severity: float = MISSING
    ns_period: int = MISSING
    ns_wet_fraction: float = MISSING
    ns_alpha: float = MISSING
    ns_mean_preserve: bool = MISSING
    ns_observe_loading: bool = MISSING
    ns_route_set: str = MISSING

    # --- PACT-1 (read by the method layer only) ----------------------------
    pact_enabled: bool = MISSING
    pact_trust: float = MISSING
    pact_kappa: float = MISSING
    pact_mu: float = MISSING
    pact_p0: float = MISSING
    pact_y_clip: float = MISSING
    pact_warmup: int = MISSING
