from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    # --- the host ----------------------------------------------------------
    max_steps: int = MISSING
    n_agents: int = MISSING
    n_nearing_agents_observed: int = MISSING
    flow_n_samples: int = MISSING
    flow_n_lookahead: int = MISSING
    flow_lookahead_stride: int = MISSING
    flow_search_window: int = MISSING
    flow_respawn: bool = MISSING
    flow_reroute_on_lap: bool = MISSING
    flow_n_background: int = MISSING
    flow_background_speed: float = MISSING
    flow_integration: str = MISSING

    # --- the severity dial (TASK physics: reaches EVERY arm) ---------------
    ns_severity: float = MISSING
    ns_period: int = MISSING
    ns_wet_fraction: float = MISSING
    ns_alpha: float = MISSING
    ns_mean_preserve: bool = MISSING
    ns_observe_loading: bool = MISSING
    ns_route_set: str = MISSING
    ns_exclude_self: bool = MISSING

    # --- PACT-1 (read by the method layer only) ----------------------------
    pact_enabled: bool = MISSING
    pact_trust: float = MISSING
    pact_kappa: float = MISSING
    pact_mu: float = MISSING
    pact_p0: float = MISSING
    pact_y_clip: float = MISSING
    pact_warmup: int = MISSING
    pact_shift_mode: str = MISSING
    pact_shift_clip: float = MISSING
