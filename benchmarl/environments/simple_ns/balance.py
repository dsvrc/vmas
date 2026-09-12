from dataclasses import dataclass, MISSING


@dataclass
class TaskConfig:
    # --- the stock host -----------------------------------------------------
    max_steps: int = MISSING
    n_agents: int = MISSING
    random_package_pos_on_line: bool = MISSING
    package_mass: float = MISSING

    # --- the exertion dial (TASK physics: reaches EVERY arm) ---------------
    ns_severity: float = MISSING
    ns_period: int = MISSING
    ns_wet_fraction: float = MISSING
    ns_loss_at_sigma1: float = MISSING
    ns_rho: float = MISSING
    ns_n_types: int = MISSING
    ns_recv_spread: float = MISSING
    ns_send_spread: float = MISSING
    ns_kernel_lambda: float = MISSING
    ns_y_clip: float = MISSING
    ns_observe_residual: bool = MISSING
    ns_direct: bool = MISSING

    # --- PACT-1 (read by the method layer only) ----------------------------
    pact_enabled: bool = MISSING
    pact_trust: float = MISSING
    pact_mu: float = MISSING
    pact_p0: float = MISSING
    pact_warmup: int = MISSING
    pact_channels: str = MISSING
    pact_oracle: bool = MISSING
    pact_corr_clip: float = MISSING
    pact_p_trace_max: float = MISSING
