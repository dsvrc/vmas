#  The Shared Link Contention config block, shared by every task in the family.
#
#  Every field here is a **declared deployment constant** except ``slc_severity``
#  which is the experimental variable, and the two PACT calibration parameters
#  ``pact_max_trust`` / ``pact_mu`` which must be swept and reported with their
#  sweep.  Defaults live in the yaml, not here: ``MISSING`` forces the yaml to
#  state every value, so a partially-overridden config cannot silently run a
#  different environment.

from dataclasses import dataclass, MISSING


@dataclass
class SlcConfigBase:
    # --- the severity dial (TASK physics: reaches every arm) ---------------
    slc_severity: float = MISSING
    slc_driver_period: int = MISSING
    slc_phase_spread: bool = MISSING
    slc_a_ref: float = MISSING
    slc_quiet_scale: float = MISSING
    slc_p_quiet: float = MISSING

    # --- the link budget: anchors sigma = 1 --------------------------------
    slc_snr_ref: float = MISSING
    slc_g_min_at_sigma1: float = MISSING
    slc_mean_preserve: bool = MISSING

    # --- the declared operator ---------------------------------------------
    slc_n_chan: int = MISSING
    slc_aclr: float = MISSING
    slc_leak_span: int = MISSING
    slc_duty_lo: float = MISSING
    slc_duty_hi: float = MISSING
    slc_exposure_lo: float = MISSING
    slc_exposure_hi: float = MISSING
    slc_lfix_frac: float = MISSING
    slc_u_nominal: float = MISSING
    slc_capacity_mode: str = MISSING
    slc_capacity_ref_agents: int = MISSING

    # --- the exertion functional -------------------------------------------
    slc_phi_floor: float = MISSING
    slc_phi_slope: float = MISSING
    slc_v_ref: float = MISSING
    slc_phi_reads_executed: bool = MISSING

    # --- the harm channel ---------------------------------------------------
    slc_harm_at_nominal: float = MISSING
    slc_harm_cap: float = MISSING
    slc_harm_enabled: bool = MISSING
    slc_discrete_nvec: int = MISSING
    slc_observe_loading: bool = MISSING

    # --- the compensator (NEVER read by the dial) --------------------------
    pact_enabled: bool = MISSING
    pact_mode: str = MISSING
    pact_gate: str = MISSING
    pact_r: int = MISSING
    pact_mu: float = MISSING
    pact_p0: float = MISSING
    pact_p_max_mult: float = MISSING
    pact_max_trust: float = MISSING
    pact_ff_gain: float = MISSING
    pact_own_gain: float = MISSING
    pact_fit_floor: float = MISSING
    pact_ready_updates: int = MISSING
    pact_fit_ema: float = MISSING
    pact_warmup_updates: int = MISSING
    pact_level_tau: float = MISSING
    pact_max_delta: float = MISSING
    pact_denom_floor: float = MISSING
    pact_u_cap: float = MISSING
