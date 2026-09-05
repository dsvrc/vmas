#  Shared Link Contention (SLC) -- the category-C non-stationarity, arithmetic only.
#
#  ---------------------------------------------------------------------------
#  The story
#  ---------------------------------------------------------------------------
#  A VMAS world is a robot swarm.  Real swarms share one finite medium that no
#  scenario models: the **radio link**.  Every robot's controller runs on state
#  that arrives over a channel it shares with the fleet -- telemetry to the
#  supervisor, ranging to the localization anchors.  A robot that moves faster
#  must report its pose more often to keep the fleet's dead-reckoning error
#  bounded (event-triggered communication: update rate scales with speed), so
#  motion *is* airtime.  When a channel is congested, setpoints arrive late, the
#  low-level velocity loop runs at a lower effective rate, and a sampled-data
#  loop with rate 1/T has closed-loop gain proportional to 1/T -- the robot
#  simply delivers less force than it commanded.
#
#  An exogenous emitter in the facility (the site's own WLAN, a neighbouring
#  cell) cycles slowly over the operating day and raises the noise floor, which
#  *shrinks channel capacity*.  Because loading is a **ratio** of demanded
#  airtime to available capacity, shrinking the denominator multiplies every
#  agent's contribution to every other agent's loading.  Reward is untouched;
#  the fleet simply achieves less because the medium delivers less.
#
#  ---------------------------------------------------------------------------
#  Why this is category C
#  ---------------------------------------------------------------------------
#  The driver reaches the agents ONLY by scaling ``K``, never by adding a term
#  to ``L`` (``NS_FORM_SPEC`` A.3).  Since ``u_i = L_i / (K_i^0 g)``, the factor
#  ``1/g`` multiplies every term including the peer sum -- and at ``N = 1`` the
#  peer sum is empty, so the cross-agent contribution is **exactly zero however
#  small g becomes**.  Irreducibility is structural, asserted by
#  ``SlcOperator.check()`` (zero diagonal of ``W``), not verified after the
#  fact.
#
#  ---------------------------------------------------------------------------
#  Scope of this module
#  ---------------------------------------------------------------------------
#  torch only.  No ``vmas``, no ``torchrl`` imports, so the arithmetic can be
#  unit-tested and the Part-C ceiling decomposition computed on a laptop with no
#  simulator installed.  The scenario AND the compensator both import from here,
#  so they cannot drift apart.

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

__all__ = [
    "SlcParams",
    "SlcOperator",
    "build_operator",
    "driver_A",
    "effective_driver",
    "dial_g",
    "exertion",
    "channel_load",
    "loading",
    "harm_coefficient",
    "apply_harm",
    "invert_harm",
    "decompose_excess",
    "SHIFT_BUSY",
    "SHIFT_QUIET",
]

SHIFT_BUSY = 0
SHIFT_QUIET = 1


# =============================================================================
#  Declared constants
# =============================================================================


@dataclass(frozen=True)
class SlcParams:
    """Every number the non-stationarity needs.  All of them are **declared**:
    deployment facts a network engineer would write down, fixed at construction
    and never fitted to run data.

    The only knob that is an experimental variable is :attr:`severity`.
    """

    # --- the severity dial -------------------------------------------------
    severity: float = 1.0
    """``sigma``.  ``0`` gives ``g == 1`` exactly at every driver value."""

    # --- the exogenous driver ----------------------------------------------
    driver_period: int = 2000
    """Steps per emitter cycle.  ~20 episodes at ``max_steps=100``: the driver
    is near-constant *within* an episode and drifts *across* them."""
    phase_spread: bool = True
    """Give each vectorised world its own fixed phase (a fleet operating round
    the clock), so every collected batch spans the whole cycle."""
    a_ref: float = 0.5
    """Reference emitter level.  The link budget is anchored here, and this is
    what makes the quiet regime a *provable* placebo (B.4)."""
    quiet_scale: float = 0.5
    """In a quiet shift the emitter never exceeds ``a_ref``, so ``g`` clips to
    exactly ``1.0000`` for every ``sigma``.  This is the placebo regime."""
    p_quiet: float = 0.0
    """Probability an episode is drawn on the quiet shift.  Set ``1.0`` for the
    placebo certificate; ``0.0`` for the headline sweep."""

    # --- the link budget (anchors sigma = 1) --------------------------------
    snr_ref: float = 100.0
    """SINR at the reference emitter level (20 dB).  A normal industrial
    2.4 GHz link budget."""
    g_min_at_sigma1: float = 0.65
    """Capacity ratio at the emitter peak when ``sigma = 1``.  **This is the
    anchor** (B.3): a measured busy-hour capacity ratio for the band, i.e. a
    1.54x intra-day swing.  ``sigma > 1`` is a beyond-physical stress test and
    must be labelled as such in every table."""
    mean_preserve: bool = False
    """D.2: divide ``g`` by its own cycle mean so only the *shape* varies and no
    total capacity is removed.  This deliberately allows ``g > 1``, trading
    B.1.3 for D.2's slack condition.  Off by default; report G4a either way."""

    # --- the medium ---------------------------------------------------------
    n_chan: int = 3
    """Channels in the deployment's frequency plan."""
    aclr: float = 0.25
    """Spectral overlap between channels one hop apart.

    ``0.25`` is the ordinary *partially overlapping* 2.4 GHz picture -- the
    unplanned channel map every warehouse actually has -- not the -30 dB
    adjacent-channel rejection of a clean orthogonal plan.  A clean plan would
    leave each agent coupled to its co-channel peers and nobody else, which
    makes the peer basis effectively rank-one; this keeps every peer live with
    genuinely different weights, which is what the basis needs.

    **These defaults must equal the shipped yaml.**  ``pact2/check_plumbing.py``
    asserts it: Phase 0 reads the dataclass while training reads the yaml, so a
    divergence means the calibration was measured against a different
    environment than the one that trains.  That happened once, silently."""
    leak_span: int = 2
    """Channels beyond which leakage is exactly zero."""
    duty_lo: float = 0.7
    duty_hi: float = 1.4
    """Per-robot duty scale, log-spaced over the fleet.  Heterogeneous robot
    classes are what make ``W`` **asymmetric**."""
    exposure_lo: float = 0.6
    exposure_hi: float = 1.4
    """Per-channel emitter exposure.  Makes the dial's damage uneven across
    channels, so the binding link actually switches over time."""
    lfix_frac: float = 0.15
    """Facility infrastructure traffic as a fraction of nominal fleet load.
    Not controllable by any agent -- this is ``L^fixed`` and it is the
    irreducible share of the excess."""
    u_nominal: float = 0.30
    """Provisioned utilisation.  ``K^0`` is derived from it, so the network is
    sized for the fleet and the N-sweep isolates the coordination gap rather
    than confounding it with raw congestion."""
    capacity_mode: str = "scaled"
    """``scaled``: ``K^0`` provisioned for the fleet size (default).
    ``fixed``: ``K^0`` provisioned for ``capacity_ref_agents`` regardless of N,
    so adding robots really does congest the medium."""
    capacity_ref_agents: int = 6

    # --- the exertion functional Phi ---------------------------------------
    phi_floor: float = 0.25
    """Airtime a robot spends at rest.  **This is where the coordination
    broadcast is paid for**: PACT's one-scalar-per-robot-per-step message rides
    in the heartbeat frame every robot already sends, and every arm is charged
    for it whether or not it uses it."""
    phi_slope: float = 1.875
    """Airtime per unit speed.  ``phi_floor + phi_slope * v_ref == 1.0``."""
    v_ref: float = 0.4
    """Declared reference speed.  VMAS holonomic terminal speed at
    ``u_range=1, mass=1, dt=0.1, drag=0.25`` is ``u*dt/(m*drag) = 0.4``."""
    phi_reads_executed: bool = True
    """A.6, the loop question.  ``True``: ``Phi`` reads the executed motion, so
    compensating feeds the medium it compensates against -- T4 applies and an
    interior optimum in ``max_trust`` is predicted.  ``False``: the no-loop
    contrast arm, where T4 is predicted NOT to apply."""

    # --- the harm channel ---------------------------------------------------
    harm_at_nominal: float = 0.30
    """Fraction of commanded force lost at the provisioned operating point.

    **This is an anchor, not a tuned knob.**  A control loop only gets its
    setpoints through in the airtime the channel is idle, so the delivered loop
    gain is the idle fraction ``1 - u``.  Setting this equal to ``u_nominal``
    makes ``harm_gain`` exactly ``1.0``, which is that statement and nothing
    more.  Any other value would need a defence."""
    harm_cap: float = 0.85
    """A robot never loses all authority; the loop degrades to dead reckoning."""
    harm_enabled: bool = True
    """``False`` recovers stock VMAS byte for byte -- the absolute reference
    B0, distinct from the ``sigma = 0`` stationary-contention task."""

    # --- observation --------------------------------------------------------
    observe_loading: bool = True
    """Append the agent's own (one-step-stale) loading to its observation.
    A.4: the sensor reports the past, and that gap is the problem."""

    # --- derived (filled by __post_init__ via object.__setattr__) -----------
    phi_max: float = field(init=False, default=0.0)
    phi_nominal: float = field(init=False, default=0.0)
    harm_gain: float = field(init=False, default=0.0)
    noise_rise: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        if not (0.0 <= self.a_ref < 1.0):
            raise ValueError(f"a_ref must be in [0,1), got {self.a_ref}")
        if self.severity < 0.0:
            raise ValueError("severity must be >= 0")
        if self.n_chan < 2:
            raise ValueError("n_chan must be >= 2 for distinct link sets")
        if not (0.0 < self.g_min_at_sigma1 < 1.0):
            raise ValueError("g_min_at_sigma1 must be in (0,1)")
        if self.capacity_mode not in ("scaled", "fixed"):
            raise ValueError(f"unknown capacity_mode {self.capacity_mode!r}")

        object.__setattr__(self, "phi_max", self.phi_floor + self.phi_slope * self.v_ref)
        object.__setattr__(
            self, "phi_nominal", 0.5 * (self.phi_floor + self.phi_max)
        )
        object.__setattr__(
            self, "harm_gain", self.harm_at_nominal / max(self.u_nominal, 1e-9)
        )

        # --- solve the noise rise from the anchor -------------------------
        # We want, at sigma = 1, exposure = 1, A_eff = 1:
        #     log2(1 + snr_peak) == g_min * log2(1 + snr_ref)
        # and  snr = snr_ref / (1 + noise_rise * (A_eff - a_ref)).
        c_ref = math.log2(1.0 + self.snr_ref)
        snr_peak = (2.0 ** (self.g_min_at_sigma1 * c_ref)) - 1.0
        if snr_peak <= 0.0:
            raise ValueError("g_min_at_sigma1 is too small for this snr_ref")
        noise_rise = (self.snr_ref / snr_peak - 1.0) / (1.0 - self.a_ref)
        object.__setattr__(self, "noise_rise", noise_rise)

    # -- convenience --------------------------------------------------------

    @property
    def phi_span(self) -> float:
        """``Phi_max - Phi_floor``: the *varying* part of the exertion."""
        return self.phi_max - self.phi_floor


# =============================================================================
#  The declared coupling operator
# =============================================================================


class SlcOperator:
    """The environment's own model of the medium, computed once at construction
    and **never fitted**.

    ``NS_FORM_SPEC`` A.2 object 2.  The pieces:

    ``E[k, j]``
        Airtime robot *j*'s traffic places on channel *k*.  Channel plan x
        adjacent-channel leakage x per-robot duty scale.  This is the domain's
        real operator, the direct analogue of a PTDF matrix.
    ``D[i, k]``
        Whether robot *i*'s control loop depends on channel *k*.  Two links per
        robot: its data channel and its ranging band.  ``u_i`` is the **max**
        over these (A.4: the binding element, worth 6x in POWER).
    ``W[i, j] = mean_{k in links(i)} E[k, j]`` for ``j != i``, ``W[i, i] = 0``
        The reduced per-agent peer operator PACT's basis is built from.  The
        own-effect lives in the self term, never in ``W`` -- **asserted**.
    """

    def __init__(
        self,
        params: SlcParams,
        n_agents: int,
        device: torch.device,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.params = params
        self.n_agents = int(n_agents)
        self.n_chan = int(params.n_chan)
        self.device = device
        self.dtype = dtype

        n, c = self.n_agents, self.n_chan
        p = params

        # --- channel plan: round robin, a deployment fact ------------------
        chan_of = torch.arange(n, device=device) % c
        # ranging band offset so robots sharing a data channel do not also share
        # their ranging band
        rng_of = (chan_of + 1 + torch.arange(n, device=device) // c) % c
        rng_of = torch.where(rng_of == chan_of, (chan_of + 1) % c, rng_of)

        # --- per-robot duty scale: heterogeneous robot classes -------------
        if n == 1:
            duty = torch.full((1,), p.duty_lo, device=device, dtype=dtype)
        else:
            t = torch.linspace(0.0, 1.0, n, device=device, dtype=dtype)
            duty = p.duty_lo * (p.duty_hi / p.duty_lo) ** t

        # --- channel overlap: ACLR per hop, hard zero past leak_span -------
        ks = torch.arange(c, device=device).view(c, 1)
        cs = torch.arange(c, device=device).view(1, c)
        hops = (ks - cs).abs()
        overlap = torch.where(
            hops <= p.leak_span,
            torch.as_tensor(p.aclr, device=device, dtype=dtype) ** hops.to(dtype),
            torch.zeros((), device=device, dtype=dtype),
        )

        # --- E[k, j] --------------------------------------------------------
        E = overlap[:, chan_of] * duty.view(1, n)

        # --- D[i, k] --------------------------------------------------------
        D = torch.zeros((n, c), device=device, dtype=torch.bool)
        D[torch.arange(n, device=device), chan_of] = True
        D[torch.arange(n, device=device), rng_of] = True

        # --- W[i, j] = mean over i's links of E[k, j], zero diagonal --------
        link_count = D.sum(-1, keepdim=True).to(dtype)
        W = (D.to(dtype) @ E) / link_count
        W.fill_diagonal_(0.0)

        # --- per-channel emitter exposure ----------------------------------
        if c == 1:
            exposure = torch.ones(1, device=device, dtype=dtype)
        else:
            exposure = torch.linspace(
                p.exposure_lo, p.exposure_hi, c, device=device, dtype=dtype
            )

        # --- fixed infrastructure load and provisioned capacity ------------
        nominal_fleet = E.sum(-1) * p.phi_nominal  # (c,)
        Lfix = p.lfix_frac * nominal_fleet
        if p.capacity_mode == "scaled":
            K0 = (nominal_fleet + Lfix) / p.u_nominal
        else:
            ref_n = max(int(p.capacity_ref_agents), 1)
            ref_chan_of = torch.arange(ref_n, device=device) % c
            if ref_n == 1:
                ref_duty = torch.full((1,), p.duty_lo, device=device, dtype=dtype)
            else:
                t = torch.linspace(0.0, 1.0, ref_n, device=device, dtype=dtype)
                ref_duty = p.duty_lo * (p.duty_hi / p.duty_lo) ** t
            ref_E = overlap[:, ref_chan_of] * ref_duty.view(1, ref_n)
            ref_nominal = ref_E.sum(-1) * p.phi_nominal
            K0 = (ref_nominal * (1.0 + p.lfix_frac)) / p.u_nominal
        K0 = K0.clamp_min(1e-9)

        self.chan_of = chan_of
        self.rng_of = rng_of
        self.duty = duty
        self.overlap = overlap
        self.E = E
        self.D = D
        self.W = W
        self.exposure = exposure
        self.Lfix = Lfix
        self.K0 = K0
        # log2(1 + snr_ref) computed in the *tensor* dtype so that the sigma = 0
        # ratio is bit-exactly 1.0 (a python-float denominator would not be).
        self.c_ref = torch.log2(
            torch.as_tensor(1.0 + p.snr_ref, device=device, dtype=dtype)
        )

        self.check()

    # -- structural checks (PACT_PIPELINE_SPEC 2.4) -------------------------

    def check(self) -> None:
        W = self.W
        diag = W.diagonal()
        if not torch.all(diag == 0):
            raise AssertionError(
                "W must be zero-diagonal: the own-effect belongs in the self "
                "term, never in the coupling operator.  Irreducibility at N=1 "
                "is structural and this assertion is what makes it so."
            )
        if not torch.isfinite(W).all():
            raise AssertionError("W contains non-finite entries")
        if float(self.K0.min()) <= 0.0:
            raise AssertionError("channel capacity must be strictly positive")
        if not bool(self.D.any(-1).all()):
            raise AssertionError("every agent must depend on at least one channel")

    # -- reporting -----------------------------------------------------------

    def live_peers(self) -> Tensor:
        """Per-agent peer coupling mass.  Agents with zero mass have no live
        coupling and stay at ``g = 0`` forever -- name them, do not hide them."""
        return self.W.sum(-1)

    def spread(self) -> float:
        """``std/mean`` of the non-zero off-diagonal weights.  POWER measured
        **1.35**; a geometric proxy that could not represent this measured
        ``fit_gain = -0.0045``."""
        off = self.W[~torch.eye(self.n_agents, dtype=torch.bool, device=self.device)]
        off = off[off > 0]
        if off.numel() < 2:
            return 0.0
        return float(off.std(unbiased=False) / off.mean().clamp_min(1e-30))

    def asymmetry(self) -> float:
        """Mean relative ``|W[i,j] - W[j,i]|``.  A symmetric two-bucket relation
        cannot represent a real operator; report the number."""
        W = self.W
        num = (W - W.T).abs()
        den = (W + W.T).clamp_min(1e-30)
        mask = ~torch.eye(self.n_agents, dtype=torch.bool, device=self.device)
        return float((num[mask] / den[mask]).mean())

    def summary(self) -> Dict[str, float]:
        live = self.live_peers()
        dead = int((live <= 0).sum())
        return {
            "n_agents": float(self.n_agents),
            "n_chan": float(self.n_chan),
            "W_spread": self.spread(),
            "W_asymmetry": self.asymmetry(),
            "W_max": float(self.W.max()),
            "W_min_positive": float(self.W[self.W > 0].min()) if bool((self.W > 0).any()) else 0.0,
            "agents_without_coupling": float(dead),
            "K0_min": float(self.K0.min()),
            "Lfix_frac": float(self.params.lfix_frac),
        }

    def banner(self) -> str:
        s = self.summary()
        live = self.live_peers()
        dead = [i for i in range(self.n_agents) if float(live[i]) <= 0.0]
        lines = [
            "SLC operator  "
            f"N={self.n_agents} chan={self.n_chan} "
            f"spread(std/mean)={s['W_spread']:.2f} "
            f"asym={s['W_asymmetry']:.2f} "
            f"W in [{s['W_min_positive']:.3e}, {s['W_max']:.3e}]",
        ]
        if dead:
            lines.append(
                f"  agents with NO live coupling (g stays 0 forever): {dead}"
            )
        return "\n".join(lines)

    def to(self, device: torch.device) -> "SlcOperator":
        for name in (
            "chan_of",
            "rng_of",
            "duty",
            "overlap",
            "E",
            "D",
            "W",
            "exposure",
            "Lfix",
            "K0",
            "c_ref",
        ):
            setattr(self, name, getattr(self, name).to(device))
        self.device = device
        return self


def build_operator(
    params: SlcParams,
    n_agents: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> SlcOperator:
    return SlcOperator(params, n_agents, device, dtype)


# =============================================================================
#  The exogenous driver and the severity dial
# =============================================================================


def driver_A(step: Tensor, phase: Tensor, period: int) -> Tensor:
    """``A(t) in [0, 1]``, a smooth raised cosine on a slow cycle.

    ``step`` is a **global** counter that episode boundaries never reset -- that
    is what makes the driver exogenous rather than something the agents' own
    resets could rewind.
    """
    x = step.to(torch.float32) / float(period) + phase
    return 0.5 * (1.0 - torch.cos(2.0 * math.pi * x))


def effective_driver(a: Tensor, shift: Tensor, params: SlcParams) -> Tensor:
    """Apply the shift.  On a quiet shift the emitter never exceeds ``a_ref``,
    so the dial provably does nothing (B.4, the placebo regime).
    """
    quiet = shift == SHIFT_QUIET
    return torch.where(quiet, a * params.quiet_scale, a)


def dial_g(
    a_eff: Tensor,
    params: SlcParams,
    operator: SlcOperator,
) -> Tensor:
    """``g(A) in (0, 1]`` per channel: the fraction of nominal capacity the
    medium currently delivers.

    Requirements from ``NS_FORM_SPEC`` B.1, and where each is met:

    1. **Identity at zero.**  ``sigma = 0`` makes the denominator exactly
       ``1.0``, so ``snr == snr_ref`` bit for bit and ``g == 1`` exactly at
       *every* driver value.  Not approximately.
    2. **Monotone in difficulty** at every driver value: above ``a_ref`` a
       larger ``sigma`` lowers ``g``; below it the raw law would *uprate*, and
       the clip in 3 pins it flat at 1.
    3. **Never generous:** ``clamp_max(1.0)``.  This is B.2's uprating trap --
       Shannon's law is two-sided and we keep only the harmful half.
    4. **Physically anchored:** ``noise_rise`` is solved from
       ``g_min_at_sigma1``, a measured busy-hour capacity ratio.

    Args:
        a_eff: ``(B,)`` or ``(B, 1)`` effective driver level.
        params: declared constants.
        operator: supplies per-channel exposure and the dtype-matched ``c_ref``.

    Returns:
        ``(B, n_chan)``
    """
    a = a_eff.reshape(-1, 1).to(operator.dtype)
    rise = params.severity * params.noise_rise
    denom = 1.0 + rise * operator.exposure.view(1, -1) * (a - params.a_ref)
    # Guard the pole.  Anywhere the denominator would go small the raw g is
    # already > 1 and the clip below pins it at exactly 1, so the floor cannot
    # change any delivered value.
    denom = denom.clamp_min(0.05)
    snr = params.snr_ref / denom
    g = torch.log2(1.0 + snr) / operator.c_ref
    g = g.clamp_max(1.0)
    if params.mean_preserve:
        g = g / _cycle_mean_g(params, operator).view(1, -1)
    return g


def _cycle_mean_g(params: SlcParams, operator: SlcOperator) -> Tensor:
    """Analytic mean of ``g`` over one driver cycle, per channel.

    D.2's mean-preserving option divides by this so only the *shape* of the
    capacity varies and no total capacity is removed.  Computed from the
    declared model on a fixed 512-point quadrature -- **never from run data**,
    which would turn a declared model class into a fit.
    """
    key = "_cycle_mean_g_cache"
    cached = getattr(operator, key, None)
    if cached is not None:
        return cached
    t = torch.linspace(0.0, 1.0, 513, device=operator.device, dtype=operator.dtype)[:-1]
    a = 0.5 * (1.0 - torch.cos(2.0 * math.pi * t))
    rise = params.severity * params.noise_rise
    denom = (
        1.0 + rise * operator.exposure.view(1, -1) * (a.view(-1, 1) - params.a_ref)
    ).clamp_min(0.05)
    g = (torch.log2(1.0 + params.snr_ref / denom) / operator.c_ref).clamp_max(1.0)
    mean = g.mean(0).clamp_min(1e-6)
    setattr(operator, key, mean)
    return mean


# =============================================================================
#  Exertion, load, loading
# =============================================================================


def exertion(vel: Tensor, params: SlcParams) -> Tensor:
    """``Phi_j`` -- the airtime robot *j* demands.

    ``NS_FORM_SPEC`` A.5, the escape-hatch rule.  This is:

    * a **magnitude** (``||v||``), never a signed sum, so no anti-symmetric
      configuration can make the fleet's total exertion cancel;
    * **uncancellable** -- driving it to the floor means not moving, and in
      every task this NS ships with (``sampling``, ``discovery``, ``navigation``)
      not moving forfeits reward directly;
    * **varying** -- check ``std(Phi)/mean(Phi) > 0.05`` before committing.

    ``phi_floor`` is the heartbeat every robot sends at rest.  **PACT's
    coordination broadcast rides in it** and every arm is charged for it.

    Args:
        vel: ``(B, N, 2)`` velocities.

    Returns:
        ``(B, N)``
    """
    speed = torch.linalg.vector_norm(vel, dim=-1)
    return params.phi_floor + params.phi_slope * speed


def channel_load(phi: Tensor, operator: SlcOperator) -> Tensor:
    """``L_k(t) = sum_j E[k, j] Phi_j(t) + L^fixed_k``.

    Args:
        phi: ``(B, N)``

    Returns:
        ``(B, n_chan)``
    """
    return phi @ operator.E.transpose(0, 1) + operator.Lfix.view(1, -1)


def loading(
    load: Tensor, g: Tensor, operator: SlcOperator
) -> Tuple[Tensor, Tensor]:
    """``u_i = max_{k in links(i)} L_k / (K^0_k g_k)``.

    A.4: the **binding** aggregate, not the average.  Congestion is a property
    of the worst link; averaging over an agent's links dilutes the signal toward
    zero (POWER measured 6x more applied compensation from this one change).

    Returns:
        ``(u, k_star)`` each ``(B, N)`` -- the loading and which channel binds.
    """
    ratio = load / (operator.K0.view(1, -1) * g)
    masked = ratio.unsqueeze(1).masked_fill(
        ~operator.D.unsqueeze(0), float("-inf")
    )
    u, k_star = masked.max(dim=-1)
    return u, k_star


# =============================================================================
#  The harm channel and its inverse
# =============================================================================


def harm_coefficient(u: Tensor, params: SlcParams) -> Tensor:
    """``c = harm_gain * u``, clamped.

    A congested link means fewer setpoint updates per unit time; a sampled-data
    loop's achievable closed-loop gain scales with its update rate, so the robot
    delivers ``(1 - c)`` of the force it commanded.  Textbook, and it keeps the
    channel **continuous and invertible** (A.2 object 4, A.7).
    """
    if not params.harm_enabled:
        return torch.zeros_like(u)
    return (params.harm_gain * u).clamp(0.0, params.harm_cap)


def apply_harm(action: Tensor, c: Tensor) -> Tensor:
    """``f_exec = (1 - c) * f_cmd``.  Multiplicative, zero-preserving."""
    return action * (1.0 - c).unsqueeze(-1)


def invert_harm(action: Tensor, c_hat: Tensor, denom_floor: float) -> Tensor:
    """``a' = a / (1 - c_hat)`` -- the harm channel inverse.

    Usable until the rail.  ``denom_floor`` is what stops a diverging estimate
    from asking for infinite force; a delta pinned at the rail is a **constant
    bias, not a compensation** (PACT_PIPELINE_SPEC 6.3), so track how often it
    binds.
    """
    denom = (1.0 - c_hat).clamp_min(denom_floor)
    return action / denom.unsqueeze(-1)


# =============================================================================
#  PART C -- the ceiling decomposition
# =============================================================================


def decompose_excess(
    phi: Tensor,
    g: Tensor,
    operator: SlcOperator,
) -> Dict[str, Tensor]:
    """The Part-C decomposition: who can actually fix each unit of damage.

    Under the dial each agent's loading exceeds its ``sigma = 0`` counterfactual
    by ``Delta_i = u_i (1 - g)``.  Every unit of that traces to a contributor,
    and the contributors partition by **who can move them**:

    ============  ==========================================  ================
    class         source                                      recoverable by
    ============  ==========================================  ================
    ``fixed``     ``L^fixed`` -- nothing any agent controls    nobody
    ``own``       agent *i*'s own exertion, amplified by 1/g   any policy, alone
    ``peer``      peers' exertion through ``W``, by 1/g        **coordination**
    ============  ==========================================  ================

    Computed from the declared operator and observed state with **no training
    and no method**.  ``NS_FORM_SPEC`` C.2 attributes in proportion to signed
    contribution, counting only what could actually help -- every entry of our
    ``E`` is non-negative, so the ``clamp_min(0)`` is a no-op here, but it is
    kept so the arithmetic matches the spec exactly.

    Args:
        phi: ``(B, N)`` exertion.
        g:   ``(B, n_chan)`` capacity ratio.

    Returns:
        dict of ``(B, N)`` tensors plus scalar fractions.
    """
    load = channel_load(phi, operator)
    u, k_star = loading(load, g, operator)

    n = operator.n_agents
    idx = torch.arange(n, device=phi.device)

    # E[k*(i), j] for every (i, j): gather the binding channel's row per agent
    E_star = operator.E[k_star]  # (B, N, N_src)
    contrib = (E_star * phi.unsqueeze(1)).clamp_min(0.0)  # (B, N, N_src)

    own = contrib[:, idx, idx]  # (B, N)
    peer = contrib.sum(-1) - own
    fixed = operator.Lfix[k_star]  # (B, N)

    total = (own + peer + fixed).clamp_min(1e-12)
    g_bind = torch.gather(g, 1, k_star)
    delta = u * (1.0 - g_bind)

    d_own = delta * own / total
    d_peer = delta * peer / total
    d_fixed = delta * fixed / total

    sum_total = delta.sum().clamp_min(1e-12)
    return {
        "u": u,
        "k_star": k_star,
        "delta": delta,
        "delta_own": d_own,
        "delta_peer": d_peer,
        "delta_fixed": d_fixed,
        "irreducible": d_fixed.sum() / sum_total,
        "own_free": d_own.sum() / sum_total,
        "coordination_gap": d_peer.sum() / sum_total,
        "decentralized_ceiling": 1.0 - d_fixed.sum() / sum_total,
        "non_coordinating_ceiling": 1.0
        - (d_fixed.sum() + d_peer.sum()) / sum_total,
    }
