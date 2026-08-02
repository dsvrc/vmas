#  PCW (Propwash-Circulation Wake) — shared arithmetic core.
#
#  This module is deliberately dependency-light: it imports **torch only**.
#  Both
#     * the VMAS scenario  (which *is* the non-stationarity), and
#     * the PACT wrapper   (which *compensates* for the non-stationarity)
#  take their arithmetic from here, so the two can never silently drift apart.
#  That is what makes the Phase-2 per-step cosine gate an exactness certificate
#  rather than a coincidence.
#
#  ---------------------------------------------------------------------------
#  The non-stationarity, in four lines
#  ---------------------------------------------------------------------------
#
#     m_j(t)    = ( p_j(t) x u_j(t) )_z                    # agent j's angular impulse
#     Phi_i(t)  = mean_{j != i} m_j(t)                     # OTHERS only  <- category-C signature
#     x2_i(t+1) = rho*x2_i(t) + (1-rho)*G*Phi_i(t)         # bulk circulation, driver-free
#     theta_i(t)= c(t) * x2_i(t),   c(t) = A(t)*sigma      # deflection angle, gated by the driver
#
#     delivered_i(t) = R(theta_i(t)) @ u_i(t)              # harm lives in the TRANSITION
#
#  `A(t)` multiplies the cross-agent sum and is never an additive term of its
#  own, so at N=1 the sum over `j != i` is empty, `x2 == 0` for all t, and the
#  environment reduces *exactly* to stationary VMAS navigation no matter how
#  large `A(t)` grows.  See `peer_mean` for where that is enforced.

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor

TWO_PI = 2.0 * math.pi

#: Value of the raw control dim ``w`` that a freshly initialised policy emits.
BETA_INIT_W = 0.0


@dataclass
class PcwParams:
    """Every constant of the non-stationarity.

    Only :attr:`severity` is a dial the user is expected to touch; everything
    else is a fixed structural constant calibrated once per environment (see
    ``pact/README.md`` §"Calibration").

    Args:
        severity: the one severity dial, ``sigma``.  ``0.0`` reduces the
            environment byte-for-byte to stationary VMAS navigation.
        gain: ``G``, radians of deflection per unit of accumulated circulation.
            Fixed structural constant; calibrated once against the *measured*
            operating scale of a trained policy.
        rho: leak constant of the medium's circulation (0.8 ~ a 5-step memory).
        driver_period: period ``T`` of the exogenous driver, in **global env
            steps**.  Must be >> the episode length so that ``c`` is
            approximately constant within an episode.
        phase_spread: if True each vectorised world sits at its own fixed
            offset in the driver cycle, so every batch spans the full cycle.
            This is an experimental-protocol switch, not a severity knob.
        freeze_driver: Phase-1 only.  When not None the driver is held at this
            constant value, turning the game stationary at effective severity
            ``freeze_driver * severity``.
    """

    severity: float = 0.45
    gain: float = 4.0
    rho: float = 0.8
    driver_period: int = 2000
    phase_spread: bool = True
    freeze_driver: Optional[float] = None

    def __post_init__(self):
        if self.severity < 0.0:
            raise ValueError(f"severity must be >= 0, got {self.severity}")
        if not 0.0 <= self.rho < 1.0:
            raise ValueError(f"rho must be in [0, 1), got {self.rho}")
        if self.driver_period <= 0:
            raise ValueError(f"driver_period must be > 0, got {self.driver_period}")
        if self.freeze_driver is not None and not 0.0 <= self.freeze_driver <= 1.0:
            raise ValueError(
                f"freeze_driver must be in [0, 1], got {self.freeze_driver}"
            )

    @property
    def peak_c(self) -> float:
        """The largest effective severity ``c = A*sigma`` the driver ever reaches."""
        if self.freeze_driver is not None:
            return self.freeze_driver * self.severity
        return self.severity


# ---------------------------------------------------------------------------
# The exogenous driver
# ---------------------------------------------------------------------------


def driver_A(
    global_step: int,
    batch_dim: int,
    *,
    period: int,
    phase_spread: bool,
    freeze: Optional[float] = None,
    device=None,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    """The exogenous driver ``A(t) in [0, 1]``.

    A smooth thermal cycle: ambient density rises and falls once per ``period``
    global steps.  It depends on a global clock only -- never on any agent --
    and it is **not** reset at episode boundaries, so a frozen-teammate rollout
    still sees the difficulty drift.

    Args:
        global_step: the persistent global step counter ``t``.
        batch_dim: number of vectorised worlds.
        period: ``T``, in global steps.
        phase_spread: give world ``b`` the fixed phase offset ``b / batch_dim``.
        freeze: if not None, return this constant (Phase-1 freeze knob).

    Returns:
        Tensor of shape ``(batch_dim,)`` with values in ``[0, 1]``.
    """
    if freeze is not None:
        return torch.full((batch_dim,), float(freeze), device=device, dtype=dtype)

    phase = torch.zeros(batch_dim, device=device, dtype=dtype)
    if phase_spread and batch_dim > 1:
        phase = torch.arange(batch_dim, device=device, dtype=dtype) / float(batch_dim)
    phase = phase + (float(global_step) / float(period))
    return 0.5 * (1.0 - torch.cos(TWO_PI * phase))


# ---------------------------------------------------------------------------
# The exertion functional  Phi  (declaration #1 of the PACT porting contract)
# ---------------------------------------------------------------------------


def angular_impulse(pos: Tensor, u: Tensor) -> Tensor:
    """The scalar each agent broadcasts: ``(p x u)_z``.

    Physically: the moment about the arena centre of the thrust vehicle ``j``
    pushes with, i.e. the rate at which it feeds angular momentum into the
    confined medium.  This is the *entire* PACT message -- one scalar per agent
    per step, computed by each agent from purely local quantities (its own
    position and its own executed command).

    Args:
        pos: ``(..., N, 2)`` agent positions.
        u: ``(..., N, 2)`` executed commands.

    Returns:
        ``(..., N)``
    """
    return pos[..., 0] * u[..., 1] - pos[..., 1] * u[..., 0]


def peer_mean(m: Tensor) -> Tensor:
    """``Phi_i = mean_{j != i} m_j`` -- the sum over the OTHER agents.

    This single function carries the category-C guarantee: with ``N == 1`` the
    sum is empty and the result is *identically zero*, so the liability can
    never be charged and the environment collapses back to the stationary task.
    The effect does not merely shrink at ``N == 1``; it disappears.

    Args:
        m: ``(..., N)`` the per-agent messages.

    Returns:
        ``(..., N)``
    """
    n_agents = m.shape[-1]
    if n_agents < 2:
        # Irreducibility certificate: no teammates -> no channel -> no liability.
        return torch.zeros_like(m)
    total = m.sum(dim=-1, keepdim=True)
    return (total - m) / float(n_agents - 1)


def leak_step(x2: Tensor, phi: Tensor, *, rho: float, gain: float) -> Tensor:
    """One step of the medium's leaky accumulator.

    ``x2(t+1) = rho * x2(t) + (1 - rho) * G * Phi(t)``

    Note that the driver does **not** appear here.  The circulation of the
    medium accumulates regardless of ambient density; density only sets how
    hard that circulation drags on a hull.  That is what makes the
    factorisation ``theta = c * x2`` exact for *any* driver path (theorem T1),
    rather than exact only to ``O((1-rho)|dc/dt| * window)``.
    """
    return rho * x2 + (1.0 - rho) * gain * phi


# ---------------------------------------------------------------------------
# The harm channel  g  and its inverse (declaration #3)
# ---------------------------------------------------------------------------


def rotate(v: Tensor, theta: Tensor) -> Tensor:
    """Rotate 2-D vectors by ``theta``.

    ``theta == 0`` returns ``v`` bit-for-bit (``cos 0 == 1.0``, ``sin 0 == 0.0``),
    which is what makes ``severity = 0`` an exact reduction to the base task and
    makes the Phase-1 transparency check pass by construction.

    Args:
        v: ``(..., 2)``
        theta: ``(...)`` broadcastable against ``v[..., 0]``.
    """
    cos_t = torch.cos(theta)
    sin_t = torch.sin(theta)
    x = v[..., 0]
    y = v[..., 1]
    return torch.stack((cos_t * x - sin_t * y, sin_t * x + cos_t * y), dim=-1)


def channel_inverse(a: Tensor, theta_hat: Tensor) -> Tensor:
    """The compensation law: pre-rotate the command by the estimated deflection.

    ``u = R(-theta_hat) a``.  With ``theta_hat == theta`` the medium delivers
    ``R(theta) R(-theta) a == a`` -- byte-identical to the stationary game
    (theorem T2), spending no magnitude at all.  The bounded resource is the
    action box: ``R(-theta_hat) a`` can leave ``[-u_range, u_range]^2`` and be
    clipped, and that clipping is what eventually caps sigma*.
    """
    return rotate(a, -theta_hat)


def wrap_angle(theta: Tensor) -> Tensor:
    """Wrap angles into ``(-pi, pi]`` -- for logging/diagnostics only."""
    return (theta + math.pi) % TWO_PI - math.pi


# ---------------------------------------------------------------------------
# The one learned scalar
# ---------------------------------------------------------------------------


def beta_from_w(w: Tensor, beta_max: float, mode: str = "affine") -> Tensor:
    """Map the policy's extra control dim ``w`` to the compensation gain ``beta``.

    ``affine`` is the natural choice when the host policy squashes actions into
    ``[-1, 1]`` (BenchMARL's PPO configs use ``use_tanh_normal: True``): it uses
    the whole range and starts at ``beta_max / 2``, i.e. the "direct mode"
    partial-compensation initialisation.  ``sigmoid`` is provided for hosts that
    emit unbounded actions.

    Args:
        w: ``(...)`` raw control dim.
        beta_max: upper bound on the gain, conventionally ``1.3 * peak c``.
        mode: ``"affine"`` or ``"sigmoid"``.

    Returns:
        ``(...)`` in ``[0, beta_max]``.
    """
    if mode == "affine":
        return beta_max * 0.5 * (w.clamp(-1.0, 1.0) + 1.0)
    if mode == "sigmoid":
        return beta_max * torch.sigmoid(w)
    raise ValueError(f"unknown beta mode {mode!r}, expected 'affine' or 'sigmoid'")


def beta_init_value(beta_max: float, mode: str = "affine") -> float:
    """The gain an untrained policy (``w == 0``) produces -- used to seed caches."""
    w = torch.zeros(())
    return float(beta_from_w(w, beta_max=beta_max, mode=mode))


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def per_step_cosine(a: Tensor, b: Tensor, eps: float = 1e-8) -> tuple[Tensor, Tensor]:
    """Per-step cosine between two agent-vectors, plus a validity mask.

    PACT's one hard gate.  ``a`` and ``b`` are ``(..., N)`` stacks of the same
    quantity across agents at a single step; the cosine of those two N-vectors
    certifies index order, reset masking and the one-step timing contract all at
    once.  Steps where either vector is numerically zero (episode starts, where
    the accumulator has not been charged yet) are reported as invalid rather
    than being folded in as spurious agreement.

    Deliberately *not* a correlation pooled over the driver's range: a pooled
    correlation reads ~0.95 off the varying-``c`` fan even when every single
    point satisfies ``theta = c * x2`` exactly.

    Returns:
        ``(cosine, valid_mask)``, both of shape ``(...)``.
    """
    na = a.norm(dim=-1)
    nb = b.norm(dim=-1)
    valid = (na > eps) & (nb > eps)
    cos = (a * b).sum(dim=-1) / (na * nb).clamp_min(eps)
    return cos, valid
