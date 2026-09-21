#  The arithmetic of the BASELINES.md baselines, with nothing but torch behind
#  it.
#
#  Everything here is a published formula transcribed from the reference code;
#  none of it touches torchrl, tensordict or vmas.  That is the point: it can be
#  imported and exercised on a machine that cannot run a training job, which is
#  what `baselines/verify.py` does.  The algorithm files hold the plumbing --
#  which tensor lives under which key, which optimiser steps what -- and this
#  file holds the parts that can be wrong silently.

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import torch


# ===========================================================================
#  HAPPO (B1)
# ===========================================================================


def block_bounds(total_calls: int, n_agents: int) -> List[int]:
    """Split ``total_calls`` optimiser calls into ``n_agents`` blocks.

    Returns the ``n_agents + 1`` boundaries, so block ``k`` is
    ``[bounds[k], bounds[k+1])``.  Even to within one call, and never empty as
    long as ``total_calls >= n_agents``.
    """
    return [round(k * total_calls / n_agents) for k in range(n_agents + 1)]


def happo_log_factor(
    log_weight: torch.Tensor, updated_mask: torch.Tensor
) -> torch.Tensor:
    """``log M`` for HAPPO's factor.

    HARL carries ``M = prod_{j already updated} pi_new_j / pi_old_j`` as a
    constant.  ``log_weight`` is ``log pi_new - log pi_old`` per agent, shaped
    ``(*batch, n_agents, 1)``; summing the log-ratios of exactly the agents in
    ``updated_mask`` gives the same product without the overflow a running
    product invites.

    Returns ``(*batch, 1, 1)``, which broadcasts against the advantage.
    """
    return (log_weight.squeeze(-1) * updated_mask).sum(-1, keepdim=True).unsqueeze(-1)


# ===========================================================================
#  LCPO (B6) -- core_trpo.conjugate_gradients and core_lcpo.get_qu
# ===========================================================================


def conjugate_gradients(
    avp: Callable[[torch.Tensor], torch.Tensor],
    b: torch.Tensor,
    nsteps: int,
    residual_tol: float = 1e-10,
) -> torch.Tensor:
    """Solve ``A x = b`` for a positive-definite ``A`` given only ``A @ v``."""
    x_k = torch.zeros_like(b)
    resid = b.clone()
    d_k = b.clone()
    r_dot_r = torch.dot(resid, resid)
    for _ in range(nsteps):
        q_dk = avp(d_k)
        denom = torch.dot(d_k, q_dk)
        if float(denom) == 0.0:
            break
        alpha = r_dot_r / denom
        x_k = x_k + alpha * d_k
        resid = resid - alpha * q_dk
        new_r_dot_r = torch.dot(resid, resid)
        beta = new_r_dot_r / r_dot_r
        d_k = resid + beta * d_k
        r_dot_r = new_r_dot_r
        if float(r_dot_r) < residual_tol:
            break
    return x_k


def get_qu(a: float, b: float, c: float):
    """The two roots of ``a s^2 + b s + c``, in LCPO's order."""
    sqr = (b**2 - 4 * a * c) ** 0.5
    return (-b + sqr) / 2 / a, (-b - sqr) / 2 / a


def gaussian_kl(loc_o, scale_o, loc_n, scale_n) -> torch.Tensor:
    """``KL(N(loc_o, scale_o) || N(loc_n, scale_n))``, summed over the last dim.

    Used as the KL between two TanhNormal policies as well: the tanh-and-affine
    map onto the action box is a fixed bijection shared by both distributions,
    and KL is invariant under a common bijection, so this is an identity rather
    than an approximation.
    """
    var_o, var_n = scale_o.pow(2), scale_n.pow(2)
    kl = (scale_n / scale_o).log() + (var_o + (loc_o - loc_n).pow(2)) / (2 * var_n) - 0.5
    return kl.sum(-1)


def categorical_kl(probs_old: torch.Tensor, probs_new: torch.Tensor) -> torch.Tensor:
    """``KL(p_old || p_new)`` for categorical policies, summed over actions."""
    return (
        probs_old * ((probs_old + 1e-20).log() - (probs_new + 1e-20).log())
    ).sum(-1)


class OutOfDistributionSampler:
    """LCPO's ``buffer_ood.OutOfDSampler``, in torch and over JOINT states.

    A reservoir over every state the agent has seen, plus a FIFO window of the
    recent past.  ``get`` returns reservoir states whose CONTEXT is far from the
    recent window, and returns ``None`` if it cannot fill a batch in five tries
    -- which is what makes LCPO fall back to plain policy gradient early in
    training, before any context has changed.

    A "state" is the joint observation of one environment step, shape
    ``(n_agents, obs_dim)``: it keeps the sampled batch in the shape a per-agent
    policy expects, and makes the distance a distance between situations rather
    than between agents.
    """

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        window: int,
        capacity: int,
        context_slice: slice,
        threshold: float,
        device,
        generator: torch.Generator,
    ):
        self.cap = int(capacity)
        self.win = max(int(window), 1)
        self.context_slice = context_slice
        self.threshold = float(threshold)
        self.generator = generator
        self.states = torch.zeros(self.cap, n_agents, obs_dim, device=device)
        self.recent = torch.zeros(self.win, n_agents, obs_dim, device=device)
        self.n_seen = 0
        self.i_win = 0

    def is_distant(self, data: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        """``env.is_different`` with ``dist_func_type='l2'``."""
        mu_base = base[..., self.context_slice].mean(dim=0).reshape(-1)
        mu_data = data[..., self.context_slice].reshape(data.shape[0], -1)
        return ((mu_data - mu_base) ** 2).sum(dim=-1) > self.threshold

    def add(self, states: torch.Tensor) -> None:
        """``add_many_exp``: reservoir sampling, plus the recent FIFO."""
        for k in range(states.shape[0]):
            if self.n_seen < self.cap:
                self.states[self.n_seen] = states[k]
            else:
                idx = int(
                    torch.randint(
                        self.n_seen + 1, (1,), generator=self.generator
                    ).item()
                )
                if idx < self.cap:
                    self.states[idx] = states[k]
            self.recent[self.i_win] = states[k]
            self.i_win = (self.i_win + 1) % self.win
            self.n_seen += 1

    def get(self, batch_size: int) -> Optional[torch.Tensor]:
        if self.n_seen == 0:
            return None
        recents = self.recent[: min(self.n_seen, self.win)]
        alls = self.states[: min(self.n_seen, self.cap)]
        out: List[torch.Tensor] = []
        collected = 0
        for _ in range(5):
            idx = torch.randint(
                alls.shape[0], (batch_size,), generator=self.generator
            ).to(alls.device)
            candidates = alls[idx]
            keep = candidates[self.is_distant(candidates, recents)]
            if keep.shape[0]:
                out.append(keep)
                collected += keep.shape[0]
            if collected >= batch_size:
                break
        if collected < batch_size:
            return None
        return torch.cat(out, dim=0)[:batch_size]


# ===========================================================================
#  LIAM (B5)
# ===========================================================================


def others_view(tensor: torch.Tensor, n_agents: int) -> torch.Tensor:
    """``(*batch, n_agents, d)`` -> ``(*batch, n_agents, (n_agents-1) * d)``.

    Row ``i`` holds every agent's value except ``i``'s own, in increasing agent
    order.  This is LIAM's ``modelled_agent_obs`` / ``modelled_agent_act``: the
    controlled agent models everybody else, and never itself.
    """
    lead = tensor.shape[:-2]
    d = tensor.shape[-1]
    flat = tensor.reshape(-1, n_agents, d)
    idx = torch.stack(
        [
            torch.tensor([j for j in range(n_agents) if j != i], device=tensor.device)
            for i in range(n_agents)
        ]
    )
    gathered = flat[:, idx]
    return gathered.reshape(*lead, n_agents, (n_agents - 1) * d)


# ===========================================================================
#  MF-AC (B4)
# ===========================================================================


def mean_action(actions: torch.Tensor, include_self: bool) -> torch.Tensor:
    """``abar^j``, shaped like ``actions``.

    ``include_self=False`` is the paper's mean over ``N(j)``, which excludes
    ``j``.  ``True`` is the reference implementation's ``former_act_prob``: one
    mean over the whole team, tiled to every agent.
    """
    n_agents = actions.shape[-2]
    total = actions.sum(dim=-2, keepdim=True)
    if include_self:
        return (total / n_agents).expand_as(actions)
    if n_agents < 2:
        return torch.zeros_like(actions)
    return (total - actions) / (n_agents - 1)


# ===========================================================================
# ===========================================================================
#  THE EXTRA BASELINES (X1 .. X6).  See baselines/README_EXTRA.md.
#
#  Same rule as everything above: a published formula, transcribed from the
#  reference, with nothing but torch behind it, so `baselines/verify.py` can
#  check it against its definition on a laptop.
# ===========================================================================
# ===========================================================================


# ===========================================================================
#  X1  QCD+ / RR -- "Is Prior-Free Black-Box Non-Stationary RL Feasible?"
#      Gerogiannis, Huang, Veeravalli, arXiv 2410.13772.
#      The detector is the Bernoulli GLR of Besson, Kaufmann, Maillard &
#      Seznec, JMLR 2022 ("Efficient Change-Point Detection for Tackling
#      Piecewise-Stationary Bandits"), which 2410.13772's Algorithm 3 plugs in.
# ===========================================================================


def bernoulli_kl(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """``kl(p, q)`` for Bernoulli means, the GLR's per-segment divergence.

    ``p log(p/q) + (1-p) log((1-p)/(1-q))``, with both arguments pushed off the
    boundary so that a segment whose observations are all 0 or all 1 -- which
    happens on the very first samples -- gives a finite statistic rather than
    an infinite one that fires the detector immediately.

    ``eps`` is ``1e-6`` and not something smaller for a reason that only shows
    up in single precision: ``1 - 1e-12`` rounds to exactly ``1.0`` in float32,
    the clamp does nothing, and ``(1-p)/(1-q)`` divides by zero -- so the very
    first all-zero segment would return ``inf`` and declare a change.  The
    reference bandit implementations guard at the same order.
    """
    p = p.clamp(eps, 1.0 - eps)
    q = q.clamp(eps, 1.0 - eps)
    return p * (p / q).log() + (1.0 - p) * ((1.0 - p) / (1.0 - q)).log()


def glr_threshold(n: int, delta: float) -> float:
    """``beta(n, delta) = log(4 n sqrt(n) / delta)``.

    The threshold 2410.13772 names for its Bernoulli GLR detector, and Besson
    et al.'s Theorem 1 threshold.  ``delta`` is the false-alarm probability;
    the paper asks only for ``delta`` of order ``1/poly(T)``.
    """
    n = max(int(n), 1)
    return math.log(4.0 * n * math.sqrt(n) / float(delta))


def glr_statistic(samples: torch.Tensor) -> torch.Tensor:
    """The Bernoulli GLR statistic of a ``[0, 1]``-valued stream.

    ``max_{1 <= s < n} [ s kl(mu_{1:s}, mu_{1:n}) + (n-s) kl(mu_{s+1:n}, mu_{1:n}) ]``

    the generalised likelihood ratio for "one change point somewhere" against
    "no change".  Computed in ``O(n)`` from the prefix sums rather than the
    ``O(n^2)`` double loop, which is the same number.
    """
    n = samples.shape[0]
    if n < 2:
        return torch.zeros((), device=samples.device, dtype=samples.dtype)
    total = samples.sum()
    mu_all = total / n
    prefix = samples.cumsum(0)[:-1]                       # sums of 1..s, s<n
    s = torch.arange(1, n, device=samples.device, dtype=samples.dtype)
    mu_pre = prefix / s
    mu_post = (total - prefix) / (n - s)
    stat = s * bernoulli_kl(mu_pre, mu_all) + (n - s) * bernoulli_kl(
        mu_post, mu_all
    )
    return stat.max()


def master_can_fire(horizon: float) -> bool:
    """Can MASTER's non-stationarity test EVER cross its threshold at ``T``?

    2410.13772's Theorem 4.  Both of MASTER's tests compare an average of
    rewards in ``[0, 1]`` -- so a statistic bounded by 1 -- against a threshold
    proportional to ``54 (log2 T + 1) log(T / delta) rho(.)``.  Rearranged, a
    crossing needs ``delta >= T exp(-sqrt(T) / (54 (log2 T + 1)))``, and since
    a probability is below 1 that is only possible once

        sqrt(T) > 54 (log2(T) + 1) log(T).

    Below that horizon MASTER cannot declare a change however non-stationary
    the problem is, and it degenerates into restarting at random -- which is
    the paper's whole point, and the reason the baseline that is actually run
    here is the quickest-change-detection one.
    """
    T = float(horizon)
    if T <= 2.0:
        return False
    return math.sqrt(T) > 54.0 * (math.log2(T) + 1.0) * math.log(T)


def master_min_horizon() -> float:
    """The smallest horizon at which MASTER's tests can fire, by bisection.

    Reported in the paper as "T must be at least 1.24 billion".
    """
    lo, hi = 2.0, 1e18
    for _ in range(400):
        mid = (lo + hi) / 2.0
        if master_can_fire(mid):
            hi = mid
        else:
            lo = mid
    return hi


class GlrDetector:
    """Algorithm 3's ``D``: a Bernoulli GLR test with a full restart on alarm.

    One scalar stream in, one boolean out.  ``observe`` appends a sample,
    recomputes the statistic and reports whether the threshold was crossed;
    ``reset`` is the reference's ``H_B <- {}``.

    The stream must be ``[0, 1]``-valued, which is what the detector is defined
    for.  Mapping a return onto that interval is the caller's job and is stated
    in ``baselines/docs/qcd.md``; it is the one adaptation this detector needs.
    """

    def __init__(self, delta: float, min_samples: int, max_samples: int):
        self.delta = float(delta)
        self.min_samples = max(int(min_samples), 2)
        self.max_samples = max(int(max_samples), self.min_samples)
        self.samples: List[float] = []
        self.n_alarms = 0
        self.last_stat = 0.0
        self.last_threshold = float("inf")

    def reset(self) -> None:
        self.samples = []
        self.last_stat = 0.0
        self.last_threshold = float("inf")

    def observe(self, value: float) -> bool:
        self.samples.append(float(value))
        if len(self.samples) > self.max_samples:
            #  The reference keeps the whole post-restart history.  Bounded
            #  here so a 3M-frame run cannot grow the statistic's cost without
            #  limit; at the launcher's defaults the cap is never reached
            #  between restarts.  See baselines/docs/qcd.md.
            self.samples = self.samples[-self.max_samples :]
        n = len(self.samples)
        if n < self.min_samples:
            self.last_stat = 0.0
            self.last_threshold = float("inf")
            return False
        stat = float(glr_statistic(torch.tensor(self.samples, dtype=torch.float64)))
        threshold = glr_threshold(n, self.delta)
        self.last_stat = stat
        self.last_threshold = threshold
        if stat > threshold:
            self.n_alarms += 1
            self.reset()
            return True
        return False


class RandomRestartSchedule:
    """Algorithm 2's restart times: i.i.d. ``Geometric(eta_r)``.

    2410.13772's "random restarting" baseline, the one Theorem 4 says MASTER
    degenerates into.  Sampled by inverse transform so the draw is
    reproducible from a torch generator alongside everything else.
    """

    def __init__(self, eta: float, generator: torch.Generator):
        self.eta = float(eta)
        self.generator = generator
        self.n_restarts = 0
        self.countdown = self._draw()

    def _draw(self) -> int:
        if not (0.0 < self.eta <= 1.0):
            raise ValueError(f"Geometric needs eta in (0, 1]; got {self.eta}")
        if self.eta >= 1.0:
            return 1
        u = float(torch.rand((), generator=self.generator))
        return max(int(math.ceil(math.log(1.0 - u) / math.log(1.0 - self.eta))), 1)

    def step(self) -> bool:
        self.countdown -= 1
        if self.countdown <= 0:
            self.countdown = self._draw()
            self.n_restarts += 1
            return True
        return False


# ===========================================================================
#  X2  DEDA-FP -- "Solving Continuous Mean Field Games: Deep RL for
#      Non-Stationary Dynamics", Magnino, Shao, Wu, Shen, Lauriere,
#      arXiv 2510.22158 (NeurIPS 2025).
# ===========================================================================


def gaussian_nll(loc, scale, target) -> torch.Tensor:
    """``-log N(target; loc, scale)``, summed over the action dimensions.

    DEDA-FP's supervised averaging step:
    ``L_NLL(thetabar) = -1/M sum_i log N(a_i; mu(s_i, t_i), sigma(s_i, t_i))``.
    The averaged policy is a Gaussian fitted by maximum likelihood to the
    actions every past best response took, which is what makes it the
    fictitious-play average of those POLICIES rather than an average of their
    parameters.
    """
    var = scale.pow(2)
    return (
        0.5 * math.log(2.0 * math.pi)
        + scale.log()
        + (target - loc).pow(2) / (2.0 * var)
    ).sum(-1)


def affine_coupling(
    x: torch.Tensor, shift: torch.Tensor, log_scale: torch.Tensor, mask: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """One conditional affine coupling layer, and its exact log-determinant.

    ``y = m * x + (1 - m) * (x * exp(s) + t)``: the masked-in half passes
    through untouched and conditions ``s`` and ``t``, the masked-out half is
    scaled and shifted.  The Jacobian is triangular, so

        log |det dy/dx| = sum over the transformed coordinates of s.

    This is the normalizing-flow layer DEDA-FP's conditional density model is
    built out of here.  The reference uses autoregressive neural spline flows;
    an affine coupling flow is the same object -- an exactly invertible map
    with a tractable log-determinant, trained by maximum likelihood -- with a
    simpler elementwise transform.  Stated as an adaptation in
    baselines/docs/dedafp.md.
    """
    keep = mask
    change = 1.0 - mask
    y = keep * x + change * (x * log_scale.exp() + shift)
    log_det = (change * log_scale).sum(-1)
    return y, log_det


def affine_coupling_inverse(
    y: torch.Tensor, shift: torch.Tensor, log_scale: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """The exact inverse of :func:`affine_coupling`, for the same shift/scale."""
    keep = mask
    change = 1.0 - mask
    return keep * y + change * ((y - shift) * (-log_scale).exp())


def standard_normal_log_prob(z: torch.Tensor) -> torch.Tensor:
    """``log N(z; 0, I)``, summed over the last dimension: the flow's base."""
    return (-0.5 * z.pow(2) - 0.5 * math.log(2.0 * math.pi)).sum(-1)


# ===========================================================================
#  X3  IPGA / INPG -- "Independent Learning in Performative Markov Potential
#      Games", Sahitaj, Sasnauskas, Yalin, Mandal, Radanovic, arXiv 2504.20593.
# ===========================================================================


def categorical_l2_sq(probs_p: torch.Tensor, probs_q: torch.Tensor) -> torch.Tensor:
    """Squared Euclidean distance between two distributions over a finite set.

    IPGA's proximal term, exactly as the paper writes it: the squared Euclidean
    distance between the two probability VECTORS.
    """
    return (probs_p - probs_q).pow(2).sum(-1)


def gaussian_l2_sq(loc_p, scale_p, loc_q, scale_q) -> torch.Tensor:
    """The squared L2 distance between two diagonal Gaussian DENSITIES.

    The continuous-action reading of IPGA's proximal term.  For a finite action
    set the paper's distance is the Euclidean norm of the difference of the
    probability vectors; its continuous counterpart is the ``L2(da)`` norm of
    the difference of the two densities, and for Gaussians that has a closed
    form, because a product of two Gaussian densities integrates to a Gaussian
    density evaluated at the difference of the means:

        int p^2 = N(0; 0, 2 Sigma_p)
        int q^2 = N(0; 0, 2 Sigma_q)
        int pq  = N(mu_p - mu_q; 0, Sigma_p + Sigma_q)

    so the distance is ``int p^2 + int q^2 - 2 int pq``.  Each term is a
    product over the (independent) action dimensions.  No sampling, no
    estimator: this is the quantity itself.
    """

    def _normal_density_at(diff, var):
        #  prod_d N(diff_d; 0, var_d), computed in log space.
        log_density = (
            -0.5 * math.log(2.0 * math.pi)
            - 0.5 * var.log()
            - 0.5 * diff.pow(2) / var
        )
        return log_density.sum(-1).exp()

    var_p, var_q = scale_p.pow(2), scale_q.pow(2)
    zero = torch.zeros_like(loc_p)
    pp = _normal_density_at(zero, 2.0 * var_p)
    qq = _normal_density_at(zero, 2.0 * var_q)
    pq = _normal_density_at(loc_p - loc_q, var_p + var_q)
    return pp + qq - 2.0 * pq


def inpg_multiplicative_update(
    probs_old: torch.Tensor, advantage: torch.Tensor, eta: float, gamma: float
) -> torch.Tensor:
    """INPG's exact update for a finite action set.

    ``pi_new(a|s) proportional to pi_old(a|s) exp(eta/(1-gamma) Abar(s, a))``

    -- the unregularised independent natural policy gradient of 2504.20593,
    which for the softmax parameterisation IS natural policy gradient.  Written
    out here so the continuous-action version used by the algorithm (a
    Fisher-preconditioned step solved by conjugate gradients) can be checked
    against the closed form it generalises.
    """
    logits = probs_old.clamp_min(1e-20).log() + (eta / (1.0 - gamma)) * advantage
    return torch.softmax(logits, dim=-1)


# ===========================================================================
#  X4  WISDOM -- "Wavelet Predictive Representations for Non-Stationary
#      Reinforcement Learning", Wang, Li, He, Li, Bennis, Islam, Wang,
#      arXiv 2510.04507.  Code: MinWangcs/WISDOM.
# ===========================================================================


def product_of_gaussians(mus: torch.Tensor, sigmas_squared: torch.Tensor):
    """PEARL's permutation-invariant posterior, WISDOMAgent._product_of_gaussians.

    ``q(z|c)`` is the product of one Gaussian per context element, which is
    Gaussian with

        sigma^2 = 1 / sum_i sigma_i^-2 ,    mu = sigma^2 sum_i mu_i sigma_i^-2 .

    The product is over the CONTEXT dimension, which the reference puts at
    ``dim=0`` after a view; here it is the second-to-last so the leading
    dimensions can be batch and agent.
    """
    sigmas_squared = sigmas_squared.clamp_min(1e-7)
    sigma_squared = 1.0 / torch.reciprocal(sigmas_squared).sum(dim=-2)
    mu = sigma_squared * (mus / sigmas_squared).sum(dim=-2)
    return mu, sigma_squared


def gaussian_kl_to_standard_normal(mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """``KL(N(mu, var) || N(0, I))``, summed over the latent dimensions.

    ``WISDOMAgent.compute_kl_div``, which builds the same quantity out of
    ``torch.distributions.kl`` against a unit prior.
    """
    return 0.5 * (var + mu.pow(2) - 1.0 - var.clamp_min(1e-12).log()).sum(-1)


def wavelet_forward_fading(
    x: torch.Tensor,
    h0: torch.Tensor,
    h1: torch.Tensor,
    w: torch.Tensor,
    depth: int,
    kernel_size: int,
):
    """WISDOM's ``forward_fading``: a learnable a-trous wavelet decomposition.

    ``x`` is ``(batch, d_model, L)``.  At each level the signal is convolved
    with a learnable low-pass ``h0`` and high-pass ``h1`` at a doubling
    dilation, with causal (left) padding, giving an approximation and a detail
    band; the output is the learned weighted sum of every detail band, the
    final approximation band, and the input itself.

    Returns the multi-scale representation and the FINAL approximation
    coefficient, which is the one the wavelet TD operator acts on.
    Transcribed from ``wisdom/networks.py``.
    """
    res_lo = x
    y = torch.zeros_like(x)
    dilation = 1
    for i in range(depth, 0, -1):
        padding = dilation * (kernel_size - 1)
        res_lo_pad = torch.nn.functional.pad(res_lo, (padding, 0), "constant", 0)
        res_hi = torch.nn.functional.conv1d(
            res_lo_pad, h1, dilation=dilation, groups=x.shape[1]
        )
        res_lo = torch.nn.functional.conv1d(
            res_lo_pad, h0, dilation=dilation, groups=x.shape[1]
        )
        y = y + w[:, i : i + 1] * res_hi
        dilation *= 2
    y = y + w[:, :1] * res_lo
    y = y + x * w[:, -1:]
    return y, res_lo


def wavelet_td_target(
    z: torch.Tensor, next_res_lo: torch.Tensor, gamma: float
) -> torch.Tensor:
    """WISDOM's wavelet TD operator: ``target = z_t + gamma * res_lo(z_{t+1})``.

    ``ReconstructionTrainer.training_step``:
    ``target_res_lo = task_z + self.gamma * next_res_lo``, with ``next_res_lo``
    taken from the TARGET wavelet network and the loss the RMS error against
    it.  Its fixed point is ``res_lo(z_t) = sum_k gamma^k z_{t+k}`` -- the
    discounted future of the task latent, which is what makes the
    representation predictive rather than merely a filter.
    """
    return z + gamma * next_res_lo


# ===========================================================================
#  X5  DORAEMON -- "Domain Randomization via Entropy Maximization",
#      Tiboni, Klink, Peters, Tommasi, D'Eramo, Chalvatzaki, ICLR 2024
#      (arXiv 2311.01885).  Code: gabrieletiboni/doraemon.
# ===========================================================================


def _f64(x) -> torch.Tensor:
    """``x`` as a float64 tensor, KEEPING the autograd graph if it has one.

    Every Beta quantity below is computed in double precision, and that is not
    fastidiousness.  ``KL(Beta(100, 100) || Beta(99.994, 99.994))`` is 9e-10;
    in float32 the same expression evaluates to -4e-5 -- negative, for a
    divergence -- because it is a difference of log-Gammas of order 400.  A
    constraint function with 1e-5 of noise makes the trust-region solver's
    finite differences meaningless and DORAEMON's distribution never moves.
    """
    if isinstance(x, torch.Tensor):
        return x.to(torch.float64)
    return torch.tensor(float(x), dtype=torch.float64)


def beta_entropy(a, b, low: float, high: float) -> torch.Tensor:
    """The entropy of ``Beta(a, b)`` rescaled onto ``[low, high]``.

    ``DomainRandDistribution.entropy`` with ``standardize=False``: the entropy
    of the standard Beta plus ``log(M - m)`` for the change of variables.  This
    is the quantity DORAEMON maximises.  Returns a 0-dim float64 tensor, so it
    is differentiable in ``a`` and ``b``; call ``float()`` on it for a number.
    """
    dist = torch.distributions.Beta(_f64(a), _f64(b))
    return dist.entropy() + math.log(max(float(high) - float(low), 1e-12))


def beta_kl(a_p, b_p, a_q, b_q) -> torch.Tensor:
    """``KL(Beta(a_p, b_p) || Beta(a_q, b_q))``, as a 0-dim float64 tensor.

    ``DomainRandDistribution.kl_divergence``.  The support cancels -- the
    reference notes "KL does not depend on loc params [m, M]" -- so this is the
    KL of the standard Betas.  It is both DORAEMON's objective (against the
    target) and its trust region (against the current distribution), and it is
    differentiable in whichever of the four arguments carries a graph.
    """
    p = torch.distributions.Beta(_f64(a_p), _f64(b_p))
    q = torch.distributions.Beta(_f64(a_q), _f64(b_q))
    return torch.distributions.kl.kl_divergence(p, q)


def beta_log_pdf(x: torch.Tensor, a: float, b: float, low: float, high: float):
    """The STANDARDISED log-density of a rescaled Beta at ``x``.

    ``DomainRandDistribution._univariate_pdf`` with ``standardize=True``: the
    sample is mapped back to ``[0, 1]`` and the density is the standard Beta's
    there.  The reference's ``standardize`` flag drops the ``-log(M-m)``
    Jacobian because it cancels in the importance ratio, and that is exactly
    how it is used -- so it is dropped here too, and the name says so.
    """
    span = max(float(high) - float(low), 1e-12)
    u = ((_f64(x) - float(low)) / span).clamp(1e-6, 1.0 - 1e-6)
    dist = torch.distributions.Beta(_f64(a), _f64(b))
    return dist.log_prob(u)


def importance_ratio(
    x: torch.Tensor,
    proposed: Tuple[float, float],
    current: Tuple[float, float],
    low: float,
    high: float,
) -> torch.Tensor:
    """The density ratio DORAEMON's performance constraint is weighted by.

    ``torch.exp(proposed.pdf(log=True) - current.pdf(log=True))`` in
    ``performance_constraint_fn``: the success rate under a PROPOSED
    distribution is estimated from episodes that were run under the CURRENT
    one, so nothing has to be re-collected to test a candidate.
    """
    return (
        beta_log_pdf(x, proposed[0], proposed[1], low, high)
        - beta_log_pdf(x, current[0], current[1], low, high)
    ).exp()


def doraemon_success_rate(
    values: torch.Tensor, threshold: float, weights: torch.Tensor = None
) -> torch.Tensor:
    """The expected success rate, optionally importance-weighted.

    ``train_success_rate = (values >= task_solved_threshold).float().mean()``
    and, under a proposed distribution,
    ``mean(importance_sampling * (values >= condition))``.
    """
    indicator = (values >= float(threshold)).to(values.dtype)
    if weights is None:
        return indicator.mean()
    return (weights * indicator).mean()


def sigmoid_bounds(x: torch.Tensor, low: float, high: float) -> torch.Tensor:
    """``DomainRandDistribution.sigmoid``: map the reals onto ``(low, high)``.

    DORAEMON optimises the Beta parameters through a sigmoid so the optimiser
    is unconstrained while ``a`` and ``b`` stay inside their bounds.
    """
    return low + (high - low) * torch.sigmoid(x)


def inv_sigmoid_bounds(y, low: float, high: float):
    """The inverse of :func:`sigmoid_bounds`, for the optimiser's start point."""
    span = max(float(high) - float(low), 1e-12)
    u = (torch.as_tensor(y, dtype=torch.float64) - low) / span
    u = u.clamp(1e-6, 1.0 - 1e-6)
    return (u / (1.0 - u)).log()


# ===========================================================================
#  X6  M3W -- "Learning and Planning Multi-Agent Tasks via an MoE-based World
#      Model", Zhao, Zhao, Xu, Fu, Chai, Zhu, Zhao, NeurIPS 2025.
#      Code: zhaozijie2022/m3w-marl (m3w/models/world_models.py,
#      m3w/runners/world_model_runner.py).
# ===========================================================================


def simnorm(x: torch.Tensor, dim: int) -> torch.Tensor:
    """M3W's ``SimNorm``: a softmax over every group of ``dim`` coordinates.

    The latent is cut into simplices, which bounds it and keeps the dynamics
    model's rollout from drifting off to infinity over a planning horizon.
    Straight out of ``world_models.py``.
    """
    shape = x.shape
    x = x.view(*shape[:-1], -1, dim)
    x = torch.softmax(x, dim=-1)
    return x.view(*shape)


def sym_log(x: torch.Tensor) -> torch.Tensor:
    """``sign(x) log(1 + |x|)``: M3W's symmetric log for the two-hot bins."""
    return torch.sign(x) * torch.log1p(x.abs())


def sym_exp(x: torch.Tensor) -> torch.Tensor:
    """The inverse of :func:`sym_log`."""
    return torch.sign(x) * (x.abs().exp() - 1.0)


def two_hot_encode(
    x: torch.Tensor, bins: torch.Tensor, vmin: float, vmax: float
) -> torch.Tensor:
    """``TwoHotProcessor.scalar_encode_logits``: a scalar as a soft two-hot row.

    The value is symlog-transformed, clamped into ``[vmin, vmax]``, and split
    between the two neighbouring bins in proportion to where it falls between
    them.  ``x`` is ``(..., 1)`` and the result is ``(..., num_bins)``.
    """
    num_bins = bins.shape[-1]
    bin_size = (vmax - vmin) / (num_bins - 1)
    value = sym_log(x).clamp(vmin, vmax).squeeze(-1)
    idx = torch.floor((value - vmin) / bin_size).long().clamp(0, num_bins - 1)
    offset = ((value - vmin) / bin_size - idx.to(value.dtype)).unsqueeze(-1)
    out = torch.zeros(*value.shape, num_bins, device=x.device, dtype=x.dtype)
    out.scatter_(-1, idx.unsqueeze(-1), 1.0 - offset)
    out.scatter_add_(
        -1, ((idx + 1) % num_bins).unsqueeze(-1), offset
    )
    return out


def two_hot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """``TwoHotProcessor.logits_decode_scalar``: bin logits back to a scalar."""
    weights = torch.softmax(logits, dim=-1)
    return sym_exp((weights * bins).sum(-1, keepdim=True))


def two_hot_loss(
    logits: torch.Tensor, target: torch.Tensor, bins: torch.Tensor, vmin: float, vmax: float
) -> torch.Tensor:
    """``TwoHotProcessor.dis_reg_loss``: cross-entropy against the two-hot target."""
    log_pred = torch.log_softmax(logits, dim=-1)
    two_hot = two_hot_encode(target, bins, vmin, vmax)
    return -(two_hot * log_pred).sum(-1, keepdim=True)


def soft_moe(
    x: torch.Tensor, phi: torch.Tensor, experts
) -> torch.Tensor:
    """M3W's ``CenMoEDynamicsModel.predict``: Soft Mixture-of-Experts routing.

    ``x`` is ``(batch, n_tokens, d)`` -- one token per agent.  ``phi`` is
    ``(d, n_experts, n_slots)``.  Each expert's slot receives a softmax-weighted
    AVERAGE over tokens (the dispatch), every expert runs on its own slots, and
    each token reads back a softmax-weighted average over all slots (the
    combine).  No token is dropped and no expert is empty, which is what makes
    it "soft".

    ``experts`` is a sequence of callables, one per expert, each mapping
    ``(batch, n_slots, d)`` to ``(batch, n_slots, d_out)``.
    """
    weights = torch.einsum("bnd,des->bnes", x, phi)
    dispatch = torch.softmax(weights, dim=1)                # over TOKENS
    expert_inputs = torch.einsum("bnes,bnd->besd", dispatch, x)
    expert_outputs = torch.stack(
        [experts[i](expert_inputs[:, i]) for i in range(len(experts))], dim=1
    )                                                        # (b, e, s, d_out)
    b, e, s, d_out = expert_outputs.shape
    expert_outputs = expert_outputs.reshape(b, e * s, d_out)
    combine = torch.softmax(weights.reshape(*weights.shape[:2], -1), dim=-1)
    return torch.einsum("bnz,bzd->bnd", combine, expert_outputs)


def cv_squared(x: torch.Tensor) -> torch.Tensor:
    """``NoisyTopKRouter.cv_squared``: the squared coefficient of variation.

    Zero when the load is perfectly balanced across experts; the router's
    auxiliary loss is this on the gate mass plus this on the load plus the
    router z-loss.
    """
    if x.shape[0] == 1:
        return torch.zeros((), device=x.device, dtype=x.dtype)
    return x.float().var() / (x.float().mean() ** 2 + 1e-10)


def router_z_loss(logits: torch.Tensor) -> torch.Tensor:
    """``NoisyTopKRouter.z_loss``: keeps the router logits from growing."""
    return torch.log(torch.exp(logits).sum(-1)).mean()


def mppi_update(
    elite_values: torch.Tensor,
    elite_actions: torch.Tensor,
    temperature: float,
    min_std: float,
    max_std: float,
):
    """M3W's MPPI moment update, ``plan()``'s inner loop.

        score = softmax(temperature * (value - max value))
        mean  = sum_k score_k a_k
        std   = sqrt( sum_k score_k (a_k - mean)^2 ), clamped

    ``elite_values`` is ``(*batch, n_elites)`` and ``elite_actions`` is
    ``(horizon, *batch, n_elites, action_dim)``.  The max is subtracted for
    numerical stability, exactly as the reference does it, and the returned
    ``score`` is the categorical the executed action is drawn from.
    """
    max_value = elite_values.max(dim=-1, keepdim=True)[0]
    score = torch.exp(temperature * (elite_values - max_value))
    score = score / score.sum(dim=-1, keepdim=True)
    weights = score.unsqueeze(0).unsqueeze(-1)
    mean = (weights * elite_actions).sum(dim=-2)
    var = (weights * (elite_actions - mean.unsqueeze(-2)) ** 2).sum(dim=-2)
    std = torch.sqrt(var + 1e-6).clamp(min_std, max_std)
    return mean, std, score


def nstep_return(
    rewards: torch.Tensor, gamma: float, dones: torch.Tensor = None
):
    """The truncated ``n``-step return and the discount that survives it.

    ``rewards`` is ``(..., n)`` in time order.  Returns
    ``sum_k gamma^k r_k`` and ``gamma^n`` with every factor after the first
    termination zeroed, which is what makes the bootstrap after a terminal
    state disappear.  M3W's buffer precomputes exactly this pair
    (``nstep_reward``, ``nstep_gamma``) and its critic target is
    ``nstep_reward + nstep_gamma * Q'(s_{t+n}, a') (1 - term)``.
    """
    n = rewards.shape[-1]
    if dones is None:
        alive = torch.ones_like(rewards)
    else:
        #  alive_k = 1 only while no done has been seen strictly before k
        alive = (1.0 - dones).cumprod(dim=-1)
        alive = torch.cat(
            [torch.ones_like(alive[..., :1]), alive[..., :-1]], dim=-1
        )
    powers = gamma ** torch.arange(
        n, device=rewards.device, dtype=rewards.dtype
    )
    ret = (rewards * powers * alive).sum(-1, keepdim=True)
    tail = (gamma**n) * alive[..., -1:] * (
        torch.ones_like(alive[..., -1:])
        if dones is None
        else (1.0 - dones[..., -1:])
    )
    return ret, tail
