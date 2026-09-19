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

from typing import Callable, List, Optional

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
