#  M3W -- learning and planning multi-agent tasks via an MoE-based world model.
#
#      Zhao, Zhao, Xu, Fu, Chai, Zhu, Zhao, "Learning and Planning Multi-Agent
#      Tasks via a MoE-based World Model", NeurIPS 2025.
#      Code: zhaozijie2022/m3w-marl (m3w/models/world_models.py,
#      m3w/runners/world_model_runner.py, m3w/algorithms/{actors,critics}).
#      Built on HARL, TDMPC2 and Light-SoftMoE.
#
#  The only MODEL-BASED baseline in this table, and the only one that chooses
#  its action by PLANNING rather than by evaluating a policy.  Five pieces:
#
#    1. A per-agent observation encoder into a SimNorm latent -- the latent is
#       cut into simplices, which bounds it so a multi-step rollout cannot
#       drift off to infinity.
#    2. A CENTRALISED SoftMoE dynamics model over the agents' latents: every
#       agent is a token, each expert's slot receives a softmax-weighted
#       average over tokens, and each token reads back a softmax-weighted
#       average over slots.  Nothing is dropped, no expert is empty.
#    3. A CENTRALISED SparseMoE reward model: a noisy top-k router over the
#       joint (z, a), self-attention experts across the agent tokens, and a
#       two-hot distributional reward head, with the router's load-balancing
#       loss.
#    4. A centralised twin Q, also two-hot, trained on n-step targets INSIDE
#       the model rollout with a geometric step weighting rho^t.
#    5. A multi-agent MPPI planner over the learned model: sample action
#       sequences, roll them through the dynamics and reward models, bootstrap
#       with Q at the horizon, and refit the sampling distribution to the
#       elites.
#
#  Why it belongs in this paper's table.  Every other baseline here is
#  policy-centric: it learns what to do.  M3W learns what HAPPENS and then
#  searches.  On this instance the disturbance is a known-form function of the
#  neighbours' exertions, so a model that can represent it should be able to
#  plan around it WITHOUT being told the form -- which is the strongest
#  possible version of the "you did not need a declared estimator" objection.
#  The multi-task MoE has a natural reading here too: the driver A(t) cycles,
#  so the dynamics visit distinct regimes within one episode, which is the
#  "bounded similarity in task dynamics" the mixture exists to exploit.
#
#  REQUIRES both of:
#      task.ns_observe_prev_action=true    so a WINDOW of observations is a
#      task.ns_observe_prev_reward=true    window of transitions (o, a, r, o'),
#                                          which is what an h-step model
#                                          rollout and an n-step return are
#                                          defined on.
#
#  THIS ROW IS SLOW.  Planning runs `iterations * horizon` dynamics and reward
#  forwards over `n_envs * num_samples` tokens at EVERY environment step.  Drop
#  `experiment.off_policy_n_envs_per_worker` for this row rather than the
#  planner's published settings; the launcher already does.
#
#  See `baselines/docs/m3w.md`.

from __future__ import annotations

import math
from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, List, Optional, Tuple, Type

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.objectives import LossModule

from benchmarl.algorithms._baseline_math import (
    cv_squared,
    mppi_update,
    nstep_return,
    router_z_loss,
    simnorm,
    two_hot_decode,
    two_hot_loss,
)
from benchmarl.algorithms._history import with_observation_history
from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.models.common import ModelConfig


HISTORY_KEY = "m3w_history"

LOG_STD_MIN = -10.0
LOG_STD_MAX = 2.0


# ===========================================================================
#  the reference's MLP stack
# ===========================================================================


class SimNorm(nn.Module):
    """``world_models.SimNorm``: a softmax over every group of ``dim`` coords."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = int(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return simnorm(x, self.dim)


class NormedLinear(nn.Linear):
    """``world_models.NormedLinear``: Linear + LayerNorm + Mish (+ dropout)."""

    def __init__(self, *args, dropout: float = 0.0, act=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ln = nn.LayerNorm(self.out_features)
        self.act = act if act is not None else nn.Mish()
        self.drop = nn.Dropout(dropout) if dropout else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = super().forward(x)
        if self.drop is not None:
            x = self.drop(x)
        return self.act(self.ln(x))


def create_mlp(in_dim: int, mlp_dims, out_dim: int, act=None, dropout: float = 0.0):
    """``world_models.create_mlp`` with ``normed=True``."""
    if isinstance(mlp_dims, int):
        mlp_dims = [mlp_dims]
    dims = [in_dim] + list(mlp_dims) + [out_dim]
    layers: List[nn.Module] = []
    for i in range(len(dims) - 2):
        layers.append(
            NormedLinear(dims[i], dims[i + 1], dropout=dropout * (i == 0))
        )
    layers.append(
        NormedLinear(dims[-2], dims[-1], act=act)
        if act is not None
        else nn.Linear(dims[-2], dims[-1])
    )
    return nn.Sequential(*layers)


# ===========================================================================
#  the two mixtures
# ===========================================================================


class SoftMoeDynamics(nn.Module):
    """``CenMoEDynamicsModel``: SoftMoE over the agent tokens.

    ``phi`` is ``(d_z + d_a, n_experts, 1)`` -- one slot per expert, as the
    reference builds it.  The dispatch softmax is over TOKENS (so a slot is a
    weighted average of agents) and the combine softmax is over slots (so an
    agent reads a weighted average of expert outputs).
    """

    def __init__(self, d_z: int, d_a: int, mlp_dims, n_experts: int, simnorm_dim: int):
        super().__init__()
        self.d_z = d_z
        self.d_a = d_a
        self.n_experts = int(n_experts)
        self.experts = nn.ModuleList(
            [
                create_mlp(d_z + d_a, mlp_dims, d_z, act=SimNorm(simnorm_dim))
                for _ in range(self.n_experts)
            ]
        )
        self.phi = nn.Parameter(
            torch.randn(d_z + d_a, self.n_experts, 1) / math.sqrt(d_z + d_a)
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z, a], dim=-1)                       # (B, N, d)
        weights = torch.einsum("bnd,des->bnes", x, self.phi)
        dispatch = torch.softmax(weights, dim=1)            # over TOKENS
        inputs = torch.einsum("bnes,bnd->besd", dispatch, x)
        outputs = torch.stack(
            [self.experts[i](inputs[:, i]) for i in range(self.n_experts)], dim=1
        )                                                    # (B, E, S, d_z)
        b, e, s, d_out = outputs.shape
        outputs = outputs.reshape(b, e * s, d_out)
        combine = torch.softmax(weights.reshape(*weights.shape[:2], -1), dim=-1)
        return torch.einsum("bnz,bzd->bnd", combine, outputs)


class NoisyTopKRouter(nn.Module):
    """``world_models.NoisyTopKRouter``, with its load-balancing auxiliary."""

    def __init__(self, in_dim: int, num_experts: int, k: int, noisy_gating: bool):
        super().__init__()
        if k > num_experts:
            raise ValueError(f"top_k={k} > n_experts={num_experts}")
        self.num_experts = int(num_experts)
        self.k = int(k)
        self.noisy_gating = bool(noisy_gating)
        self.w_gate = nn.Parameter(torch.zeros(in_dim, num_experts))
        self.w_noise = nn.Parameter(torch.zeros(in_dim, num_experts))
        self.softplus = nn.Softplus()

    def forward(self, x_flat: torch.Tensor):
        clean_logits = x_flat @ self.w_gate
        if self.noisy_gating and self.training:
            noise_stddev = self.softplus(x_flat @ self.w_noise) + 1e-2
            logits = clean_logits + torch.randn_like(clean_logits) * noise_stddev
        else:
            logits = clean_logits
        top_logits, top_indices = logits.topk(
            min(self.k + 1, self.num_experts), dim=1
        )
        top_k_gates = torch.softmax(top_logits[:, : self.k], dim=1)
        gates = torch.zeros_like(logits, requires_grad=True).scatter(
            1, top_indices[:, : self.k], top_k_gates
        )
        #  The reference estimates `load` with a Gaussian CDF when the gating
        #  is noisy.  The counting form is used here for both branches: the
        #  probabilistic estimate exists to make the auxiliary differentiable
        #  through the noise scale, and the counting form is what the reference
        #  itself falls back to whenever the module is in eval mode or
        #  `k == num_experts`.  Stated in baselines/docs/m3w.md.
        load = (gates > 0).sum(0).to(gates.dtype)
        importance = gates.sum(0)
        balancing = cv_squared(importance) + cv_squared(load) + router_z_loss(logits)
        return gates, balancing


class SelfAttnExpert(nn.Module):
    """``world_models.SelfAttnExpert``: attention across the agent tokens."""

    def __init__(self, d_model: int, n_heads: int, ffn_hidden: int, dropout: float):
        super().__init__()
        self.attn = nn.MultiheadAttention(
            d_model, n_heads, dropout=dropout, batch_first=True
        )
        self.ln1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden),
            nn.ReLU(),
            nn.Linear(ffn_hidden, d_model),
        )
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attended, _ = self.attn(x, x, x, need_weights=False)
        x = self.ln1(x + attended)
        return self.ln2(x + self.ffn(x))


class SparseMoeReward(nn.Module):
    """``CenMoERewardModel``: noisy top-k router, attention experts, two-hot head."""

    def __init__(
        self,
        d_z: int,
        d_a: int,
        n_agents: int,
        n_experts: int,
        k: int,
        n_bins: int,
        n_heads: int,
        expert_ffn_hidden: int,
        head_hidden: int,
        noisy_gating: bool,
    ):
        super().__init__()
        self.d = d_z + d_a
        self.n_agents = n_agents
        self.n_experts = int(n_experts)
        self.router = NoisyTopKRouter(
            in_dim=n_agents * self.d,
            num_experts=n_experts,
            k=k,
            noisy_gating=noisy_gating,
        )
        self.experts = nn.ModuleList(
            [
                SelfAttnExpert(self.d, n_heads, expert_ffn_hidden, 0.0)
                for _ in range(self.n_experts)
            ]
        )
        self.head = nn.Sequential(
            nn.Linear(n_agents * self.d, head_hidden),
            nn.ReLU(),
            nn.Linear(head_hidden, n_bins),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor):
        x = torch.cat([z, a], dim=-1)                       # (B, N, d)
        gates, balancing = self.router(x.reshape(x.shape[0], -1))
        y = torch.zeros_like(x)
        for index, expert in enumerate(self.experts):
            mask = gates[:, index] > 0
            if not bool(mask.any()):
                continue
            weight = gates[mask, index].view(-1, 1, 1)
            y = y.index_put(
                (mask.nonzero(as_tuple=True)[0],),
                y[mask] + weight * expert(x[mask]),
            )
        return self.head(y.reshape(y.shape[0], -1)), balancing


class DisRegQNet(nn.Module):
    """``DisRegQNet``: a centralised Q with a two-hot distributional head."""

    def __init__(self, joint_z: int, joint_a: int, hidden, n_bins: int, dropout: float):
        super().__init__()
        self.mlp = create_mlp(joint_z + joint_a, hidden, n_bins, dropout=dropout)
        #  `self.critic.mlp[-1].weight.data.fill_(0)` -- the reference starts
        #  every bin logit at zero, i.e. a uniform predictive distribution.
        with torch.no_grad():
            self.mlp[-1].weight.zero_()

    def forward(self, joint_z: torch.Tensor, joint_a: torch.Tensor) -> torch.Tensor:
        return self.mlp(torch.cat([joint_z, joint_a], dim=-1))


class RunningScale(nn.Module):
    """``world_models.RunningScale``: a polyak-tracked 5-95 percentile range."""

    def __init__(self, tau: float):
        super().__init__()
        self.tau = float(tau)
        self.register_buffer("value", torch.ones(()))

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        flat = x.detach().reshape(-1)
        if flat.numel() < 2:
            return
        low, high = torch.quantile(
            flat, torch.tensor([0.05, 0.95], device=flat.device, dtype=flat.dtype)
        )
        self.value.lerp_((high - low).clamp_min(1.0), self.tau)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x / self.value


class M3wPolicy(nn.Module):
    """``WorldModelPolicy``: a squashed Gaussian on the LATENT ``z``.

    The action it produces lives in ``[-1, 1]``; the executed action is that
    times the action box's half-width.  The reference multiplies by
    ``action_space.high[0]`` in the same place.
    """

    def __init__(self, d_z: int, act_dim: int, hidden):
        super().__init__()
        self.net = create_mlp(d_z, list(hidden)[:-1], list(hidden)[-1], act=nn.Mish())
        self.mu_layer = nn.Linear(list(hidden)[-1], act_dim)
        self.log_std_layer = nn.Linear(list(hidden)[-1], act_dim)

    def forward(self, z: torch.Tensor, stochastic: bool = True, with_logprob: bool = False):
        out = self.net(z)
        mu = self.mu_layer(out)
        log_std = self.log_std_layer(out)
        log_std = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (
            torch.tanh(log_std) + 1.0
        )
        if stochastic:
            eps = torch.randn_like(mu)
        else:
            eps = torch.zeros_like(mu)
        pi = mu + eps * log_std.exp()
        log_pi = None
        if with_logprob:
            residual = (-0.5 * eps.pow(2) - log_std).sum(-1, keepdim=True)
            log_pi = (
                residual - 0.5 * math.log(2.0 * math.pi) * eps.shape[-1]
            )
        pi = torch.tanh(pi)
        if log_pi is not None:
            log_pi = log_pi - torch.log(
                torch.nn.functional.relu(1.0 - pi.pow(2)) + 1e-6
            ).sum(-1, keepdim=True)
        return pi, log_pi


# ===========================================================================
#  the world model
# ===========================================================================


class M3wWorldModel(nn.Module):
    """Encoders, dynamics, reward, twin critic, targets and actors."""

    def __init__(
        self,
        n_agents: int,
        obs_dim: int,
        act_dim: int,
        latent_dim: int,
        simnorm_dim: int,
        num_enc_layers: int,
        dynamics_hidden,
        num_dynamics_experts: int,
        num_reward_experts: int,
        top_k: int,
        reward_ffn_hidden: int,
        reward_head_hidden: int,
        num_bins: int,
        reward_min: float,
        reward_max: float,
        critic_hidden,
        critic_dropout: float,
        actor_hidden,
        scale_tau: float,
        device,
    ):
        super().__init__()
        self.n_agents = n_agents
        self.latent_dim = latent_dim
        self.act_dim = act_dim
        self.num_bins = num_bins
        self.reward_min = reward_min
        self.reward_max = reward_max

        self.encoders = nn.ModuleList(
            [
                create_mlp(
                    obs_dim,
                    [latent_dim] * max(num_enc_layers - 1, 1),
                    latent_dim,
                    act=SimNorm(simnorm_dim),
                )
                for _ in range(n_agents)
            ]
        )
        self.dynamics = SoftMoeDynamics(
            d_z=latent_dim,
            d_a=act_dim,
            mlp_dims=dynamics_hidden,
            n_experts=num_dynamics_experts,
            simnorm_dim=simnorm_dim,
        )
        self.reward = SparseMoeReward(
            d_z=latent_dim,
            d_a=act_dim,
            n_agents=n_agents,
            n_experts=num_reward_experts,
            k=top_k,
            n_bins=num_bins,
            n_heads=1,
            expert_ffn_hidden=reward_ffn_hidden,
            head_hidden=reward_head_hidden,
            noisy_gating=True,
        )
        joint_z = latent_dim * n_agents
        joint_a = act_dim * n_agents
        self.critic1 = DisRegQNet(joint_z, joint_a, critic_hidden, num_bins, critic_dropout)
        self.critic2 = DisRegQNet(joint_z, joint_a, critic_hidden, num_bins, critic_dropout)
        self.target_critic1 = DisRegQNet(
            joint_z, joint_a, critic_hidden, num_bins, critic_dropout
        )
        self.target_critic2 = DisRegQNet(
            joint_z, joint_a, critic_hidden, num_bins, critic_dropout
        )
        self.target_critic1.load_state_dict(self.critic1.state_dict())
        self.target_critic2.load_state_dict(self.critic2.state_dict())
        for param in list(self.target_critic1.parameters()) + list(
            self.target_critic2.parameters()
        ):
            param.requires_grad = False

        self.actors = nn.ModuleList(
            [M3wPolicy(latent_dim, act_dim, actor_hidden) for _ in range(n_agents)]
        )
        self.scale = RunningScale(scale_tau)
        self.register_buffer(
            "bins", torch.linspace(reward_min, reward_max, num_bins)
        )
        self.to(device)

    # -- the pieces ------------------------------------------------------

    def encode(self, observation: torch.Tensor) -> torch.Tensor:
        """``(..., n_agents, obs_dim)`` -> ``(..., n_agents, latent_dim)``."""
        return torch.stack(
            [self.encoders[i](observation[..., i, :]) for i in range(self.n_agents)],
            dim=-2,
        )

    def next_latent(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.dynamics(z, a)

    def reward_logits(self, z: torch.Tensor, a: torch.Tensor):
        return self.reward(z, a)

    def decode_reward(self, logits: torch.Tensor) -> torch.Tensor:
        return two_hot_decode(logits, self.bins)

    def q_logits(self, z: torch.Tensor, a: torch.Tensor, target: bool = False):
        joint_z = z.reshape(*z.shape[:-2], -1)
        joint_a = a.reshape(*a.shape[:-2], -1)
        if target:
            return (
                self.target_critic1(joint_z, joint_a),
                self.target_critic2(joint_z, joint_a),
            )
        return self.critic1(joint_z, joint_a), self.critic2(joint_z, joint_a)

    def q_value(self, z, a, mode: str = "mean", target: bool = False):
        first, second = self.q_logits(z, a, target=target)
        one = two_hot_decode(first, self.bins)
        two = two_hot_decode(second, self.bins)
        return torch.min(one, two) if mode == "min" else (one + two) / 2.0

    def actor_actions(self, z: torch.Tensor, stochastic: bool = True, with_logprob=False):
        outs = [
            self.actors[i](z[..., i, :], stochastic=stochastic, with_logprob=with_logprob)
            for i in range(self.n_agents)
        ]
        actions = torch.stack([o[0] for o in outs], dim=-2)
        logps = (
            torch.stack([o[1] for o in outs], dim=-2) if with_logprob else None
        )
        return actions, logps

    @torch.no_grad()
    def polyak(self, tau: float) -> None:
        for target, live in (
            (self.target_critic1, self.critic1),
            (self.target_critic2, self.critic2),
        ):
            for t, p in zip(target.parameters(), live.parameters()):
                t.data.mul_(1.0 - tau).add_(p.data, alpha=tau)

    # -- parameter groups -------------------------------------------------

    def model_parameters(self) -> List[torch.nn.Parameter]:
        """The reference's ``model_optimizer``: encoders, dynamics, reward, Q."""
        params: List[torch.nn.Parameter] = []
        for module in (self.encoders, self.dynamics, self.reward, self.critic1, self.critic2):
            params += list(module.parameters())
        return params


# ===========================================================================
#  the planner
# ===========================================================================


class M3wPlanner(nn.Module):
    """``world_model_runner.plan``: multi-agent MPPI over the learned model."""

    def __init__(
        self,
        world_model: M3wWorldModel,
        horizon: int,
        iterations: int,
        num_samples: int,
        num_pi_trajs: int,
        num_elites: int,
        min_std: float,
        max_std: float,
        temperature: float,
        gamma: float,
        use_plan: bool,
        act_limit: torch.Tensor,
    ):
        super().__init__()
        self.wm = world_model
        self.horizon = int(horizon)
        self.iterations = int(iterations)
        self.num_samples = int(num_samples)
        self.num_pi_trajs = int(num_pi_trajs)
        self.num_elites = int(num_elites)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.temperature = float(temperature)
        self.gamma = float(gamma)
        self.use_plan = bool(use_plan)
        self.register_buffer("act_limit", act_limit)
        self._running_mean: Optional[torch.Tensor] = None

    # ------------------------------------------------------------------

    @torch.no_grad()
    def estimate_value(self, z: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """``estimate_value``: roll the model, sum discounted rewards, bootstrap.

        ``z`` is ``(B, n_agents, d_z)`` and ``actions`` is
        ``(horizon, B, n_agents, d_a)``; the return is ``(B, 1)``.  The
        reference builds one return per agent from the SAME centralised reward
        prediction, so they are identical and the planner averages them -- that
        average is computed once here.
        """
        total = torch.zeros(z.shape[0], 1, device=z.device, dtype=z.dtype)
        current = z
        for t in range(self.horizon):
            logits, _ = self.wm.reward_logits(current, actions[t])
            total = total + (self.gamma**t) * self.wm.decode_reward(logits)
            current = self.wm.next_latent(current, actions[t])
        terminal_actions, _ = self.wm.actor_actions(current, stochastic=True)
        total = total + (self.gamma**self.horizon) * self.wm.q_value(
            current, terminal_actions, mode="mean"
        )
        return total.nan_to_num(0.0)

    @torch.no_grad()
    def plan(self, observation: torch.Tensor) -> torch.Tensor:
        """``(B, n_agents, obs_dim)`` -> ``(B, n_agents, act_dim)`` in ``[-1,1]``."""
        device = observation.device
        batch = observation.shape[0]
        n_agents = self.wm.n_agents
        act_dim = self.wm.act_dim
        z = self.wm.encode(observation)

        if not self.use_plan:
            actions, _ = self.wm.actor_actions(z, stochastic=True)
            return actions

        #  Warm start from the previous step's plan, shifted by one.  The
        #  reference resets it at the first step of an episode; here it is
        #  always shifted, because the acting module is not told where an
        #  episode began.  One stale plan per episode; see
        #  baselines/docs/m3w.md.
        mean = torch.zeros(self.horizon, batch, n_agents, act_dim, device=device)
        if (
            self._running_mean is not None
            and self._running_mean.shape == mean.shape
        ):
            mean[:-1] = self._running_mean[1:]
        std = torch.full_like(mean, self.max_std)

        z_rep = z.unsqueeze(1).expand(batch, self.num_samples, n_agents, -1)
        z_flat = z_rep.reshape(batch * self.num_samples, n_agents, -1)

        pi_actions = None
        if self.num_pi_trajs > 0:
            #  A fraction of the candidates comes from the policy, rolled
            #  through the model -- the reference's `pi_actions`.
            pi_actions = torch.zeros(
                self.horizon, batch, self.num_pi_trajs, n_agents, act_dim, device=device
            )
            current = (
                z.unsqueeze(1)
                .expand(batch, self.num_pi_trajs, n_agents, -1)
                .reshape(batch * self.num_pi_trajs, n_agents, -1)
            )
            for t in range(self.horizon):
                step, _ = self.wm.actor_actions(current, stochastic=True)
                pi_actions[t] = step.reshape(batch, self.num_pi_trajs, n_agents, act_dim)
                if t < self.horizon - 1:
                    current = self.wm.next_latent(current, step)

        for iteration in range(self.iterations):
            shape = (self.horizon, batch, self.num_samples, n_agents, act_dim)
            samples = (
                mean.unsqueeze(2) + std.unsqueeze(2) * torch.randn(shape, device=device)
            ).clamp(-1.0, 1.0)
            if pi_actions is not None:
                samples[:, :, : self.num_pi_trajs] = pi_actions

            flat_actions = samples.reshape(
                self.horizon, batch * self.num_samples, n_agents, act_dim
            )
            value = self.estimate_value(z_flat, flat_actions).reshape(
                batch, self.num_samples
            )

            elite_idx = torch.topk(value, self.num_elites, dim=-1)[1]
            elite_values = torch.gather(value, -1, elite_idx)
            gather_idx = (
                elite_idx.reshape(1, batch, self.num_elites, 1, 1)
                .expand(self.horizon, batch, self.num_elites, n_agents, act_dim)
            )
            elite_actions = torch.gather(samples, 2, gather_idx)

            mean, std, score = mppi_update(
                elite_values,
                elite_actions.reshape(
                    self.horizon, batch, self.num_elites, n_agents * act_dim
                ),
                self.temperature,
                self.min_std,
                self.max_std,
            )
            mean = mean.reshape(self.horizon, batch, n_agents, act_dim)
            std = std.reshape(self.horizon, batch, n_agents, act_dim)

            if iteration == self.iterations - 1:
                #  The executed action is DRAWN from the elites in proportion
                #  to their MPPI score, then perturbed by the fitted standard
                #  deviation -- the reference's `np.random.choice(p=score)`
                #  followed by `+= randn * act_std`.
                pick = torch.multinomial(score, 1).squeeze(-1)      # (batch,)
                chosen = elite_actions[0][
                    torch.arange(batch, device=device), pick
                ]                                                   # (B, N, D)
                noise = torch.randn_like(chosen) * std[0]
                self._running_mean = mean
                return (chosen + noise).clamp(-1.0, 1.0)

        self._running_mean = mean
        return mean[0]

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return self.plan(observation) * self.act_limit


# ===========================================================================
#  the loss
# ===========================================================================


class M3wLoss(LossModule):
    """``model_train`` and ``actor_train``, on a window of transitions.

    The model step is BenchMARL's ``loss_model`` and is stepped by BenchMARL's
    optimiser.  The actor step is performed HERE, agent by agent, with its own
    per-agent optimisers, for the same reason HASAC's sweep is: agent ``m+1``
    has to be trained against agent ``m``'s ALREADY-UPDATED policy, and one
    backward pass over a single summed loss cannot express that.
    """

    def __init__(
        self,
        world_model: M3wWorldModel,
        group: str,
        n_agents: int,
        horizon: int,
        n_step: int,
        gamma: float,
        step_rho: float,
        polyak_tau: float,
        q_coef: float,
        reward_coef: float,
        dynamics_coef: float,
        balance_coef: float,
        actor_entropy_coef: float,
        fixed_order: bool,
        actor_lr: float,
        action_slice: slice,
        reward_slice: slice,
        obs_dim: int,
        history_len: int,
    ):
        super().__init__()
        self.wm = world_model
        self.group = group
        self.n_agents = n_agents
        self.horizon = int(horizon)
        self.n_step = int(n_step)
        self.gamma = float(gamma)
        self.step_rho = float(step_rho)
        self.polyak_tau = float(polyak_tau)
        self.q_coef = float(q_coef)
        self.reward_coef = float(reward_coef)
        self.dynamics_coef = float(dynamics_coef)
        self.balance_coef = float(balance_coef)
        self.actor_entropy_coef = float(actor_entropy_coef)
        self.fixed_order = bool(fixed_order)
        self.action_slice = action_slice
        self.reward_slice = reward_slice
        self.obs_dim = int(obs_dim)
        self.history_len = int(history_len)
        self._actor_optimizers: Optional[List[torch.optim.Optimizer]] = None
        self._actor_lr = float(actor_lr)

    # ------------------------------------------------------------------

    def _unpack(self, history: torch.Tensor):
        """A window of observations -> the sequence of transitions in it.

        Frame ``k`` of the window carries ``(o_{s+k}, a_{s+k-1}, r_{s+k-1})``:
        the environment appends the agent's own PREVIOUS action and the reward
        of the transition that produced the observation.  So frames ``k`` and
        ``k+1`` together are one complete transition, and ``W`` frames are
        ``W - 1`` of them.

        ``CatFrames`` refills the window on reset, so a window NEVER spans two
        episodes -- which is why no termination mask appears below.  The frames
        before the first real step of an episode repeat the reset observation,
        whose action and reward columns the environment zeroes; that is the same
        zero padding the reference's ``add_zero_elements`` puts in its buffer.
        """
        lead = history.shape[:-1]
        frames = history.reshape(*lead, self.history_len, self.obs_dim)
        observations = frames[..., :-1, :]                     # o_0 .. o_{W-2}
        next_observations = frames[..., 1:, :]                 # o_1 .. o_{W-1}
        actions = next_observations[..., self.action_slice]    # a_0 .. a_{W-2}
        rewards = next_observations[..., self.reward_slice]    # r_0 .. r_{W-2}
        return observations, actions, rewards, next_observations

    # ------------------------------------------------------------------

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        group = self.group
        history = tensordict.get((group, HISTORY_KEY))
        observations, actions, rewards, next_observations = self._unpack(history)
        #  (batch, n_agents, T, .) -> (T, batch, n_agents, .), which is the
        #  layout `model_train` works in.
        observations = observations.movedim(-2, 0)
        next_observations = next_observations.movedim(-2, 0)
        actions = actions.movedim(-2, 0)
        rewards = rewards.movedim(-2, 0)
        team_reward = rewards.mean(-2)                          # (T, batch, 1)

        device = history.device
        n_transitions = observations.shape[0]
        if n_transitions < self.horizon + self.n_step:
            raise RuntimeError(
                f"M3W needs horizon + n_step = {self.horizon + self.n_step} "
                f"transitions per sample but the window holds {n_transitions}. "
                "history_len must be horizon + n_step + 1."
            )

        # ---- n-step targets, off the TARGET critic -----------------------
        with torch.no_grad():
            next_z = self.wm.encode(next_observations)
            targets = []
            for t in range(self.horizon):
                window = team_reward[t : t + self.n_step].movedim(0, -1).squeeze(-2)
                ret, tail = nstep_return(window, self.gamma)
                boot_z = next_z[t + self.n_step - 1]
                boot_a, _ = self.wm.actor_actions(boot_z, stochastic=True)
                q_next = self.wm.q_value(boot_z, boot_a, mode="min", target=True)
                targets.append(ret + tail * q_next)

        # ---- the h-step rollout ------------------------------------------
        dynamics_loss = torch.zeros((), device=device)
        reward_loss = torch.zeros((), device=device)
        q_loss = torch.zeros((), device=device)
        balance_loss = torch.zeros((), device=device)
        reward_error = torch.zeros((), device=device)

        z = self.wm.encode(observations[0])
        latents = [z]
        for t in range(self.horizon):
            weight = self.step_rho**t
            action_t = actions[t]
            reward_logits, balancing = self.wm.reward_logits(z, action_t)
            q1, q2 = self.wm.q_logits(z, action_t)

            z_pred = self.wm.next_latent(z, action_t)
            dynamics_loss = dynamics_loss + weight * torch.nn.functional.mse_loss(
                z_pred, next_z[t]
            )
            reward_loss = reward_loss + weight * two_hot_loss(
                reward_logits,
                team_reward[t],
                self.wm.bins,
                self.wm.reward_min,
                self.wm.reward_max,
            ).mean()
            q_loss = q_loss + weight * 0.5 * (
                two_hot_loss(
                    q1, targets[t], self.wm.bins, self.wm.reward_min, self.wm.reward_max
                ).mean()
                + two_hot_loss(
                    q2, targets[t], self.wm.bins, self.wm.reward_min, self.wm.reward_max
                ).mean()
            )
            balance_loss = balance_loss + weight * balancing
            with torch.no_grad():
                reward_error = reward_error + (
                    self.wm.decode_reward(reward_logits) - team_reward[t]
                ).abs().mean()
            z = z_pred
            latents.append(z)

        dynamics_loss = dynamics_loss / self.horizon
        reward_loss = reward_loss / self.horizon
        q_loss = q_loss / self.horizon
        balance_loss = balance_loss / self.horizon
        reward_error = reward_error / self.horizon

        loss_model = (
            self.q_coef * q_loss
            + self.reward_coef * reward_loss
            + self.dynamics_coef * dynamics_loss
            + self.balance_coef * balance_loss
        )

        #  `zs.detach()` -- the actor is trained on the IMAGINED latents, and
        #  its gradient never reaches the model.
        actor_loss = self._actor_step([latent.detach() for latent in latents[:-1]])
        self.wm.polyak(self.polyak_tau)

        out = {
            "loss_model": loss_model,
            "m3w_dynamics": dynamics_loss.detach(),
            "m3w_reward": reward_loss.detach(),
            "m3w_q": q_loss.detach(),
            "m3w_balance": balance_loss.detach(),
            "m3w_reward_err": reward_error,
            "m3w_actor": torch.as_tensor(actor_loss, device=device),
            "m3w_pi_scale": self.wm.scale.value.detach().clone(),
        }
        return TensorDict(out, batch_size=[])

    # ------------------------------------------------------------------

    def _actor_step(self, latents: List[torch.Tensor]) -> float:
        """``actor_train``: one agent at a time, in a fresh random order.

        Agent ``m``'s action is refreshed from its UPDATED policy before agent
        ``m+1`` is trained, which is the sequential decomposition HARL's
        heterogeneous-agent family rests on and which M3W inherits.
        """
        if self._actor_optimizers is None:
            self._actor_optimizers = [
                torch.optim.Adam(actor.parameters(), lr=self._actor_lr)
                for actor in self.wm.actors
            ]
        with torch.no_grad():
            #  `actions` is filled by the BEFORE-trained actors and refreshed
            #  agent by agent as the sweep proceeds.
            actions = [
                self.wm.actor_actions(latent, stochastic=True)[0]
                for latent in latents
            ]

        order = (
            list(range(self.n_agents))
            if self.fixed_order
            else torch.randperm(self.n_agents).tolist()
        )
        total = 0.0
        for agent in order:
            optimizer = self._actor_optimizers[agent]
            loss = torch.zeros((), device=latents[0].device)
            fresh: List[torch.Tensor] = []
            for t, latent in enumerate(latents):
                own, own_logp = self.wm.actors[agent](
                    latent[..., agent, :], stochastic=True, with_logprob=True
                )
                joint = actions[t].clone()
                joint[..., agent, :] = own
                value = self.wm.q_value(latent, joint, mode="mean")
                if t == 0:
                    self.wm.scale.update(value)
                value = self.wm.scale(value)
                loss = loss + (self.step_rho**t) * (
                    self.actor_entropy_coef * own_logp - value
                ).mean()
                fresh.append(own.detach())
            loss = loss / max(len(latents), 1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += float(loss.detach())
            with torch.no_grad():
                for t in range(len(latents)):
                    refreshed, _ = self.wm.actors[agent](
                        latents[t][..., agent, :], stochastic=True
                    )
                    actions[t][..., agent, :] = refreshed
        return total / max(self.n_agents, 1)


# ===========================================================================


class M3w(Algorithm):
    """M3W on BenchMARL.

    The world model, the planner, the critic and the actors are all M3W's own
    architectures, built here rather than through BenchMARL's ``model``
    config: an MoE dynamics model and a two-hot distributional critic are not
    expressible as an MLP width, and swapping them for one would make the row
    about something else.  ``model=...`` on the command line is therefore
    IGNORED for this algorithm, and the construction banner says so.
    """

    def __init__(
        self,
        latent_dim: int,
        simnorm_dim: int,
        num_enc_layers: int,
        dynamics_hidden: int,
        num_dynamics_experts: int,
        num_reward_experts: int,
        top_k: int,
        reward_ffn_hidden: int,
        reward_head_hidden: int,
        num_bins: int,
        reward_min: float,
        reward_max: float,
        critic_hidden: int,
        critic_dropout: float,
        actor_hidden: int,
        scale_tau: float,
        horizon: int,
        n_step: int,
        step_rho: float,
        q_coef: float,
        reward_coef: float,
        dynamics_coef: float,
        balance_coef: float,
        actor_entropy_coef: float,
        fixed_order: bool,
        plan_iterations: int,
        num_samples: int,
        num_pi_trajs: int,
        num_elites: int,
        min_std: float,
        max_std: float,
        temperature: float,
        use_plan: bool,
        action_slice_start: int,
        action_slice_size: int,
        reward_slice_start: int,
        **kwargs,
    ):
        self.latent_dim = int(latent_dim)
        self.simnorm_dim = int(simnorm_dim)
        self.num_enc_layers = int(num_enc_layers)
        self.dynamics_hidden = int(dynamics_hidden)
        self.num_dynamics_experts = int(num_dynamics_experts)
        self.num_reward_experts = int(num_reward_experts)
        self.top_k = int(top_k)
        self.reward_ffn_hidden = int(reward_ffn_hidden)
        self.reward_head_hidden = int(reward_head_hidden)
        self.num_bins = int(num_bins)
        self.reward_min = float(reward_min)
        self.reward_max = float(reward_max)
        self.critic_hidden = int(critic_hidden)
        self.critic_dropout = float(critic_dropout)
        self.actor_hidden = int(actor_hidden)
        self.scale_tau = float(scale_tau)
        self.horizon = int(horizon)
        self.n_step = int(n_step)
        self.step_rho = float(step_rho)
        self.q_coef = float(q_coef)
        self.reward_coef = float(reward_coef)
        self.dynamics_coef = float(dynamics_coef)
        self.balance_coef = float(balance_coef)
        self.actor_entropy_coef = float(actor_entropy_coef)
        self.fixed_order = bool(fixed_order)
        self.plan_iterations = int(plan_iterations)
        self.num_samples = int(num_samples)
        self.num_pi_trajs = int(num_pi_trajs)
        self.num_elites = int(num_elites)
        self.min_std = float(min_std)
        self.max_std = float(max_std)
        self.temperature = float(temperature)
        self.use_plan = bool(use_plan)
        self.action_slice_start = int(action_slice_start)
        self.action_slice_size = int(action_slice_size)
        self.reward_slice_start = int(reward_slice_start)
        super().__init__(**kwargs)

        if self.has_rnn:
            raise NotImplementedError(
                "M3W here does not support recurrent models: the planner "
                "rolls a latent state forward with a feed-forward dynamics "
                "model, and a recurrent encoder would need its hidden state "
                "carried through the imagination."
            )
        if self.num_elites > self.num_samples:
            raise ValueError(
                f"num_elites={self.num_elites} > num_samples={self.num_samples}"
            )
        if self.num_pi_trajs > self.num_samples:
            raise ValueError(
                f"num_pi_trajs={self.num_pi_trajs} > num_samples={self.num_samples}"
            )
        self._models: Dict[str, M3wWorldModel] = {}
        self._planners: Dict[str, M3wPlanner] = {}

    # ------------------------------------------------------------------

    @property
    def history_len(self) -> int:
        """``horizon + n_step + 1`` frames give ``horizon + n_step`` transitions."""
        return self.horizon + self.n_step + 1

    def _obs_dim(self, group: str) -> int:
        return int(self.observation_spec[group, "observation"].shape[-1])

    def _act_dim(self, group: str) -> int:
        return int(self.action_spec[group, "action"].shape[-1])

    def _act_limit(self, group: str) -> torch.Tensor:
        spec = self.action_spec[group, "action"]
        return spec.space.high.to(self.device).clone()

    def _slices(self, group: str) -> Tuple[slice, slice]:
        obs_dim = self._obs_dim(group)
        act_dim = self._act_dim(group)
        size = self.action_slice_size or act_dim
        start = self.action_slice_start
        if start == 0:
            #  The environment appends, in order: residual, driver,
            #  prev_action (D), prev_reward (1).  So with prev_reward on, the
            #  action block ends one before the end.
            start = obs_dim - 1 - size
        elif start < 0:
            start += obs_dim
        action_slice = slice(start, start + size)
        reward_start = self.reward_slice_start
        if reward_start < 0:
            reward_start += obs_dim
        reward_slice = slice(reward_start, reward_start + 1)
        if not (0 <= action_slice.start < action_slice.stop <= obs_dim):
            raise ValueError(
                f"M3W previous-action slice [{action_slice.start}, "
                f"{action_slice.stop}) does not fit an observation of width "
                f"{obs_dim}. The model's actions are read off the observation "
                "window: launch with task.ns_observe_prev_action=true."
            )
        if not (0 <= reward_slice.start < reward_slice.stop <= obs_dim):
            raise ValueError(
                f"M3W previous-reward slice [{reward_slice.start}, "
                f"{reward_slice.stop}) does not fit an observation of width "
                f"{obs_dim}. The n-step return is read off the observation "
                "window: launch with task.ns_observe_prev_reward=true."
            )
        if action_slice.stop > reward_slice.start >= action_slice.start:
            raise ValueError(
                "M3W's previous-action and previous-reward slices overlap: "
                f"[{action_slice.start}, {action_slice.stop}) and "
                f"[{reward_slice.start}, {reward_slice.stop})."
            )
        return action_slice, reward_slice

    def process_env_fun(self, env_fun):
        """A window of ``horizon + n_step + 1`` observations per transition.

        M3W trains on SEQUENCES -- an ``h``-step latent rollout and an
        ``n``-step return -- and BenchMARL's off-policy buffer samples flat
        transitions.  Frame stacking puts the sequence inside the transition,
        so the two fit together without a sequence sampler; with
        ``ns_observe_prev_action`` and ``ns_observe_prev_reward`` on, each
        frame carries its own action and reward and the window is a trajectory
        chunk.  See ``M3wLoss._unpack``.
        """
        return with_observation_history(
            env_fun,
            groups=list(self.group_map.keys()),
            history_len=self.history_len,
            history_key=HISTORY_KEY,
        )

    def _build(self, group: str) -> M3wWorldModel:
        if group in self._models:
            return self._models[group]
        model = M3wWorldModel(
            n_agents=len(self.group_map[group]),
            obs_dim=self._obs_dim(group),
            act_dim=self._act_dim(group),
            latent_dim=self.latent_dim,
            simnorm_dim=self.simnorm_dim,
            num_enc_layers=self.num_enc_layers,
            dynamics_hidden=[self.dynamics_hidden, self.dynamics_hidden],
            num_dynamics_experts=self.num_dynamics_experts,
            num_reward_experts=self.num_reward_experts,
            top_k=self.top_k,
            reward_ffn_hidden=self.reward_ffn_hidden,
            reward_head_hidden=self.reward_head_hidden,
            num_bins=self.num_bins,
            reward_min=self.reward_min,
            reward_max=self.reward_max,
            critic_hidden=[self.critic_hidden, self.critic_hidden],
            critic_dropout=self.critic_dropout,
            actor_hidden=[self.actor_hidden, self.actor_hidden],
            scale_tau=self.scale_tau,
            device=self.device,
        )
        self._models[group] = model
        return model

    # ------------------------------------------------------------------
    #  the BenchMARL interface
    # ------------------------------------------------------------------

    def _get_policy_for_loss(
        self, group: str, model_config: ModelConfig, continuous: bool
    ) -> TensorDictModule:
        if not continuous:
            raise NotImplementedError(
                "M3W covers continuous actions only: the planner samples and "
                "refits a Gaussian over action SEQUENCES, which a discrete "
                "action set does not admit without a different search."
            )
        model = self._build(group)
        planner = M3wPlanner(
            world_model=model,
            horizon=self.horizon,
            iterations=self.plan_iterations,
            num_samples=self.num_samples,
            num_pi_trajs=self.num_pi_trajs,
            num_elites=self.num_elites,
            min_std=self.min_std,
            max_std=self.max_std,
            temperature=self.temperature,
            gamma=self.experiment_config.gamma,
            use_plan=self.use_plan,
            act_limit=self._act_limit(group),
        ).to(self.device)
        self._planners[group] = planner
        return TensorDictModule(
            planner,
            in_keys=[(group, "observation")],
            out_keys=[(group, "action")],
        )

    def _get_policy_for_collection(
        self, policy_for_loss: TensorDictModule, group: str, continuous: bool
    ) -> TensorDictModule:
        #  The planner IS the exploration: it draws the executed action from
        #  the elites in proportion to their MPPI score and perturbs it by the
        #  fitted standard deviation, and `experiment.off_policy_init_random_frames`
        #  supplies M3W's `warmup_steps` of uniformly random actions before any
        #  of it runs.
        return policy_for_loss

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        model = self._build(group)
        action_slice, reward_slice = self._slices(group)
        print(
            f"M3W {group}: latent {self.latent_dim} (SimNorm groups of "
            f"{self.simnorm_dim}); SoftMoE dynamics with "
            f"{self.num_dynamics_experts} experts; SparseMoE reward with "
            f"{self.num_reward_experts} experts, top-{self.top_k}; two-hot "
            f"heads over {self.num_bins} bins on "
            f"[{self.reward_min}, {self.reward_max}]; rollout horizon "
            f"{self.horizon} with rho={self.step_rho}; {self.n_step}-step "
            f"targets; window = {self.history_len} frames, actions read from "
            f"observation[..., {action_slice.start}:{action_slice.stop}] and "
            f"rewards from observation[..., {reward_slice.start}:"
            f"{reward_slice.stop}]; planner "
            + (
                f"MPPI {self.plan_iterations} iterations x {self.num_samples} "
                f"samples ({self.num_pi_trajs} from the policy), "
                f"{self.num_elites} elites, temperature {self.temperature}"
                if self.use_plan
                else "DISABLED (acting with the policy directly)"
            )
            + ". BenchMARL's `model=` config is IGNORED here: the world "
            "model's architecture is the paper's."
        )
        if self.experiment_config.clip_grad_val != 20:
            print(
                "[m3w] the reference clips the world-model gradient at norm "
                f"20; this run has experiment.clip_grad_val="
                f"{self.experiment_config.clip_grad_val}. Set "
                "experiment.clip_grad_norm=true experiment.clip_grad_val=20 "
                "to match it."
            )
        loss = M3wLoss(
            world_model=model,
            group=group,
            n_agents=len(self.group_map[group]),
            horizon=self.horizon,
            n_step=self.n_step,
            gamma=self.experiment_config.gamma,
            step_rho=self.step_rho,
            polyak_tau=self.experiment_config.polyak_tau,
            q_coef=self.q_coef,
            reward_coef=self.reward_coef,
            dynamics_coef=self.dynamics_coef,
            balance_coef=self.balance_coef,
            actor_entropy_coef=self.actor_entropy_coef,
            fixed_order=self.fixed_order,
            actor_lr=self.experiment_config.lr,
            action_slice=action_slice,
            reward_slice=reward_slice,
            obs_dim=self._obs_dim(group),
            history_len=self.history_len,
        ).to(self.device)
        #  use_target=False: the twin critic's targets are polyak-updated
        #  inside the loss, on the reference's own schedule (once per training
        #  step, `polyak`), not by BenchMARL's TargetNetUpdater.
        return loss, False

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        model = self._models[group]
        #  The ACTORS are deliberately absent: they are stepped agent by agent
        #  inside the loss, so that agent m+1 trains against agent m's updated
        #  policy. See M3wLoss._actor_step.
        return {"loss_model": model.model_parameters()}

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        keys = list(batch.keys(True, True))
        group_shape = batch.get(group).shape
        for source, nested in (
            (("next", "done"), ("next", group, "done")),
            (("next", "terminated"), ("next", group, "terminated")),
            (("next", "reward"), ("next", group, "reward")),
        ):
            if nested not in keys:
                batch.set(
                    nested,
                    batch.get(source).unsqueeze(-1).expand((*group_shape, 1)),
                )
        return batch


@dataclass
class M3wConfig(AlgorithmConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.M3w`."""

    latent_dim: int = MISSING
    simnorm_dim: int = MISSING
    num_enc_layers: int = MISSING
    dynamics_hidden: int = MISSING
    num_dynamics_experts: int = MISSING
    num_reward_experts: int = MISSING
    top_k: int = MISSING
    reward_ffn_hidden: int = MISSING
    reward_head_hidden: int = MISSING
    num_bins: int = MISSING
    reward_min: float = MISSING
    reward_max: float = MISSING
    critic_hidden: int = MISSING
    critic_dropout: float = MISSING
    actor_hidden: int = MISSING
    scale_tau: float = MISSING
    horizon: int = MISSING
    n_step: int = MISSING
    step_rho: float = MISSING
    q_coef: float = MISSING
    reward_coef: float = MISSING
    dynamics_coef: float = MISSING
    balance_coef: float = MISSING
    actor_entropy_coef: float = MISSING
    fixed_order: bool = MISSING
    plan_iterations: int = MISSING
    num_samples: int = MISSING
    num_pi_trajs: int = MISSING
    num_elites: int = MISSING
    min_std: float = MISSING
    max_std: float = MISSING
    temperature: float = MISSING
    use_plan: bool = MISSING
    action_slice_start: int = MISSING
    action_slice_size: int = MISSING
    reward_slice_start: int = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return M3w

    @staticmethod
    def on_policy() -> bool:
        return False

    @staticmethod
    def supports_continuous_actions() -> bool:
        return True

    @staticmethod
    def supports_discrete_actions() -> bool:
        return False

    @staticmethod
    def has_centralized_critic() -> bool:
        return True
