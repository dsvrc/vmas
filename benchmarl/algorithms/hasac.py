#  HASAC -- Heterogeneous-Agent Soft Actor-Critic.
#
#      Liu et al., "Maximum Entropy Heterogeneous-Agent Reinforcement
#      Learning", ICLR 2024 (arXiv 2306.10715); shipped in HARL, JMLR 2024.
#      Reference code: PKU-MARL/HARL --
#        `harl/algorithms/actors/hasac.py`
#        `harl/algorithms/critics/soft_twin_continuous_q_critic.py`
#        `harl/runners/off_policy_ha_runner.py`
#
#  BASELINES.md B1, off-policy half: the same sequential-update answer to
#  non-stationarity as HAPPO, in a maximum-entropy off-policy algorithm, so it
#  drops into the MATD3/MASAC slot of the ladder rather than the PPO one.
#
#  Three things separate HASAC from BenchMARL's MASAC, and they are the only
#  three things this file adds:
#
#  1. SEQUENTIAL ACTOR UPDATE.  The agents are updated one at a time in a random
#     order; when agent m's turn comes, the joint action fed to the Q function
#     uses the ALREADY-UPDATED policies of the agents before it.  (With every
#     agent's action differentiable at once -- MASAC -- the gradient for agent m
#     is the same as HASAC's except that the earlier agents' actions are the
#     pre-update ones.  The whole difference is the refresh.)
#
#  2. JOINT SOFT TARGET.  HARL's critic target subtracts the entropy of the
#     WHOLE joint action:
#         y = r + gamma (1-d) ( min_j Q'_j(s', a') - alpha_c * SUM_i log pi_i )
#     MASAC subtracts only agent i's own log-prob from agent i's target, which
#     -- with a shared centralised Q -- makes its effective entropy weight
#     alpha/N.  This is the maximum-entropy part of "maximum entropy
#     heterogeneous-agent", so it is implemented, not approximated.
#
#  3. PER-AGENT TEMPERATURE.  HARL tunes one alpha per agent for the actors and
#     one further alpha for the critic target, the latter against the SUMMED
#     target entropy.
#
#  See `baselines/docs/hasac.md` for the clause-by-clause checklist and for the
#  two places this is not bit-identical to HARL.

from __future__ import annotations

import warnings
from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Optional, Tuple, Type

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import LossModule, SACLoss, ValueEstimators

from benchmarl.algorithms import _compat
from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.algorithms.masac import Masac, MasacConfig


def _log_prob(dist, action) -> torch.Tensor:
    """Log-probability of ``action`` under ``dist``, aggregated over action dims.

    For the distributions BenchMARL builds for continuous control (TanhNormal,
    IndependentNormal) this is already the joint log-probability of an agent's
    action vector, shaped ``(*batch, n_agents)``.
    """
    lp = dist.log_prob(action)
    if hasattr(lp, "keys"):  # a composite log-prob; sum its leaves
        lp = sum(lp.values(True, True))
    return lp


class HasacLoss(SACLoss):
    """SAC with HASAC's joint soft target and one-agent-at-a-time actor update.

    The current agent advances by one every ``calls_per_agent`` optimiser calls
    and the permutation is redrawn when the sweep wraps, which is the off-policy
    analogue of HARL's ``agent_order`` loop: the sweep is spread over
    consecutive optimiser calls instead of running inside one, so agent m+1 is
    still trained against agent m's UPDATED policy -- the property the method
    rests on -- while each call costs one backward pass rather than N.
    """

    def __init__(
        self,
        *args,
        n_agents: int,
        calls_per_agent: int,
        fixed_order: bool,
        per_agent_alpha: bool,
        joint_entropy_target: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_agents = int(n_agents)
        self.calls_per_agent = max(int(calls_per_agent), 1)
        self.fixed_order = bool(fixed_order)
        self.joint_entropy_target = bool(joint_entropy_target)

        if self.skip_done_states:
            raise NotImplementedError(
                "HasacLoss reimplements the SAC target to subtract the joint "
                "entropy and does not implement torchrl's skip_done_states path."
            )

        self._call = 0
        self._order = list(range(self.n_agents))
        self._position = 0

        # --- per-agent actor temperature (HARL: self.log_alpha[agent_id]) ----
        self.per_agent_alpha = bool(per_agent_alpha) and not self.fixed_alpha
        if self.per_agent_alpha:
            try:
                init = self.log_alpha.detach().clone().reshape(())
                del self._parameters["log_alpha"]
                self.register_parameter(
                    "log_alpha",
                    torch.nn.Parameter(init.expand(self.n_agents).clone()),
                )
            except Exception as err:  # pragma: no cover - defensive
                self.per_agent_alpha = False
                warnings.warn(
                    "HASAC: could not install a per-agent temperature "
                    f"({type(err).__name__}: {err}); falling back to torchrl's "
                    "single shared alpha. This is MASAC's convention, not "
                    "HARL's."
                )

        # --- the critic's own temperature (HARL: critic.log_alpha) ----------
        #  A SEPARATE scalar, tuned against the SUM of the per-agent target
        #  entropies, used only inside the Q target.
        self.register_parameter(
            "log_alpha_critic",
            torch.nn.Parameter(self.log_alpha.detach().reshape(-1)[0].clone()),
        )

    # ------------------------------------------------------------------
    #  the sweep
    # ------------------------------------------------------------------

    def _advance(self) -> None:
        if self._call % self.calls_per_agent == 0:
            if self._call % (self.calls_per_agent * self.n_agents) == 0:
                # HARL: `agent_order = list(np.random.permutation(num_agents))`
                self._order = (
                    list(range(self.n_agents))
                    if self.fixed_order
                    else torch.randperm(self.n_agents).tolist()
                )
                self._position = 0
            else:
                self._position = (self._position + 1) % self.n_agents
        self._call += 1

    def current_agent(self) -> int:
        return self._order[self._position]

    def _agent_mask(self, like: torch.Tensor, agent: int) -> torch.Tensor:
        mask = torch.zeros(self.n_agents, device=like.device, dtype=like.dtype)
        mask[agent] = 1.0
        return mask

    # ------------------------------------------------------------------
    #  the critic target: HARL's joint soft value
    # ------------------------------------------------------------------

    def _compute_target_v2(self, tensordict) -> torch.Tensor:
        tensordict = tensordict.clone(False)
        with torch.no_grad():
            with set_exploration_type(
                ExplorationType.RANDOM
            ), self.actor_network_params.to_module(self.actor_network):
                next_tensordict = tensordict.get("next").copy()
                next_dist = self.actor_network.get_dist(next_tensordict)
                next_action = next_dist.rsample()
                next_tensordict.set(self.tensor_keys.action, next_action)
                next_log_prob = _log_prob(next_dist, next_action)

            next_tensordict_expand = self._vmap_qnetworkN0(
                next_tensordict, self.target_qvalue_network_params
            )
            state_action_value = next_tensordict_expand.get(
                self.tensor_keys.state_action_value
            )
            if self.joint_entropy_target:
                #  HARL: next_logp_actions = sum over agents, one scalar per
                #  transition, subtracted from the single joint Q.
                entropy_term = next_log_prob.sum(-1, keepdim=True)
            else:
                entropy_term = next_log_prob
            if (
                state_action_value.shape[-len(entropy_term.shape) :]
                != entropy_term.shape
            ):
                entropy_term = entropy_term.unsqueeze(-1)
            alpha_c = self._alpha_critic
            next_state_value = state_action_value - alpha_c * entropy_term
            next_state_value = next_state_value.min(0)[0]
            tensordict.set(
                ("next", self.value_estimator.tensor_keys.value), next_state_value
            )
            return self.value_estimator.value_estimate(tensordict).squeeze(-1)

    @property
    def _alpha_critic(self):
        if self.min_log_alpha is not None or self.max_log_alpha is not None:
            self.log_alpha_critic.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            return self.log_alpha_critic.exp()

    @property
    def _alpha_vector(self):
        if self.min_log_alpha is not None or self.max_log_alpha is not None:
            self.log_alpha.data.clamp_(self.min_log_alpha, self.max_log_alpha)
        with torch.no_grad():
            return self.log_alpha.exp()

    # ------------------------------------------------------------------
    #  the actor: one agent, against the others' current actions
    # ------------------------------------------------------------------

    def hasac_actor_loss(self, tensordict, agent: int):
        with set_exploration_type(
            ExplorationType.RANDOM
        ), self.actor_network_params.to_module(self.actor_network):
            dist = self.actor_network.get_dist(tensordict)
            a_reparm = dist.rsample()
        log_prob = _log_prob(dist, a_reparm)

        #  HARL holds actions[j != m] fixed at the value the CURRENT policy of
        #  agent j produces and lets only a_m carry gradient.  Detaching is
        #  exactly that: the values are draws from the live policies (agents
        #  before m in the order have already been updated), and no gradient
        #  reaches them.
        #  (n_agents, 1) broadcasts against (*batch, n_agents, action_dim):
        #  trailing dimensions align, so every batch element and every action
        #  component of agent m is selected and no other agent's is.
        mask = self._agent_mask(a_reparm, agent).unsqueeze(-1)
        a_mixed = torch.where(mask.bool(), a_reparm, a_reparm.detach())

        td_q = tensordict.select(*self.qvalue_network.in_keys, strict=False)
        td_q.set(self.tensor_keys.action, a_mixed)
        td_q = self._vmap_qnetworkN0(td_q, self._cached_detached_qvalue_params)
        min_q = td_q.get(self.tensor_keys.state_action_value).min(0)[0].squeeze(-1)

        alpha = self._alpha_vector if self.per_agent_alpha else self._alpha
        per_agent = alpha * log_prob - min_q
        agent_mask = self._agent_mask(per_agent, agent)
        loss_actor = (per_agent * agent_mask).mean()
        return loss_actor, log_prob.detach()

    # ------------------------------------------------------------------

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        self._advance()
        agent = self.current_agent()

        #  Through _compat: torchrl 0.11 calls this `qvalue_v2_loss`, 0.7.x
        #  -- the cluster's version -- only has `_qvalue_v2_loss`.
        loss_qvalue, value_metadata = _compat.qvalue_v2_loss(self, tensordict)
        loss_actor, log_prob = self.hasac_actor_loss(tensordict, agent)

        #  HARL: alpha_loss = -(log_alpha[m] * (logp[m].detach() + H*[m])).mean()
        #  for the agent whose turn it is, and
        #  -(log_alpha_critic * (sum_i logp_i.detach() + sum_i H*_i)).mean()
        #  for the critic's own temperature.
        agent_mask = self._agent_mask(log_prob, agent)
        if self.fixed_alpha:
            loss_alpha = torch.zeros((), device=log_prob.device)
            loss_alpha_critic = torch.zeros((), device=log_prob.device)
        else:
            loss_alpha = (
                -self.log_alpha * (log_prob + self.target_entropy) * agent_mask
            ).mean()
            joint_log_prob = log_prob.sum(-1)
            loss_alpha_critic = (
                -self.log_alpha_critic
                * (joint_log_prob + self.n_agents * self.target_entropy)
            ).mean()

        tensordict.set(self.tensor_keys.priority, value_metadata["td_error"])
        out = {
            "loss_actor": loss_actor,
            "loss_qvalue": loss_qvalue.mean(),
            "loss_alpha": loss_alpha,
            "loss_alpha_critic": loss_alpha_critic,
            "alpha": (self._alpha_vector if self.per_agent_alpha else self._alpha)
            .reshape(-1)
            .mean(),
            "alpha_critic": self._alpha_critic,
            "entropy": -log_prob.mean(),
            "hasac_agent": torch.tensor(
                float(agent), device=log_prob.device, dtype=log_prob.dtype
            ),
        }
        return TensorDict(out, batch_size=[])


class Hasac(Masac):
    """HASAC on BenchMARL's MASAC host.

    Args:
        fixed_order (bool): HARL's ``fixed_order``. ``False`` redraws the agent
            permutation at the start of every sweep.
        calls_per_agent (int): how many consecutive optimiser calls one agent
            holds before the turn passes. ``1`` is the intended value and makes
            one full sweep every ``n_agents`` optimiser calls.
        per_agent_alpha (bool): one temperature per agent, as HARL has.
            ``False`` keeps torchrl's single shared alpha (MASAC's convention).
        joint_entropy_target (bool): subtract the SUMMED log-probability of the
            joint action in the Q target, as HARL does. ``False`` reverts to
            MASAC's per-agent target and is only there to isolate what this
            term is worth.

    All other arguments are :class:`~benchmarl.algorithms.Masac`'s.
    """

    def __init__(
        self,
        fixed_order: bool,
        calls_per_agent: int,
        per_agent_alpha: bool,
        joint_entropy_target: bool,
        **kwargs,
    ):
        self.fixed_order = bool(fixed_order)
        self.calls_per_agent = int(calls_per_agent)
        self.per_agent_alpha = bool(per_agent_alpha)
        self.joint_entropy_target = bool(joint_entropy_target)
        super().__init__(**kwargs)

        if not self.share_param_critic:
            raise ValueError(
                "HASAC needs share_param_critic=True: its critic is ONE joint "
                "soft Q over the centralised state and every agent's action, "
                "which is what the sequential actor update maximises. A "
                "per-agent critic is a different algorithm."
            )
        if self.experiment_config.share_policy_params:
            warnings.warn(
                "HASAC is running with experiment.share_policy_params=True. The "
                "agents then have ONE policy, so 'update agent m and refresh "
                "its action for agent m+1' moves every agent at once and the "
                "sequential decomposition -- the content of the method -- does "
                "not exist. Launch with share_policy_params=False."
            )

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        if not continuous:
            raise NotImplementedError(
                "HASAC here covers continuous actions only. HARL's discrete "
                "HASAC replaces the squashed Gaussian with a Gumbel-softmax "
                "policy, which is a different actor, not a flag."
            )
        n_agents = len(self.group_map[group])
        print(
            f"HASAC {group}: {n_agents} agents, one actor update per optimiser "
            f"call, a full sweep every {n_agents * self.calls_per_agent} calls "
            f"(effective policy_freq={n_agents * self.calls_per_agent} against "
            f"the critic; HARL's HASAC default is 1). "
            f"fixed_order={self.fixed_order} "
            f"per_agent_alpha={self.per_agent_alpha} "
            f"joint_entropy_target={self.joint_entropy_target}"
        )

        loss_module = HasacLoss(
            actor_network=policy_for_loss,
            qvalue_network=self.get_continuous_value_module(group),
            num_qvalue_nets=self.num_qvalue_nets,
            loss_function=self.loss_function,
            alpha_init=self.alpha_init,
            min_alpha=self.min_alpha,
            max_alpha=self.max_alpha,
            action_spec=self.action_spec,
            fixed_alpha=self.fixed_alpha,
            target_entropy=self.target_entropy,
            delay_qvalue=self.delay_qvalue,
            n_agents=n_agents,
            calls_per_agent=self.calls_per_agent,
            fixed_order=self.fixed_order,
            per_agent_alpha=self.per_agent_alpha,
            joint_entropy_target=self.joint_entropy_target,
        )
        loss_module.set_keys(
            state_action_value=(group, "state_action_value"),
            action=(group, "action"),
            reward=(group, "reward"),
            priority=(group, "td_error"),
            done=(group, "done"),
            terminated=(group, "terminated"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.TD0, gamma=self.experiment_config.gamma
        )
        return loss_module, True

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        items = {
            "loss_actor": list(loss.actor_network_params.flatten_keys().values()),
            "loss_qvalue": list(loss.qvalue_network_params.flatten_keys().values()),
        }
        if not self.fixed_alpha:
            items.update(
                {
                    "loss_alpha": [loss.log_alpha],
                    "loss_alpha_critic": [loss.log_alpha_critic],
                }
            )
        return items


@dataclass
class HasacConfig(MasacConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Hasac`."""

    fixed_order: bool = MISSING
    calls_per_agent: int = MISSING
    per_agent_alpha: bool = MISSING
    joint_entropy_target: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Hasac

    @staticmethod
    def supports_discrete_actions() -> bool:
        return False
