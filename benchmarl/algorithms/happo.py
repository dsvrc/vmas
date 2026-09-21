#  HAPPO -- Heterogeneous-Agent Proximal Policy Optimisation.
#
#      Kuba et al., "Trust Region Policy Optimisation in Multi-Agent
#      Reinforcement Learning", ICLR 2022 (arXiv 2109.11251); the library
#      version is Zhong et al., "Heterogeneous-Agent Reinforcement Learning",
#      JMLR 2024 (arXiv 2304.09870).
#      Reference code: PKU-MARL/HARL, `harl/algorithms/actors/happo.py` and
#      `harl/runners/on_policy_ha_runner.py`.
#
#  BASELINES.md B1: the MARL literature's own answer to non-stationarity is to
#  constrain policy change.  HAPPO does it by updating the agents ONE AT A TIME
#  in a random order, each one's clipped objective scaled by the product of the
#  ratios of the agents that already moved this iteration.  That product is the
#  "factor" M, and it is what turns N independent local improvements into one
#  monotone joint improvement.
#
#      permute the agents;  M <- 1
#      for each agent i in the permutation:
#          maximise  E[ min( M r_i A , M clip(r_i) A ) ]
#          M <- M * ( pi_new_i(a_i|o_i) / pi_old_i(a_i|o_i) )
#
#  Everything HAPPO adds to MAPPO is in that block.  The critic, the GAE, the
#  policy network, the clipping and every shared hyper-parameter are MAPPO's,
#  which is what makes the HAPPO-vs-MAPPO row measure the sequential update and
#  nothing else.
#
#  See `baselines/docs/happo.md` for the clause-by-clause checklist against the
#  paper and against HARL's code, including the places this implementation is
#  not bit-identical to HARL and why.

from __future__ import annotations

import warnings
from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, Optional, Tuple, Type

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms._baseline_math import block_bounds as _block_bounds
from benchmarl.algorithms._baseline_math import happo_log_factor
from benchmarl.algorithms import _compat  # submodule import: safe while
                                          # benchmarl.algorithms is still
                                          # being initialised
from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.algorithms.mappo import Mappo, MappoConfig

#  benchmarl.experiment.callback.Callback is NOT imported here: benchmarl's own
#  __init__ imports benchmarl.algorithms before benchmarl.experiment, and
#  benchmarl.experiment imports names back out of benchmarl.algorithms, so an
#  import-time dependency on it here makes `import benchmarl` fail outright.
#  The base class is fetched at construction instead -- see
#  `_sequential_freeze_callback` below and `_compat.callback_base`.


#  HARL gives every agent ``ppo_epoch`` epochs over ``actor_num_mini_batch``
#  minibatches.  BenchMARL instead fixes the number of optimiser calls per
#  collection iteration at ``n_optimizer_steps * n_minibatches``, so HAPPO
#  spends that budget agent by agent.  With the launcher's recommended setting
#  (``on_policy_n_minibatch_iters = n_agents * E``) each block is exactly E
#  epochs, i.e. HARL's ``ppo_epoch``; with any other setting the blocks are as
#  even as the budget allows and the sizes are printed at construction.
#  `_block_bounds` is imported above so that `baselines/verify.py` can check the
#  split without importing torchrl.


_FREEZE_CALLBACK_CLS = None


def _sequential_freeze_callback(losses: Dict[str, "HappoLoss"]):
    """A callback that writes the authoritative parameters back once per
    iteration per group.

    The sequential update must leave the agents that are not currently being
    updated exactly where they were.  Their gradient is already exactly zero --
    with ``share_policy_params=False`` each agent's policy is its own slice of
    the stacked parameter tensor and the masked objective does not touch the
    others -- but Adam keeps stepping a parameter whose gradient is zero, from
    the momentum left over by its own block, by roughly ``lr/(1-beta1)`` per
    iteration.  Over a run that is not noise.

    torchrl stores the non-shared agent parameters as ONE tensor with a leading
    agent dimension (``MultiAgentMLP``), so there is no way to give each agent
    its own optimiser and no way to hand Adam a ``None`` gradient for the
    others.  The loss module therefore keeps an authoritative copy and restores
    it at the top of every forward; this callback does the final restore after
    the last optimiser step of the iteration, so the parameters used for the
    NEXT collection are the sequential ones too.

    Built here rather than at module level because its base class lives in
    ``benchmarl.experiment``, which imports back out of ``benchmarl.algorithms``
    -- see ``_compat.callback_base``.
    """
    global _FREEZE_CALLBACK_CLS
    if _FREEZE_CALLBACK_CLS is None:

        class _SequentialFreezeCallback(_compat.callback_base()):
            def __init__(self, losses):
                super().__init__()
                self._losses = losses

            def on_train_end(self, training_td: TensorDictBase, group: str):
                loss = self._losses.get(group)
                if loss is not None:
                    loss.commit_current_agent()
                    loss.restore_frozen_agents()

        _FREEZE_CALLBACK_CLS = _SequentialFreezeCallback
    return _FREEZE_CALLBACK_CLS(losses)


class HappoLoss(ClipPPOLoss):
    """MAPPO's clipped loss, restricted to one agent and scaled by the factor.

    The whole of HAPPO is two multiplications of the advantage:

    * by ``M``, the product of the ratios of the agents already updated this
      iteration.  ``min(r M A, clip(r) M A) == M min(r A, clip(r) A)`` because
      ``M > 0``, so scaling the advantage IS HARL's
      ``factor_batch * torch.min(surr1, surr2)``;
    * by a 0/1 agent mask, which is how "update one agent" is expressed when
      every agent's loss is computed in the same forward pass.

    ``M`` is recomputed from the CURRENT parameters at every minibatch rather
    than frozen when an agent finishes its block.  It is the same number: an
    agent that has finished is not updated again (the freeze guarantees it), so
    ``pi_current == pi_after_its_own_update``, which is exactly what HARL
    stores.
    """

    #  Redeclared so torchrl's convert_to_functional does not warn: it
    #  checks the SUBCLASS's own __annotations__, which is empty unless
    #  the names are repeated here.  Same list torchrl's own losses carry.
    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams
    critic_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_critic_network_params: TensorDictParams

    def __init__(
        self,
        *args,
        n_agents: int,
        total_calls: int,
        fixed_order: bool,
        factor_clip: float,
        freeze_non_updating_agents: bool,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.n_agents = int(n_agents)
        self.total_calls = int(total_calls)
        self.fixed_order = bool(fixed_order)
        self.factor_clip = float(factor_clip)
        self.freeze_non_updating_agents = bool(freeze_non_updating_agents)

        self._bounds = _block_bounds(self.total_calls, self.n_agents)
        self._call = 0
        self._order = list(range(self.n_agents))
        self._position = 0
        # authoritative parameter copy for the freeze; see the callback above
        self._authoritative: Optional[Dict] = None
        self._freeze_usable: Optional[bool] = None

    # ------------------------------------------------------------------
    #  the permutation and the block schedule
    # ------------------------------------------------------------------

    def _advance(self) -> None:
        """Move ``self._position`` to the agent whose block contains this call."""
        call = self._call % self.total_calls
        if call == 0:
            # HARL: `agent_order = list(torch.randperm(self.num_agents).numpy())`
            # drawn once per training iteration, unless fixed_order is set.
            if self.fixed_order:
                self._order = list(range(self.n_agents))
            else:
                self._order = torch.randperm(self.n_agents).tolist()
            self._position = 0
            self._snapshot_all()
        else:
            while (
                self._position + 1 < self.n_agents
                and call >= self._bounds[self._position + 1]
            ):
                # the agent whose block just ended keeps the parameters it
                # reached; every other agent is held at the snapshot
                self._commit(self._order[self._position])
                self._position += 1
        self._call += 1

    def current_agent(self) -> int:
        return self._order[self._position]

    def updated_mask(self, device) -> torch.Tensor:
        """1.0 for agents whose block is already finished this iteration."""
        mask = torch.zeros(self.n_agents, device=device)
        for k in range(self._position):
            mask[self._order[k]] = 1.0
        return mask

    # ------------------------------------------------------------------
    #  the freeze
    # ------------------------------------------------------------------

    def _leaves(self):
        return _compat.leaf_params(self.actor_network_params)

    def _check_freeze_usable(self) -> bool:
        if self._freeze_usable is not None:
            return self._freeze_usable
        usable = True
        for key, leaf in self._leaves():
            if leaf.ndim == 0 or leaf.shape[0] != self.n_agents:
                usable = False
                warnings.warn(
                    "HAPPO: actor parameter "
                    f"{key} has shape {tuple(leaf.shape)}, which does not carry "
                    f"a leading agent dimension of {self.n_agents}. The "
                    "sequential-update freeze is disabled for this run; agents "
                    "outside the current block will drift by Adam momentum. "
                    "Check that experiment.share_policy_params=False and that "
                    "the policy model is a per-agent model."
                )
                break
        self._freeze_usable = usable
        return usable

    def _snapshot_all(self) -> None:
        if not self.freeze_non_updating_agents or not self._check_freeze_usable():
            return
        self._authoritative = {
            key: leaf.detach().clone() for key, leaf in self._leaves()
        }

    def _commit(self, agent: int) -> None:
        """Adopt the live parameters of ``agent`` into the authoritative copy."""
        if self._authoritative is None:
            return
        with torch.no_grad():
            for key, leaf in self._leaves():
                stored = self._authoritative.get(key)
                if stored is not None and stored.shape == leaf.shape:
                    stored[agent] = leaf.detach()[agent]

    def commit_current_agent(self) -> None:
        self._commit(self.current_agent())

    def restore_frozen_agents(self, exclude: Optional[int] = None) -> None:
        """Write the authoritative copy over every agent except ``exclude``."""
        if self._authoritative is None:
            return
        with torch.no_grad():
            for key, leaf in self._leaves():
                stored = self._authoritative.get(key)
                if stored is None or stored.shape != leaf.shape:
                    continue
                for agent in range(self.n_agents):
                    if agent != exclude:
                        leaf.data[agent] = stored[agent]

    # ------------------------------------------------------------------
    #  the objective
    # ------------------------------------------------------------------

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        self._advance()
        agent = self.current_agent()
        if self.freeze_non_updating_agents:
            self.restore_frozen_agents(exclude=agent)

        tensordict = tensordict.clone(False)
        advantage = tensordict.get(self.tensor_keys.advantage, None)
        if advantage is None:
            raise KeyError(
                f"HAPPO expects {self.tensor_keys.advantage} to be present; "
                "BenchMARL computes it in Algorithm.process_batch."
            )
        # normalize_advantage stays False: HAPPO standardises the advantage once
        # over the whole collected batch, in Happo.process_batch, exactly as
        # HARL's runner does, not per minibatch.

        log_weight, dist, kl_approx = _compat.log_weight(
            self, tensordict, advantage.shape[:-1]
        )

        # ---- the factor -------------------------------------------------
        #  HARL: factor = prod over the agents already updated of
        #  exp(new_logprob - old_logprob), recomputed after each agent's block
        #  and carried as a constant.  Summing the log-ratios of exactly those
        #  agents gives the same product with none of the overflow, and
        #  detaching makes it the constant HARL stores as numpy.
        done = self.updated_mask(log_weight.device)
        with torch.no_grad():
            log_factor = happo_log_factor(log_weight.detach(), done)
            factor = log_factor.exp()
            if self.factor_clip > 0.0:
                # NOT in HARL: a declared guard rail, off by default.  A product
                # of N ratios can overflow, and an inf factor puts a NaN in the
                # objective and ends the run.
                factor = factor.clamp(1.0 / self.factor_clip, self.factor_clip)

        # ---- the agent mask ---------------------------------------------
        agent_mask = torch.zeros(
            self.n_agents, 1, device=log_weight.device, dtype=advantage.dtype
        )
        agent_mask[agent] = 1.0

        weighted_advantage = advantage * factor * agent_mask

        gain1 = log_weight.exp() * weighted_advantage
        log_weight_clip = log_weight.clamp(*_compat.clip_bounds(self))
        gain2 = log_weight_clip.exp() * weighted_advantage
        gain = torch.stack([gain1, gain2], -1).min(dim=-1).values

        with torch.no_grad():
            clipped = (log_weight_clip != log_weight).to(log_weight.dtype)
            clip_fraction = (clipped * agent_mask).sum() / agent_mask.sum().clamp_min(
                1.0
            ) / max(int(log_weight.shape[0]), 1)

        td_out = TensorDict({"loss_objective": -gain.mean()}, batch_size=[])
        td_out.set("clip_fraction", clip_fraction)
        td_out.set("kl_approx", kl_approx.detach().mean())
        td_out.set(
            "happo_agent",
            torch.tensor(float(agent), device=gain.device, dtype=gain.dtype),
        )
        td_out.set("happo_factor", factor.detach().mean())

        if self.entropy_bonus:
            entropy = _compat.entropy(self, dist, advantage.shape[:-1])
            entropy = entropy * agent_mask
            td_out.set("entropy", entropy.detach().mean())
            td_out.set("loss_entropy", -_compat.entropy_coeff(self) * entropy.mean())
        if _compat.has_critic(self):
            critic_out = self.loss_critic(tensordict)
            loss_critic = critic_out[0] if isinstance(critic_out, tuple) else critic_out
            td_out.set("loss_critic", loss_critic.mean())
        return td_out


class Happo(Mappo):
    """HAPPO on BenchMARL's MAPPO host.

    Args:
        fixed_order (bool): HARL's ``fixed_order``. ``False`` draws a fresh
            random permutation of the agents every training iteration, which is
            what the monotone-improvement argument assumes.
        standardize_advantage (bool): standardise the advantage once over the
            whole collected batch before the agent blocks, as HARL's runner
            does. This is HAPPO's own normalisation and is separate from
            torchrl's per-minibatch ``normalize_advantage``, which stays off.
        factor_clip (float): NOT part of HAPPO. ``0.0`` reproduces HARL exactly.
            Any value ``c > 1`` clamps the factor to ``[1/c, c]`` as a guard
            against the product of N ratios overflowing.
        freeze_non_updating_agents (bool): hold the agents outside the current
            block exactly still. See :func:`_sequential_freeze_callback`.

    All other arguments are :class:`~benchmarl.algorithms.Mappo`'s and should be
    left at the values the MAPPO row uses, so that the comparison isolates the
    sequential update.
    """

    def __init__(
        self,
        fixed_order: bool,
        standardize_advantage: bool,
        factor_clip: float,
        freeze_non_updating_agents: bool,
        **kwargs,
    ):
        self.fixed_order = bool(fixed_order)
        self.standardize_advantage = bool(standardize_advantage)
        self.factor_clip = float(factor_clip)
        self.freeze_non_updating_agents = bool(freeze_non_updating_agents)
        super().__init__(**kwargs)

        if self.experiment_config.share_policy_params:
            raise ValueError(
                "HAPPO requires experiment.share_policy_params=False. The "
                "algorithm updates one agent at a time and multiplies the "
                "objective of the later agents by the ratio of the earlier "
                "ones; with a single shared policy every agent moves on every "
                "update and the sequential decomposition -- the whole content "
                "of the method -- does not exist. Launch with "
                "`experiment.share_policy_params=False`."
            )
        if self.has_rnn:
            raise NotImplementedError(
                "HAPPO here does not support recurrent models: the factor would "
                "have to be recomputed through the recurrent state of every "
                "already-updated agent. Use algorithm=mappo model=layers/gru "
                "for the memory baseline (BASELINES.md B2)."
            )

        self._happo_losses: Dict[str, HappoLoss] = {}
        _compat.attach_callback(
            self.experiment, _sequential_freeze_callback(self._happo_losses)
        )

    # ------------------------------------------------------------------

    def _n_minibatches(self) -> int:
        cfg = self.experiment_config
        return -(
            -cfg.train_batch_size(self.on_policy)
            // cfg.train_minibatch_size(self.on_policy)
        )

    def _total_calls_per_iteration(self) -> int:
        return self.experiment_config.n_optimizer_steps(self.on_policy) * (
            self._n_minibatches()
        )

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        n_agents = len(self.group_map[group])
        total_calls = self._total_calls_per_iteration()
        if total_calls < n_agents:
            raise ValueError(
                f"HAPPO needs at least one optimiser call per agent: group "
                f"{group!r} has {n_agents} agents but the experiment performs "
                f"{total_calls} optimiser calls per iteration. Raise "
                "experiment.on_policy_n_minibatch_iters."
            )
        bounds = _block_bounds(total_calls, n_agents)
        sizes = [bounds[k + 1] - bounds[k] for k in range(n_agents)]
        n_minibatches = self._n_minibatches()
        print(
            f"HAPPO {group}: {n_agents} agents, {total_calls} optimiser calls "
            f"per iteration -> blocks {sizes} "
            f"(= {[round(s / n_minibatches, 2) for s in sizes]} epochs each; "
            f"HARL's ppo_epoch default is 5). fixed_order={self.fixed_order} "
            f"standardize_advantage={self.standardize_advantage} "
            f"factor_clip={self.factor_clip if self.factor_clip else 'off (HARL)'} "
            f"freeze={self.freeze_non_updating_agents}"
        )

        loss_module = HappoLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            **_compat.coefficient_kwargs(
                HappoLoss, entropy_coef=self.entropy_coef, critic_coef=self.critic_coef
            ),
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
            n_agents=n_agents,
            total_calls=total_calls,
            fixed_order=self.fixed_order,
            factor_clip=self.factor_clip,
            freeze_non_updating_agents=self.freeze_non_updating_agents,
        )
        loss_module.set_keys(
            reward=(group, "reward"),
            action=(group, "action"),
            done=(group, "done"),
            terminated=(group, "terminated"),
            advantage=(group, "advantage"),
            value_target=(group, "value_target"),
            value=(group, "state_value"),
            sample_log_prob=(group, "log_prob"),
        )
        loss_module.make_value_estimator(
            ValueEstimators.GAE, gamma=self.experiment_config.gamma, lmbda=self.lmbda
        )
        self._happo_losses[group] = loss_module
        return loss_module, False

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        return {
            "loss_objective": list(loss.actor_network_params.flatten_keys().values()),
            "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
        }

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        batch = super().process_batch(group, batch)
        if self.standardize_advantage:
            #  HARL, on_policy_ha_runner.train / happo.train:
            #     advantages = (advantages - nanmean) / (nanstd + 1e-5)
            #  one mean and one std over the whole collected batch, all agents,
            #  before the first agent block.  np.nanstd is the population std,
            #  hence the explicit second moment rather than torch's default
            #  unbiased estimator.
            key = (group, "advantage")
            advantage = batch.get(key)
            mean = advantage.mean()
            std = ((advantage - mean) ** 2).mean().sqrt()
            batch.set(key, (advantage - mean) / (std + 1e-5))
        return batch

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        #  HARL backpropagates `(policy_loss - dist_entropy * entropy_coef)` in
        #  one call, which is MAPPO's convention in BenchMARL too.
        if "loss_entropy" in loss_vals.keys():
            loss_vals.set(
                "loss_objective",
                loss_vals["loss_objective"] + loss_vals["loss_entropy"],
            )
            del loss_vals["loss_entropy"]
        return loss_vals


@dataclass
class HappoConfig(MappoConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Happo`."""

    fixed_order: bool = MISSING
    standardize_advantage: bool = MISSING
    factor_clip: float = MISSING
    freeze_non_updating_agents: bool = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Happo
