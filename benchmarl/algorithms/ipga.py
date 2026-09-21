#  IPGA / INPG -- independent learning in PERFORMATIVE Markov potential games.
#
#      Sahitaj, Sasnauskas, Yalin, Mandal, Radanovic, "Independent Learning in
#      Performative Markov Potential Games", arXiv 2504.20593.
#
#  Why this baseline belongs here, and why it is not just another PG variant.
#
#  A PERFORMATIVE Markov game is one whose reward and transition kernels depend
#  on the DEPLOYED joint policy: G(pibar) = (N, S, A, r_{i,pibar}, P_{pibar},
#  gamma, rho).  `simple_ns` is exactly that and not by analogy: the
#  disturbance an agent feels is a weighted mean of its NEIGHBOURS' exertions,
#  so the transition kernel each agent faces is a function of what the other
#  agents' policies do -- change the deployed policies and you change the
#  environment.  That is the paper's setting, verbatim, and it means this row
#  is the one baseline here whose ASSUMPTIONS the instance actually satisfies.
#
#  The equilibrium concept is the PERFORMATIVELY STABLE EQUILIBRIUM: a policy
#  that is optimal in the environment IT ITSELF induces,
#
#      V_{i,pi}^{pi_i, pi_-i}(rho) >= V_{i,pi}^{pi'_i, pi_-i}(rho) - eps
#
#  -- note the subscript pi on V, which is what separates a PSE from a Nash
#  equilibrium of a fixed game.  The paper proves existence under a sensitivity
#  assumption (rewards and transitions Lipschitz in policy distance, constants
#  omega_r and omega_p) and gives two independent algorithms that reach it:
#
#      IPGA   pi^{t+1}_i(.|s) = argmax_pi <pi, Qbar^t_i(s,.)>
#                                          - 1/(2 eta) ||pi - pi^t_i(.|s)||^2
#             independent projected/proximal policy gradient ascent.
#             Best-iterate convergence to an approximate PSE.
#
#      INPG   pi^{t+1}_i(a|s) prop-to pi^t_i(a|s) exp(eta/(1-gamma) Abar^t_i(s,a))
#             independent natural policy gradient.  ASYMPTOTIC LAST-ITERATE
#             convergence, which is the stronger result and the reason both are
#             run.
#
#  Both are INDEPENDENT: no centralised critic, no shared parameters, no
#  communication.  The host is therefore IPPO and not MAPPO, and
#  `share_policy_params=false` is the launcher's setting for both rows.
#
#  Repeated retraining -- the paper's third procedure -- is what BenchMARL's
#  loop already does: collect under the deployed policy, optimise the surrogate
#  against THAT data, redeploy.  One deployment per iteration,
#  `experiment.on_policy_n_minibatch_iters` inner steps each.  The special case
#  the paper proves finite-time last-iterate convergence for optimises
#  OCCUPANCY MEASURES directly and is not implemented; see
#  baselines/docs/ipga.md.
#
#  See `baselines/docs/ipga.md`.

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Dict, Iterable, List, Optional, Tuple, Type

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms import _compat, _dists
from benchmarl.algorithms._baseline_math import (
    categorical_l2_sq,
    conjugate_gradients,
    gaussian_l2_sq,
)
from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.ippo import Ippo, IppoConfig


OLD_LOC_KEY = "ipga_old_loc"
OLD_SCALE_KEY = "ipga_old_scale"
OLD_PROBS_KEY = "ipga_old_probs"


class IpgaLoss(ClipPPOLoss):
    """IPGA's proximal step or INPG's natural-gradient step, per agent.

    ``variant="ipga"``
        The objective is evaluated on every minibatch and stepped by
        BenchMARL's optimiser, because a proximal argmax IS solved by gradient
        steps on the penalised objective.  The reference policy ``pi^t`` is the
        one that COLLECTED the batch -- its parameters are frozen into the
        batch by ``Ipga.process_batch`` before a single update has run -- so
        the proximal term is anchored to the deployed policy for the whole
        iteration, which is what the argmax says.

    ``variant="inpg"``
        One Fisher-preconditioned step per iteration, taken inside ``forward``
        on the whole rollout, exactly as LCPO's trust-region step is.  The
        actor is DELIBERATELY not given to an optimiser in that mode: a second,
        unpreconditioned Adam step down the same gradient is not a natural
        gradient.
    """

    #  Redeclared so torchrl's convert_to_functional does not warn: it checks
    #  the SUBCLASS's own __annotations__, which is empty unless the names are
    #  repeated here.  Same list torchrl's own losses carry.
    actor_network: TensorDictModule
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams
    critic_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_critic_network_params: TensorDictParams

    STAT_KEYS = (
        "ipga_prox",
        "ipga_surrogate",
        "ipga_dist_move",
        "ipga_step_norm",
        "ipga_param_move",
        "ipga_fisher_ok",
    )

    def __init__(
        self,
        *args,
        variant: str,
        group: str,
        n_agents: int,
        eta: float,
        gamma: float,
        damping: float,
        cg_iters: int,
        max_kl: float,
        nat_batch: int,
        seed: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.variant = str(variant)
        self.group = group
        self.n_agents = int(n_agents)
        self.eta = float(eta)
        self.gamma = float(gamma)
        self.damping = float(damping)
        self.cg_iters = int(cg_iters)
        self.max_kl = float(max_kl)
        self.nat_batch = int(nat_batch)
        self._generator = torch.Generator().manual_seed(int(seed))
        self._leaves: Optional[List[torch.Tensor]] = None
        self._pending: Optional[TensorDictBase] = None
        self._last_stats: Dict[str, float] = {}

    # ------------------------------------------------------------------
    #  flat parameter access over torchrl's functional actor parameters
    # ------------------------------------------------------------------

    def leaves(self) -> List[torch.Tensor]:
        if self._leaves is None:
            self._leaves = [v for _, v in self.actor_network_params.items(True, True)]
        return self._leaves

    def get_flat_params(self) -> torch.Tensor:
        return torch.cat([p.data.reshape(-1) for p in self.leaves()])

    def set_flat_params(self, flat: torch.Tensor) -> None:
        offset = 0
        with torch.no_grad():
            for p in self.leaves():
                n = p.numel()
                p.data.copy_(flat[offset : offset + n].view_as(p))
                offset += n

    def flat_grad(self, out: torch.Tensor, create_graph: bool = False):
        leaves = self.leaves()
        grads = torch.autograd.grad(
            out, leaves, create_graph=create_graph, allow_unused=True
        )
        return torch.cat(
            [
                (torch.zeros_like(p) if g is None else g).contiguous().reshape(-1)
                for p, g in zip(leaves, grads)
            ]
        )

    def _dist(self, observation: torch.Tensor):
        batch = observation.shape[0]
        td = TensorDict(
            {
                self.group: TensorDict(
                    {"observation": observation},
                    batch_size=[batch, self.n_agents],
                )
            },
            batch_size=[batch],
        )
        with self.actor_network_params.to_module(self.actor_network):
            return self.actor_network.get_dist(td)

    # ------------------------------------------------------------------
    #  the proximal term, in the paper's own metric
    # ------------------------------------------------------------------

    def proximal(self, tensordict: TensorDictBase, new_params) -> torch.Tensor:
        """``||pi - pi^t||_2^2``, per state, averaged.

        For a finite action set this is the paper's expression exactly: the
        squared Euclidean distance between the two probability vectors.  For a
        continuous action set it is the ``L2(da)`` distance between the two
        DENSITIES, which is the same functional with the sum over actions
        replaced by the integral, and which has a closed form for diagonal
        Gaussians (see ``_baseline_math.gaussian_l2_sq``).  No sampling: the
        quantity itself.
        """
        kind, params = new_params
        if kind == "categorical":
            old = tensordict.get((self.group, OLD_PROBS_KEY))
            return categorical_l2_sq(params, old)
        loc, scale = params
        old_loc = tensordict.get((self.group, OLD_LOC_KEY))
        old_scale = tensordict.get((self.group, OLD_SCALE_KEY))
        return gaussian_l2_sq(loc, scale, old_loc, old_scale)

    # ------------------------------------------------------------------
    #  the surrogate both variants ascend
    # ------------------------------------------------------------------

    def _surrogate(self, tensordict: TensorDictBase):
        """``E_{a ~ pi^t}[ (pi/pi^t) Abar ]``, and the current distribution.

        This is ``<pi, Qbar>`` up to a term that does not depend on ``pi``: the
        advantage differs from the marginalised action value by ``V(s)``, and
        ``sum_a pi(a|s) V(s) = V(s)`` whatever ``pi`` is, so the argmax is
        unchanged and the estimator has far lower variance.  ``Abar_i`` is the
        agent's OWN advantage, which with IPPO's independent critic is exactly
        the marginalised advantage the paper's update uses.
        """
        advantage = tensordict.get(self.tensor_keys.advantage)
        log_weight, dist, _ = _compat.log_weight(
            self, tensordict, adv_shape=advantage.shape[:-1]
        )
        ratio = log_weight.exp()
        return (ratio * advantage).mean(), dist

    # ------------------------------------------------------------------
    #  INPG: one natural-gradient step per iteration
    # ------------------------------------------------------------------

    def stash_batch(self, batch: TensorDictBase) -> None:
        self._pending = batch

    def _blank_stats(self) -> Dict[str, float]:
        return {key: 0.0 for key in self.STAT_KEYS}

    def inpg_step(self) -> Dict[str, float]:
        batch = self._pending
        self._pending = None
        stats = self._blank_stats()

        observation = batch.get((self.group, "observation"))
        flat = observation.reshape(-1, self.n_agents, observation.shape[-1])
        if 0 < self.nat_batch < flat.shape[0]:
            keep = torch.randperm(flat.shape[0], generator=self._generator)[
                : self.nat_batch
            ].to(flat.device)
            local = flat[keep].detach()
        else:
            local = flat.detach()

        with torch.no_grad():
            before = _dists.params_of(self._dist(local))

        surrogate, _ = self._surrogate(batch)
        #  Ascend the surrogate, so the gradient of the LOSS is its negative.
        grad = self.flat_grad(-surrogate).detach()

        def fisher_vector_product(v: torch.Tensor) -> torch.Tensor:
            kl = _dists.kl_of(before, _dists.params_of(self._dist(local))).mean()
            flat_grad_kl = self.flat_grad(kl, create_graph=True)
            kl_v = (flat_grad_kl * v).sum()
            return self.flat_grad(kl_v).detach() + v * self.damping

        #  F^-1 g, the natural gradient.  For a softmax parameterisation this
        #  step IS the multiplicative update in the paper's Section 4 (Kakade;
        #  Agarwal et al.), which is why the closed form is checked against
        #  this one in baselines/verify.py.
        direction = conjugate_gradients(
            fisher_vector_product, -grad, self.cg_iters
        )
        step = (self.eta / (1.0 - self.gamma)) * direction

        if self.max_kl > 0.0:
            #  NOT part of INPG.  Off by default.  A declared safety bound for
            #  a step that the paper takes at a fixed size and that this
            #  instance can blow up on; see baselines/docs/ipga.md.
            quad = float((step * fisher_vector_product(step)).sum())
            if quad > 0.0:
                scale = min(1.0, (2.0 * self.max_kl / quad) ** 0.5)
                step = step * scale

        prev = self.get_flat_params()
        self.set_flat_params(prev + step)
        with torch.no_grad():
            after = _dists.params_of(self._dist(local))
            stats["ipga_dist_move"] = float(
                self.proximal_from_params(before, after).mean()
            )
        stats["ipga_surrogate"] = float(surrogate.detach())
        stats["ipga_step_norm"] = float(step.norm())
        stats["ipga_param_move"] = float((self.get_flat_params() - prev).norm())
        stats["ipga_fisher_ok"] = float(torch.isfinite(direction).all())
        return stats

    @staticmethod
    def proximal_from_params(old, new) -> torch.Tensor:
        kind, old_p = old
        _, new_p = new
        if kind == "categorical":
            return categorical_l2_sq(new_p, old_p)
        return gaussian_l2_sq(*new_p, *old_p)

    # ------------------------------------------------------------------

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        #  Cloned for the same reason HAPPO's forward clones: torchrl's
        #  `_log_weight` runs the actor over the tensordict and writes keys
        #  into it, and the same minibatch is handed back on the next
        #  optimiser call.
        tensordict = tensordict.clone(False)
        if self.variant == "inpg":
            if self._pending is not None:
                self._last_stats = self.inpg_step()
            stats = self._last_stats or self._blank_stats()
            critic_out = self.loss_critic(tensordict)
            loss_critic = (
                critic_out[0] if isinstance(critic_out, tuple) else critic_out
            )
            td_out = TensorDict({"loss_critic": loss_critic.mean()}, batch_size=[])
        else:
            surrogate, dist = self._surrogate(tensordict)
            prox = self.proximal(tensordict, _dists.params_of(dist)).mean()
            #  argmax <pi, Qbar> - 1/(2 eta) ||pi - pi^t||^2, as a loss.
            loss_objective = -surrogate + prox / (2.0 * self.eta)
            critic_out = self.loss_critic(tensordict)
            loss_critic = (
                critic_out[0] if isinstance(critic_out, tuple) else critic_out
            )
            td_out = TensorDict(
                {
                    "loss_objective": loss_objective,
                    "loss_critic": loss_critic.mean(),
                },
                batch_size=[],
            )
            stats = self._blank_stats()
            stats["ipga_prox"] = float(prox.detach())
            stats["ipga_surrogate"] = float(surrogate.detach())
            stats["ipga_dist_move"] = float(prox.detach())

        for key, value in stats.items():
            td_out.set(
                key,
                torch.as_tensor(
                    value, device=loss_critic.device, dtype=torch.float32
                ),
            )
        return td_out


class Ipga(Ippo):
    """IPGA / INPG on BenchMARL's IPPO host.

    Args:
        variant (str): ``"ipga"`` is the proximal (projected) gradient ascent
            of the paper's Algorithm 1; ``"inpg"`` is the independent natural
            policy gradient, the one with last-iterate convergence.
        eta (float): the step size ``eta``. In IPGA it appears as the proximal
            weight ``1/(2 eta)``; in INPG as the multiplier ``eta/(1-gamma)``
            on the natural gradient. A swept quantity in the paper, not a
            published constant.
        damping (float): Fisher damping for INPG's conjugate gradients.
        cg_iters (int): conjugate-gradient iterations for INPG.
        max_kl (float): OFF (``0``) by default and NOT part of the method: a
            declared cap on the INPG step, for a run where the unconstrained
            natural gradient diverges.
        nat_batch (int): cap on the joint states used for INPG's Fisher solve.
            ``0`` uses the whole rollout.

    All other arguments are :class:`~benchmarl.algorithms.Ippo`'s.
    """

    def __init__(
        self,
        variant: str,
        eta: float,
        damping: float,
        cg_iters: int,
        max_kl: float,
        nat_batch: int,
        **kwargs,
    ):
        self.variant = str(variant)
        if self.variant not in ("ipga", "inpg"):
            raise ValueError(
                f"variant must be 'ipga' or 'inpg'; got {variant!r}"
            )
        self.eta = float(eta)
        self.damping = float(damping)
        self.cg_iters = int(cg_iters)
        self.max_kl = float(max_kl)
        self.nat_batch = int(nat_batch)
        super().__init__(**kwargs)

        if self.eta <= 0.0:
            raise ValueError(f"eta must be positive; got {self.eta}")
        if self.has_rnn:
            raise NotImplementedError(
                "IPGA/INPG here do not support recurrent models: the proximal "
                "term and the Fisher are both evaluated by re-running the "
                "policy on stored observations, which a recurrent policy "
                "cannot be evaluated on without its hidden state."
            )
        if self.experiment_config.share_policy_params:
            import warnings

            warnings.warn(
                "IPGA/INPG are INDEPENDENT learners: every agent runs its own "
                "update on its own parameters, and the convergence results are "
                "about what that does to the joint policy. With "
                "experiment.share_policy_params=True there is one policy and "
                "one update, so 'independent' is vacuous. Launch with "
                "share_policy_params=false."
            )
        self._losses: Dict[str, IpgaLoss] = {}

    # ------------------------------------------------------------------

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        n_agents = len(self.group_map[group])
        print(
            f"IPGA {group}: variant={self.variant}, eta={self.eta} "
            + (
                f"(proximal weight 1/(2 eta) = {1.0 / (2.0 * self.eta):.4g} on "
                "the squared L2 distance between the new and deployed action "
                "distributions)"
                if self.variant == "ipga"
                else f"(natural-gradient multiplier eta/(1-gamma) = "
                f"{self.eta / (1.0 - self.experiment_config.gamma):.4g}, "
                f"damping={self.damping}, cg_iters={self.cg_iters}, "
                f"max_kl={self.max_kl or 'off'}, "
                f"nat_batch={self.nat_batch or 'whole rollout'})"
            )
            + "; INDEPENDENT critic (IPPO), so the advantage each agent sees "
            "IS the marginalised advantage the paper's update takes"
        )
        loss_module = IpgaLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            **_compat.coefficient_kwargs(
                IpgaLoss, entropy_coef=0.0, critic_coef=self.critic_coef
            ),
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
            variant=self.variant,
            group=group,
            n_agents=n_agents,
            eta=self.eta,
            gamma=self.experiment_config.gamma,
            damping=self.damping,
            cg_iters=self.cg_iters,
            max_kl=self.max_kl,
            nat_batch=self.nat_batch,
            seed=self.experiment.seed,
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
        self._losses[group] = loss_module
        return loss_module, False

    def _get_parameters(self, group: str, loss: LossModule) -> Dict[str, Iterable]:
        params = {
            "loss_critic": list(loss.critic_network_params.flatten_keys().values())
        }
        if self.variant == "ipga":
            params["loss_objective"] = list(
                loss.actor_network_params.flatten_keys().values()
            )
        #  For INPG the actor is DELIBERATELY absent: its step is the
        #  Fisher-preconditioned one taken inside the loss forward, and an Adam
        #  optimiser bound here would take a second, unpreconditioned step down
        #  the same gradient. Same reasoning as Lcpo._get_parameters.
        return params

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        batch = super().process_batch(group, batch)
        loss = self._losses[group]
        self._freeze_deployed_policy(group, loss, batch)
        if self.variant == "inpg":
            loss.stash_batch(batch.detach())
        return batch

    def _freeze_deployed_policy(
        self, group: str, loss: IpgaLoss, batch: TensorDictBase
    ) -> None:
        """Write ``pi^t``'s parameters into the batch, before any update.

        IPGA's proximal term is anchored at the policy that was DEPLOYED --
        ``pi^t``, the one whose induced environment produced this data, which
        is the whole content of "performative".  Anchoring it at the previous
        minibatch's policy instead would make it a smoothing term and the
        algorithm something else.  ``process_batch`` runs once per iteration,
        before the batch reaches the buffer, so the values written here ride
        along into every minibatch of this iteration.
        """
        observation = batch.get((group, "observation"))
        shape = observation.shape[:-1]
        flat = observation.reshape(-1, *observation.shape[-2:])
        with torch.no_grad():
            kind, params = _dists.params_of(loss._dist(flat))
        if kind == "categorical":
            batch.set((group, OLD_PROBS_KEY), params.reshape(*shape, -1))
        else:
            loc, scale = params
            batch.set((group, OLD_LOC_KEY), loc.reshape(*shape, -1))
            batch.set((group, OLD_SCALE_KEY), scale.reshape(*shape, -1))

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        return loss_vals


@dataclass
class IpgaConfig(IppoConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Ipga`."""

    variant: str = MISSING
    eta: float = MISSING
    damping: float = MISSING
    cg_iters: int = MISSING
    max_kl: float = MISSING
    nat_batch: int = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Ipga
