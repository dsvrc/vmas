#  LCPO -- Locally Constrained Policy Optimisation.
#
#      Hamadanian, Gao, Schwarzkopf, Alizadeh, "Online Reinforcement Learning in
#      Non-Stationary Context-Driven Environments", ICLR 2025.
#      Reference code: pouyahmdn/LCPO --
#        `windy-gym/agent/core_alg/core_lcpo.py`   (locopo, trpo_step, linesearch)
#        `windy-gym/agent/core_alg/core_trpo.py`   (conjugate_gradients)
#        `windy-gym/buffer/buffer_ood.py`          (OutOfDSampler)
#        `windy-gym/agent/lcpo.py`                 (the training loop)
#
#  BASELINES.md B6: the most direct recent competitor -- non-stationarity
#  induced by an EXOGENOUS OBSERVED CONTEXT, which is exactly the driver A(t),
#  with a method designed for it.
#
#  LCPO is a trust-region method with TWO constraints:
#
#      maximise  E_local[ -A * pi_new/pi_old ]   +  entropy
#      s.t.      KL over the CURRENT context   <= kl_in     (the usual region)
#                KL over OUT-OF-DISTRIBUTION contexts <= kl_out
#
#  The second constraint is the method.  States whose CONTEXT is far from the
#  context the agent is currently living in are sampled from a reservoir of
#  everything it has ever seen, and the policy is forbidden to move much on
#  them.  That is what stops a context-driven learner from forgetting the wet
#  half of the cycle while it trains through the dry half.
#
#  REQUIRES `task.ns_observe_driver=true`: the context must be observed.  The
#  algorithm raises if the context slice cannot be located.
#
#  See `baselines/docs/lcpo.md` for the clause-by-clause checklist, the
#  single-agent-to-multi-agent lift, and what is not implemented.

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Type

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms._baseline_math import (
    categorical_kl,
    conjugate_gradients,
    gaussian_kl,
    get_qu,
    OutOfDistributionSampler,
)
from benchmarl.algorithms import _compat  # submodule import: safe while
                                          # benchmarl.algorithms is still
                                          # being initialised
from benchmarl.algorithms.common import Algorithm, AlgorithmConfig
from benchmarl.algorithms.ippo import Ippo, IppoConfig


# ===========================================================================
#  the loss
# ===========================================================================


class LcpoLoss(ClipPPOLoss):
    """IPPO's critic machinery; LCPO's policy step.

    BenchMARL drives training through ``loss -> backward -> optimizer.step()``,
    which cannot express a trust-region step: TRPO does not descend a gradient,
    it solves a constrained quadratic and then writes the parameters directly.
    So the policy step happens INSIDE this forward and the algorithm hands
    BenchMARL an optimiser for the critic only -- the returned
    ``loss_objective`` is a detached number for the log, with no optimiser
    bound to it, so nothing steps the actor twice.
    """

    def __init__(
        self,
        *args,
        group: str,
        n_agents: int,
        context_slice: slice,
        kl_in: float,
        kl_out: float,
        damping: float,
        solve_dual: bool,
        cg_iters: int,
        max_backtracks: int,
        accept_ratio: float,
        entropy_max: float,
        entropy_min: float,
        entropy_decay: float,
        ood_window: int,
        ood_capacity: int,
        ood_subsample: int,
        ood_threshold: float,
        trpo_batch: int,
        fallback_lr: float,
        grad_clip: float,
        seed: int,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.group = group
        self.n_agents = int(n_agents)
        self.context_slice = context_slice
        self.kl_in = float(kl_in)
        self.kl_out = float(kl_out)
        self.damping = float(damping)
        self.solve_dual = bool(solve_dual)
        self.cg_iters = int(cg_iters)
        self.max_backtracks = int(max_backtracks)
        self.accept_ratio = float(accept_ratio)
        self.entropy_factor = float(entropy_max)
        self.entropy_min = float(entropy_min)
        self.entropy_decay = float(entropy_decay)
        self.ood_window = int(ood_window)
        self.ood_capacity = int(ood_capacity)
        self.ood_subsample = max(int(ood_subsample), 1)
        self.ood_threshold = float(ood_threshold)
        self.trpo_batch = int(trpo_batch)
        self.grad_clip = float(grad_clip)

        self._generator = torch.Generator().manual_seed(int(seed))
        self._buffer: Optional[OutOfDistributionSampler] = None
        self._leaves: Optional[List[torch.Tensor]] = None
        self._fallback_opt: Optional[torch.optim.Optimizer] = None
        self._fallback_lr = float(fallback_lr)
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

    # ------------------------------------------------------------------
    #  distributions
    # ------------------------------------------------------------------

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

    @staticmethod
    def _entropy(dist) -> torch.Tensor:
        """``dist.entropy()`` where it exists, one-sample Monte Carlo where not.

        TanhNormal is a transformed distribution with no closed-form entropy;
        torchrl's own PPO loss falls back to the same estimator.
        """
        try:
            entropy = dist.entropy()
            if entropy.isfinite().all():
                return entropy
        except NotImplementedError:
            pass
        sample = (
            dist.rsample() if getattr(dist, "has_rsample", False) else dist.sample()
        )
        return -dist.log_prob(sample)

    @staticmethod
    def _params_of(dist):
        probs = getattr(dist, "probs", None)
        if probs is not None:
            return ("categorical", probs)
        loc = getattr(dist, "loc", None)
        scale = getattr(dist, "scale", None)
        if loc is None or scale is None:
            base = dist
            while hasattr(base, "base_dist"):
                base = base.base_dist
            loc, scale = base.loc, base.scale
        return ("gaussian", (loc, scale))

    @staticmethod
    def _kl(old, new) -> torch.Tensor:
        """``KL(pi_old || pi_new)``, summed over action dims."""
        kind, old_p = old
        _, new_p = new
        if kind == "categorical":
            return categorical_kl(old_p, new_p)
        return gaussian_kl(*old_p, *new_p)

    # ------------------------------------------------------------------
    #  the LCPO step
    # ------------------------------------------------------------------

    def stash_batch(self, batch: TensorDictBase) -> None:
        """Called once per collection iteration with the WHOLE batch.

        LCPO takes one policy step per rollout, on the whole rollout. BenchMARL
        hands a loss minibatches, so the algorithm stashes the full batch here
        and the first optimiser call of the iteration consumes it.
        """
        self._pending = batch

    #: Emitted on EVERY optimiser call, with the same keys in the same order,
    #: because BenchMARL stacks the per-call tensordicts and a key that appears
    #: only on one branch breaks the stack.
    STAT_KEYS = (
        "lcpo/branch",
        "lcpo/loss",
        "lcpo/lm",
        "lcpo/linesearch_ok",
        "lcpo/step_norm",
        "lcpo/kl_in_d",
        "lcpo/kl_out_of_d",
        "lcpo/ood_batch",
        "lcpo/ood_seen",
        "lcpo/entropy",
        "lcpo/entropy_factor",
    )

    def _blank_stats(self) -> Dict[str, float]:
        return {key: 0.0 for key in self.STAT_KEYS}

    def lcpo_step(self) -> Dict[str, float]:
        batch = self._pending
        self._pending = None
        stats = self._blank_stats()

        observation = batch.get((self.group, "observation"))
        action = batch.get((self.group, "action"))
        advantage = batch.get((self.group, "advantage"))
        flat = observation.reshape(-1, self.n_agents, observation.shape[-1])
        flat_action = action.reshape(-1, self.n_agents, action.shape[-1])
        flat_adv = advantage.reshape(-1, self.n_agents, advantage.shape[-1])

        if self._buffer is None:
            self._buffer = OutOfDistributionSampler(
                n_agents=self.n_agents,
                obs_dim=flat.shape[-1],
                window=max(self.ood_window, 1),
                capacity=self.ood_capacity,
                context_slice=self.context_slice,
                threshold=self.ood_threshold,
                device=flat.device,
                generator=self._generator,
            )
        #  `self.ood_buf.add_many_exp(obs_np)` -- every state the agent saw this
        #  rollout goes into the reservoir, before the OOD draw.
        self._buffer.add(flat[:: self.ood_subsample].detach())

        if 0 < self.trpo_batch < flat.shape[0]:
            keep = torch.randperm(flat.shape[0], generator=self._generator)[
                : self.trpo_batch
            ].to(flat.device)
            local = flat[keep].detach()
            local_action = flat_action[keep].detach()
            local_adv = flat_adv[keep].detach()
        else:
            local = flat.detach()
            local_action = flat_action.detach()
            local_adv = flat_adv.detach()

        ood = self._buffer.get(local.shape[0])

        #  frozen references, `with torch.no_grad(): ... _before`
        with torch.no_grad():
            dist_local_before = self._dist(local)
            params_local_before = self._params_of(dist_local_before)
            log_pi_before = dist_local_before.log_prob(local_action)
            entropy_before = float(self._entropy(dist_local_before).mean())
            params_ood_before = (
                self._params_of(self._dist(ood)) if ood is not None else None
            )

        def get_loss(volatile: bool = False) -> torch.Tensor:
            context = torch.no_grad() if volatile else torch.enable_grad()
            with context:
                dist = self._dist(local)
                log_pi = dist.log_prob(local_action)
                entropy = self._entropy(dist)
                #  core_lcpo.get_loss_lcpo, with the advantage's trailing
                #  singleton dropped so it lines up with the log-prob.
                action_loss = -local_adv.squeeze(-1) * torch.exp(
                    log_pi - log_pi_before
                )
                return action_loss.mean() - entropy.mean() * self.entropy_factor

        def kl_in() -> torch.Tensor:
            return self._kl(params_local_before, self._params_of(self._dist(local)))

        def kl_out() -> torch.Tensor:
            return self._kl(params_ood_before, self._params_of(self._dist(ood)))

        if ood is None:
            #  `if len(ood_obs_np) > 0: locopo(...) else: policy_gradient(...)`
            #  -- plain advantage actor-critic until the reservoir holds states
            #  from a context far enough from the present one.
            stats.update(self._policy_gradient_step(get_loss))
        else:
            stats.update(self._trpo_step(get_loss, kl_out, kl_in))
            stats["lcpo/ood_batch"] = float(ood.shape[0])
            with torch.no_grad():
                stats["lcpo/kl_out_of_d"] = float(kl_out().mean())
                stats["lcpo/kl_in_d"] = float(kl_in().mean())

        #  `tune_entropy`: linear decay, once per rollout.
        self.entropy_factor = max(
            self.entropy_factor - self.entropy_decay, self.entropy_min
        )
        stats["lcpo/entropy_factor"] = self.entropy_factor
        stats["lcpo/entropy"] = entropy_before
        stats["lcpo/ood_seen"] = float(self._buffer.n_seen)
        return stats

    def _policy_gradient_step(self, get_loss) -> Dict[str, float]:
        if self._fallback_opt is None:
            #  `net_opt_p = Adam(policy_net.parameters(), lr, weight_decay=1e-4,
            #  eps=1e-5)` -- LCPO's own actor optimiser, used only on this
            #  branch.  It lives here rather than in BenchMARL's optimiser dict
            #  because on the LCPO branch the actor must NOT be stepped by a
            #  gradient at all.
            self._fallback_opt = torch.optim.Adam(
                self.leaves(), lr=self._fallback_lr, weight_decay=1e-4, eps=1e-5
            )
        loss = get_loss()
        self._fallback_opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.leaves(), self.grad_clip)
        self._fallback_opt.step()
        return {"lcpo/branch": 0.0, "lcpo/loss": float(loss.detach())}

    def _linesearch(self, get_loss, kl_out, kl_in, x_old, fullstep, expected):
        """``core_lcpo.linesearch``: both constraints must hold, unchanged."""
        fval = get_loss(True).detach()
        for n in range(self.max_backtracks):
            stepfrac = 0.5**n
            x_new = x_old + stepfrac * fullstep
            self.set_flat_params(x_new)
            newfval = get_loss(True).detach()
            with torch.no_grad():
                new_kl_in = kl_in().mean()
                new_kl_out = kl_out().mean()
            actual = fval - newfval
            expected_improve = expected * stepfrac
            ratio = actual / expected_improve if expected_improve != 0 else actual
            if (
                float(ratio) > self.accept_ratio
                and float(actual) > 0
                and float(new_kl_out) <= self.kl_out
                and float(new_kl_in) <= self.kl_in
            ):
                return True, x_new
        return False, x_old

    def _trpo_step(self, get_loss, kl_out, kl_in) -> Dict[str, float]:
        loss = get_loss()
        loss_grad = self.flat_grad(loss).detach()

        def quad(kl_func):
            def avp(v: torch.Tensor) -> torch.Tensor:
                kl = kl_func().mean()
                flat_grad_kl = self.flat_grad(kl, create_graph=True)
                kl_v = (flat_grad_kl * v).sum()
                return self.flat_grad(kl_v).detach() + v * self.damping

            return avp

        q_out, q_in = quad(kl_out), quad(kl_in)

        step_out = conjugate_gradients(q_out, -loss_grad, self.cg_iters)
        vout = float(-(loss_grad * step_out).sum())
        lm = 0.0
        if vout <= 0:
            fullstep = torch.zeros_like(step_out)
        elif self.solve_dual:
            step_in = conjugate_gradients(q_in, -loss_grad, self.cg_iters)
            vin = float(-(loss_grad * step_in).sum())
            a_in = [0.5 * vin, vout, 0.5 * float((step_out * q_in(step_out)).sum())]
            a_out = [
                0.5 * float((step_in * q_out(step_out)).sum()),
                vin,
                0.5 * vout,
            ]
            diff = [
                a_in[k] / self.kl_in - a_out[k] / self.kl_out for k in range(3)
            ]
            if diff[0] == 0:
                fullstep = (2 * self.kl_out / vout) ** 0.5 * step_out
            else:
                diff = [d / diff[0] for d in diff]
                if diff[1] ** 2 <= 4 * diff[2] or max(get_qu(*diff)) <= 0:
                    fullstep = (2 * self.kl_out / vout) ** 0.5 * step_out
                else:
                    s = max(get_qu(*diff))
                    l_out = (
                        self.kl_in / (a_in[0] * s**2 + a_in[1] * s + a_in[0])
                    ) ** 0.5
                    fullstep = (l_out * s) * step_in + l_out * step_out
                    lm = s
        else:
            fullstep = (2 * self.kl_out / vout) ** 0.5 * step_out

        if vout <= 0:
            return {"lcpo/branch": 1.0, "lcpo/loss": float(loss.detach())}

        expected = (-loss_grad * fullstep).sum()
        prev = self.get_flat_params()
        ok, new_params = self._linesearch(
            get_loss, kl_out, kl_in, prev, fullstep, expected
        )
        self.set_flat_params(new_params)
        return {
            "lcpo/branch": 1.0,
            "lcpo/loss": float(loss.detach()),
            "lcpo/lm": float(lm),
            "lcpo/linesearch_ok": float(ok),
            "lcpo/step_norm": float((new_params - prev).abs().mean()),
        }

    # ------------------------------------------------------------------

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self._pending is not None:
            self._last_stats = self.lcpo_step()
        stats = self._last_stats or self._blank_stats()

        critic_out = self.loss_critic(tensordict)
        loss_critic = critic_out[0] if isinstance(critic_out, tuple) else critic_out

        td_out = TensorDict({"loss_critic": loss_critic.mean()}, batch_size=[])
        #  Detached on purpose: `Lcpo._get_parameters` binds an optimiser to
        #  `loss_critic` and to nothing else, so this is a log column.
        for key, value in stats.items():
            td_out.set(
                key.replace("/", "_"),
                torch.as_tensor(value, device=loss_critic.device, dtype=torch.float32),
            )
        return td_out


class Lcpo(Ippo):
    """LCPO with independent per-agent learners on BenchMARL's IPPO host.

    Args:
        context_start (int): index of the first context component in the
            agent's observation. Negative indexes from the end, so the default
            ``-1`` picks the driver appended by ``task.ns_observe_driver=true``.
        context_size (int): how many components the context has.
        kl_in, kl_out (float): the two trust-region radii. LCPO's defaults are
            ``1e-1`` and ``1e-3``.
        damping (float): Fisher damping, LCPO's ``trpo_damping``.
        solve_dual (bool): LCPO's ``--trpo_dual``. ``False`` steps along the
            out-of-distribution direction alone; ``True`` solves the two-
            constraint dual, which is the version the paper reports.
        ood_threshold (float): LCPO's ``lcpo_thresh`` with ``lcpo_ood_type=l2``.
        ood_subsample (int): keep one state in ``n`` for the reservoir, LCPO's
            ``ood_subsample``.
        trpo_batch (int): cap on the number of joint states used per
            trust-region step. ``0`` uses the whole rollout, which is LCPO's
            behaviour; a positive value bounds the cost of the conjugate
            gradients, which run over the whole set at every iteration.
    """

    def __init__(
        self,
        context_start: int,
        context_size: int,
        kl_in: float,
        kl_out: float,
        damping: float,
        solve_dual: bool,
        cg_iters: int,
        max_backtracks: int,
        accept_ratio: float,
        entropy_max: float,
        entropy_min: float,
        entropy_decay: float,
        ood_window: int,
        ood_capacity: int,
        ood_subsample: int,
        ood_threshold: float,
        trpo_batch: int,
        grad_clip: float,
        **kwargs,
    ):
        self.context_start = int(context_start)
        self.context_size = int(context_size)
        self.kl_in = float(kl_in)
        self.kl_out = float(kl_out)
        self.damping = float(damping)
        self.solve_dual = bool(solve_dual)
        self.cg_iters = int(cg_iters)
        self.max_backtracks = int(max_backtracks)
        self.accept_ratio = float(accept_ratio)
        self.entropy_max = float(entropy_max)
        self.entropy_min = float(entropy_min)
        self.entropy_decay = float(entropy_decay)
        self.ood_window = int(ood_window)
        self.ood_capacity = int(ood_capacity)
        self.ood_subsample = int(ood_subsample)
        self.ood_threshold = float(ood_threshold)
        self.trpo_batch = int(trpo_batch)
        self.grad_clip = float(grad_clip)
        super().__init__(**kwargs)

        if self.has_rnn:
            raise NotImplementedError(
                "LCPO here does not support recurrent models: the trust region "
                "is evaluated on a resampled set of stored states, which a "
                "recurrent policy cannot be evaluated on without its hidden "
                "state."
            )
        self._losses: Dict[str, LcpoLoss] = {}

    def _context_slice(self, group: str) -> slice:
        spec = self.observation_spec[group, "observation"]
        obs_dim = int(spec.shape[-1])
        start = self.context_start
        if start < 0:
            start += obs_dim
        stop = start + self.context_size
        if not (0 <= start < stop <= obs_dim):
            raise ValueError(
                f"LCPO context slice [{start}, {stop}) does not fit an "
                f"observation of width {obs_dim} for group {group!r}. LCPO is "
                "defined for an OBSERVED context: set "
                "`task.ns_observe_driver=true` so the driver A(t) is appended "
                "to the observation, and point context_start/context_size at "
                "it."
            )
        return slice(start, stop)

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        context_slice = self._context_slice(group)
        n_agents = len(self.group_map[group])
        print(
            f"LCPO {group}: context = observation[..., {context_slice.start}:"
            f"{context_slice.stop}], kl_in={self.kl_in} kl_out={self.kl_out} "
            f"dual={self.solve_dual} damping={self.damping} "
            f"ood_threshold={self.ood_threshold} "
            f"ood_capacity={self.ood_capacity} subsample={self.ood_subsample} "
            f"trpo_batch={self.trpo_batch or 'whole rollout'}"
        )
        loss_module = LcpoLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            **_compat.coefficient_kwargs(
                LcpoLoss, entropy_coef=0.0, critic_coef=self.critic_coef
            ),
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
            group=group,
            n_agents=n_agents,
            context_slice=context_slice,
            kl_in=self.kl_in,
            kl_out=self.kl_out,
            damping=self.damping,
            solve_dual=self.solve_dual,
            cg_iters=self.cg_iters,
            max_backtracks=self.max_backtracks,
            accept_ratio=self.accept_ratio,
            entropy_max=self.entropy_max,
            entropy_min=self.entropy_min,
            entropy_decay=self.entropy_decay,
            ood_window=self.ood_window,
            ood_capacity=self.ood_capacity,
            ood_subsample=self.ood_subsample,
            ood_threshold=self.ood_threshold,
            trpo_batch=self.trpo_batch,
            fallback_lr=self.experiment_config.lr,
            grad_clip=self.grad_clip,
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
        #  THE ACTOR IS DELIBERATELY ABSENT.  LCPO's policy step is a
        #  trust-region solve performed inside the loss forward; binding an Adam
        #  optimiser to the actor here would step it a second time, down the
        #  gradient the trust region just refused to follow.
        return {
            "loss_critic": list(loss.critic_network_params.flatten_keys().values()),
        }

    def process_batch(self, group: str, batch: TensorDictBase) -> TensorDictBase:
        batch = super().process_batch(group, batch)
        #  LCPO takes ONE policy step per rollout, on the whole rollout.
        self._losses[group].stash_batch(batch.detach())
        return batch

    def process_loss_vals(
        self, group: str, loss_vals: TensorDictBase
    ) -> TensorDictBase:
        return loss_vals


@dataclass
class LcpoConfig(IppoConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Lcpo`."""

    context_start: int = MISSING
    context_size: int = MISSING
    kl_in: float = MISSING
    kl_out: float = MISSING
    damping: float = MISSING
    solve_dual: bool = MISSING
    cg_iters: int = MISSING
    max_backtracks: int = MISSING
    accept_ratio: float = MISSING
    entropy_max: float = MISSING
    entropy_min: float = MISSING
    entropy_decay: float = MISSING
    ood_window: int = MISSING
    ood_capacity: int = MISSING
    ood_subsample: int = MISSING
    ood_threshold: float = MISSING
    trpo_batch: int = MISSING
    grad_clip: float = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Lcpo
