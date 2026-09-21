#  DORAEMON -- Domain RAndomization via Entropy MaximizatiON.
#
#      Tiboni, Klink, Peters, Tommasi, D'Eramo, Chalvatzaki, "Domain
#      Randomization via Entropy Maximization", ICLR 2024 (arXiv 2311.01885).
#      Code: gabrieletiboni/doraemon (doraemon/doraemon/doraemon.py).
#
#  BASELINES.md B9's must-run row is FIXED domain randomisation: draw
#  sigma ~ U[0, 3] per episode and train through it.  Its weakness is the one
#  every DR paper opens with -- too little variability and the policy does not
#  generalise, too much and it goes conservative -- and the range is a guess
#  nobody can justify.  DORAEMON removes the guess.  It solves, between
#  training rounds,
#
#      min_phi  KL( phi || phi_target )
#      s.t.     E_phi[ 1(return >= tau) ] >= alpha         (performance)
#               KL( phi_i || phi ) <= epsilon              (trust region)
#
#  where phi is a Beta on the declared severity range and phi_target is the
#  maximum-entropy distribution on it (the uniform).  Minimising the KL to the
#  uniform IS maximising entropy, subject to the policy still solving the task
#  often enough.  The distribution therefore starts narrow and WIDENS exactly
#  as fast as the policy can take it -- a curriculum over severity that nobody
#  has to tune.
#
#  The performance constraint is evaluated by IMPORTANCE SAMPLING over the
#  episodes just collected, so testing a candidate distribution costs no
#  environment steps: that is the trick that makes the outer loop affordable,
#  and it is why the environment has to report the severity each episode was
#  actually run at (`info["ns_sigma"]`, which it already did for B9).
#
#  Two halves, in two places:
#
#      this file            the outer loop and the constrained solve
#      simple_ns/dr_state   the distribution the environment draws from,
#                           published here and read at every episode reset
#
#  and the environment side is inert unless `task.ns_dr_dist=beta`.
#
#  REQUIRES:
#      task.ns_dr_enabled=true   task.ns_dr_dist=beta
#      algorithm.dr_low / dr_high equal to task.ns_dr_low / ns_dr_high
#
#  See `baselines/docs/doraemon.md`.

from __future__ import annotations

from dataclasses import dataclass, MISSING
from typing import List, Optional, Tuple, Type

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms import _compat
from benchmarl.algorithms._baseline_math import (
    beta_entropy,
    beta_kl,
    beta_log_pdf,
    doraemon_success_rate,
    inv_sigmoid_bounds,
    sigmoid_bounds,
)
from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.mappo import Mappo, MappoConfig


class DoraemonState:
    """The live distribution, the episode buffer, and the diagnostics."""

    def __init__(self, a: float, b: float, low: float, high: float):
        self.a = float(a)
        self.b = float(b)
        self.low = float(low)
        self.high = float(high)
        self.iteration = 0
        self.returns: List[float] = []
        self.sigmas: List[float] = []
        self.partial: Optional[torch.Tensor] = None
        self.started = False          # DORAEMON's `train_until_done`
        self.n_skipped = 0
        self.last_success = 0.0
        self.last_entropy = float(beta_entropy(a, b, low, high))
        self.last_kl_target = 0.0
        self.last_kl_step = 0.0
        self.last_solver_ok = 1.0
        self.last_median_return = 0.0

    def reset_buffer(self) -> None:
        """``self.training_subrtn.reset_buffer()`` at the end of every step."""
        self.returns = []
        self.sigmas = []


class DoraemonLoss(ClipPPOLoss):
    """MAPPO's loss, plus the outer loop's diagnostics.

    DORAEMON does not change the RL algorithm at all -- it is a wrapper that
    decides what environment the RL algorithm trains in.  That is the point of
    the row: the doraemon-vs-dr_sigma comparison prices the CURRICULUM, with
    everything else held fixed.
    """

    #  Redeclared so torchrl's convert_to_functional does not warn: it checks
    #  the SUBCLASS's own __annotations__.
    actor_network: TensorDictModule
    critic_network: TensorDictModule

    def __init__(self, *args, state: DoraemonState, **kwargs):
        super().__init__(*args, **kwargs)
        self.doraemon_state = state

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        td_out = super().forward(tensordict)
        state = self.doraemon_state
        device = td_out.device

        def _log(name, value):
            td_out.set(
                name,
                torch.as_tensor(float(value), device=device, dtype=torch.float32),
            )

        _log("doraemon_iter", state.iteration)
        _log("doraemon_a", state.a)
        _log("doraemon_b", state.b)
        _log("doraemon_entropy", state.last_entropy)
        _log("doraemon_kl_target", state.last_kl_target)
        _log("doraemon_kl_step", state.last_kl_step)
        _log("doraemon_success", state.last_success)
        _log("doraemon_episodes", len(state.returns))
        _log("doraemon_solver_ok", state.last_solver_ok)
        _log("doraemon_median_return", state.last_median_return)
        _log("doraemon_skipped", state.n_skipped)
        return td_out


_DORAEMON_CALLBACK_CLS = None


def _doraemon_callback(algorithm: "Doraemon"):
    """Collect the episodes, run the outer step at the iteration boundary."""
    global _DORAEMON_CALLBACK_CLS
    if _DORAEMON_CALLBACK_CLS is None:

        class _DoraemonCallback(_compat.callback_base()):
            def __init__(self, algorithm):
                super().__init__()
                self._algorithm = algorithm

            def on_batch_collected(self, batch: TensorDictBase):
                self._algorithm.collect_episodes(batch)
                self._algorithm.maybe_step()

        _DORAEMON_CALLBACK_CLS = _DoraemonCallback
    return _DORAEMON_CALLBACK_CLS(algorithm)


class Doraemon(Mappo):
    """DORAEMON around BenchMARL's MAPPO, over the severity ``sigma``.

    Args:
        dr_low, dr_high (float): the support. MUST equal ``task.ns_dr_low`` /
            ``task.ns_dr_high``; the environment raises otherwise.
        init_a, init_b (float): the starting Beta. The reference's
            ``init_beta_param`` is ``100``, a distribution tightly concentrated
            on the middle of the support -- entropy maximisation has to start
            somewhere narrow or there is nothing to maximise.
        target_a, target_b (float): the maximum-entropy target. ``1``/``1`` is
            the uniform on the support, which is what DORAEMON's
            ``target_distr`` is.
        success_return (float): the return at or above which an episode counts
            as solved -- the reference's ``performance_lower_bound`` when a
            ``success_rate_condition`` is given.
        success_rate (float): ``alpha``, the required expected success rate.
            The paper's own experiments use ``0.5`` ("succRate50").
        kl_upper_bound (float): ``epsilon``, the trust region between
            consecutive distributions. Swept in the paper over
            ``{0.1, 0.05, 0.01, 0.005, 0.001}``.
        n_iters (int): how many outer iterations the frame budget is split into.
        train_until_lb (bool): do not touch the distribution until the
            performance constraint holds for the first time. The reference's
            default is on.
        hard_constraint (bool): the performance constraint may not be violated;
            the update is skipped if the current distribution already fails it.
            The reference's default is on.
        min_bound, max_bound (float): the sigmoid bounds DORAEMON optimises the
            Beta parameters inside.
        max_episodes (int): cap on the episodes kept per iteration for the
            importance-sampled constraint.

    All other arguments are :class:`~benchmarl.algorithms.Mappo`'s.
    """

    def __init__(
        self,
        dr_low: float,
        dr_high: float,
        init_a: float,
        init_b: float,
        target_a: float,
        target_b: float,
        success_return: float,
        success_rate: float,
        kl_upper_bound: float,
        n_iters: int,
        train_until_lb: bool,
        hard_constraint: bool,
        min_bound: float,
        max_bound: float,
        max_episodes: int,
        **kwargs,
    ):
        self.dr_low = float(dr_low)
        self.dr_high = float(dr_high)
        self.init_a = float(init_a)
        self.init_b = float(init_b)
        self.target_a = float(target_a)
        self.target_b = float(target_b)
        self.success_return = float(success_return)
        self.success_rate = float(success_rate)
        self.kl_upper_bound = float(kl_upper_bound)
        self.n_iters = int(n_iters)
        self.train_until_lb = bool(train_until_lb)
        self.hard_constraint = bool(hard_constraint)
        self.min_bound = float(min_bound)
        self.max_bound = float(max_bound)
        self.max_episodes = int(max_episodes)
        super().__init__(**kwargs)

        if not (0.0 <= self.success_rate <= 1.0):
            raise ValueError(
                f"success_rate is a probability; got {self.success_rate}"
            )
        if self.n_iters < 2:
            raise ValueError(
                f"n_iters={self.n_iters}: DORAEMON needs at least two rounds "
                "to widen anything."
            )
        try:
            import scipy.optimize  # noqa: F401
        except ImportError as err:  # pragma: no cover - environment dependent
            raise ImportError(
                "DORAEMON solves a constrained optimisation problem with "
                "scipy's trust-constr, which is what the reference uses. "
                "scipy is not importable in this environment: install it, or "
                "run the fixed-range `dr_sigma` row instead (BASELINES.md B9). "
                f"({err})"
            ) from err

        self._state = DoraemonState(
            self.init_a, self.init_b, self.dr_low, self.dr_high
        )
        self._publish()
        _compat.attach_callback(self.experiment, _doraemon_callback(self))

    # ------------------------------------------------------------------
    #  the environment side
    # ------------------------------------------------------------------

    def _publish(self) -> None:
        """Hand the environment the distribution it should draw from.

        Imported here rather than at module scope: an algorithm module that
        imports from the task package at import time is the same cycle
        ``_compat.callback_base`` exists to avoid, and this algorithm should
        also be importable without ``simple_ns`` on the path.
        """
        from simple_ns.dr_state import BetaDr, publish

        state = self._state
        publish(BetaDr(a=state.a, b=state.b, low=state.low, high=state.high))

    def _iter_frames(self) -> int:
        total = int(self.experiment_config.get_max_n_frames(self.on_policy))
        return max(total // self.n_iters, 1)

    # ------------------------------------------------------------------
    #  the episode buffer: (dynamics, return) pairs, as the reference calls it
    # ------------------------------------------------------------------

    @staticmethod
    def _first_present(batch: TensorDictBase, candidates):
        keys = batch.keys(True, True)
        for key in candidates:
            if key in keys:
                return batch.get(key)
        return None

    def collect_episodes(self, batch: TensorDictBase) -> None:
        """``get_buffer()``: the severity each episode ran at, and its return.

        The reference reads this off a per-environment wrapper that records one
        ``{dynamics, return}`` row per episode.  Here the same two numbers come
        out of the collected batch: the reward stream, the done flags, and
        ``info["ns_sigma"]`` -- which the environment has published since B9
        precisely so that "what severity was this trajectory at" is answerable
        from the data rather than from the config.
        """
        group = next(iter(self.group_map.keys()))
        reward = self._first_present(
            batch, [("next", group, "reward"), ("next", "reward")]
        )
        done = self._first_present(
            batch, [("next", group, "done"), ("next", "done")]
        )
        sigma = self._first_present(batch, [("next", group, "info", "ns_sigma")])
        if reward is None or done is None or sigma is None:
            raise RuntimeError(
                "DORAEMON needs the per-episode return AND the severity each "
                "episode was drawn at. One of reward / done / "
                "info['ns_sigma'] is missing from the collected batch: this "
                "row only runs on a simple_ns task with "
                "task.ns_dr_enabled=true."
            )
        n_envs, horizon = batch.batch_size[0], batch.batch_size[1]
        #  Team reward, one number per (env, step): every simple_ns host gives
        #  every agent the same reward, and DORAEMON's "return" is the
        #  episode's, not an agent's.
        rewards = reward.reshape(n_envs, horizon, -1).mean(-1)
        dones = done.reshape(n_envs, horizon, -1).any(-1)
        sigmas = sigma.reshape(n_envs, horizon, -1).mean(-1)

        state = self._state
        if state.partial is None or state.partial.shape[0] != n_envs:
            state.partial = torch.zeros(n_envs, device=rewards.device)
        for t in range(horizon):
            state.partial = state.partial + rewards[:, t]
            finished = dones[:, t]
            if bool(finished.any()):
                idx = finished.nonzero(as_tuple=True)[0]
                state.returns.extend(state.partial[idx].tolist())
                #  sigma is constant WITHIN an episode (it is redrawn in
                #  reset_world_at and nowhere else), so the value at the final
                #  step is the value the whole episode ran at.
                state.sigmas.extend(sigmas[idx, t].tolist())
                state.partial = state.partial.clone()
                state.partial[idx] = 0.0
        if len(state.returns) > self.max_episodes:
            state.returns = state.returns[-self.max_episodes :]
            state.sigmas = state.sigmas[-self.max_episodes :]

    # ------------------------------------------------------------------
    #  the outer loop
    # ------------------------------------------------------------------

    def maybe_step(self) -> None:
        iteration = min(
            self.experiment.total_frames // self._iter_frames(), self.n_iters - 1
        )
        if iteration == self._state.iteration:
            return
        self._state.iteration = int(iteration)
        self.doraemon_step()
        self._state.reset_buffer()

    def doraemon_step(self) -> None:
        state = self._state
        if len(state.returns) < 8:
            print(
                f"[doraemon] iteration {state.iteration}: only "
                f"{len(state.returns)} finished episodes in this round -- "
                "not enough to estimate the success rate; the distribution is "
                "left where it is."
            )
            state.n_skipped += 1
            return

        returns = torch.tensor(state.returns, dtype=torch.float64)
        sigmas = torch.tensor(state.sigmas, dtype=torch.float64)
        state.last_median_return = float(returns.median())
        current = (state.a, state.b)
        success_now = float(
            doraemon_success_rate(returns, self.success_return)
        )
        state.last_success = success_now

        #  `train_until_performance_lb`: the reference skips the update
        #  entirely until the constraint holds once.  Widening a distribution
        #  the policy cannot yet handle is how DR goes conservative, which is
        #  the failure the paper exists to avoid.
        if self.train_until_lb and not state.started:
            if success_now < self.success_rate:
                state.n_skipped += 1
                print(
                    f"[doraemon] iteration {state.iteration} SKIPPED: success "
                    f"rate {success_now:.3f} < {self.success_rate} and the "
                    "distribution has not started widening yet "
                    f"(median return {state.last_median_return:.3f}, "
                    f"threshold {self.success_return})"
                )
                return
            state.started = True

        #  `hard_performance_constraint`: the reference solves an inverted
        #  problem to find a feasible start inside the trust region.  Here the
        #  update is skipped instead, and the reason is printed -- see
        #  baselines/docs/doraemon.md, "NOT met".
        if self.hard_constraint and success_now < self.success_rate:
            state.n_skipped += 1
            print(
                f"[doraemon] iteration {state.iteration} SKIPPED: the CURRENT "
                f"distribution already violates the performance constraint "
                f"({success_now:.3f} < {self.success_rate}); with "
                "hard_constraint=true the distribution is held where it is "
                "until the policy catches up."
            )
            return

        new_a, new_b, ok = self._solve(current, sigmas, returns)
        state.last_solver_ok = 1.0 if ok else 0.0
        state.last_kl_step = float(beta_kl(state.a, state.b, new_a, new_b))
        state.a, state.b = new_a, new_b
        state.last_entropy = float(
            beta_entropy(state.a, state.b, state.low, state.high)
        )
        state.last_kl_target = float(
            beta_kl(state.a, state.b, self.target_a, self.target_b)
        )
        self._publish()
        print(
            f"[doraemon] iteration {state.iteration} at "
            f"{self.experiment.total_frames} frames: success "
            f"{success_now:.3f} over {len(state.returns)} episodes (median "
            f"return {state.last_median_return:.3f}); Beta({state.a:.4g}, "
            f"{state.b:.4g}) on [{state.low}, {state.high}]; entropy "
            f"{state.last_entropy:.4f} (max "
            f"{float(beta_entropy(self.target_a, self.target_b, state.low, state.high)):.4f}); "
            f"KL step {state.last_kl_step:.5f} (bound {self.kl_upper_bound}); "
            f"KL to target {state.last_kl_target:.5f}; solver "
            f"{'ok' if ok else 'FAILED, parameters kept'}"
        )

    # ------------------------------------------------------------------
    #  the constrained solve, in the reference's own formulation
    # ------------------------------------------------------------------

    def _solve(self, current, sigmas, returns) -> Tuple[float, float, bool]:
        """``min KL(phi || phi_target)  s.t.  J >= alpha,  KL(phi_i || phi) <= eps``.

        trust-constr with ANALYTIC jacobians on the objective and on both
        constraints, which is the reference's construction
        (``objective_fn``, ``kl_constraint_fn_prime``,
        ``performance_constraint_fn_prime``) and not an optimisation of it.
        Finite differences are not an option here: the trust-region KL between
        two nearby Betas with shape parameters of order 100 is ~1e-9, so a
        difference quotient of it is noise.
        """
        import numpy as np
        from scipy.optimize import minimize, NonlinearConstraint

        low, high = self.dr_low, self.dr_high
        lo_b, hi_b = self.min_bound, self.max_bound
        #  The indicator is a CONSTANT in the optimisation, exactly as the
        #  reference has it: `perf_values = (values.detach() >= condition)`.
        #  Only the importance weight depends on the candidate distribution.
        solved = (returns >= self.success_return).to(torch.float64)

        def _ab(x_opt, grad: bool = False):
            x = torch.as_tensor(np.asarray(x_opt), dtype=torch.float64)
            if grad:
                x = x.clone().requires_grad_(True)
            return x, sigmoid_bounds(x, lo_b, hi_b)

        def _with_grad(fn):
            """``(value, gradient)`` of a scalar torch function of ``x_opt``."""

            def wrapped(x_opt):
                x, ab = _ab(x_opt, grad=True)
                value = fn(ab)
                (grad,) = torch.autograd.grad(value, x)
                return float(value.detach()), grad.detach().numpy()

            return wrapped

        def _kl_target(ab):
            return beta_kl(ab[0], ab[1], self.target_a, self.target_b)

        def _kl_step(ab):
            return beta_kl(current[0], current[1], ab[0], ab[1])

        def _performance(ab):
            log_new = beta_log_pdf(sigmas, ab[0], ab[1], low, high)
            log_old = beta_log_pdf(sigmas, current[0], current[1], low, high)
            return (log_new - log_old).exp().mul(solved).mean()

        objective = _with_grad(_kl_target)
        kl_step_pair = _with_grad(_kl_step)
        performance_pair = _with_grad(_performance)

        def kl_step(x_opt):
            return kl_step_pair(x_opt)[0]

        def kl_step_jac(x_opt):
            return kl_step_pair(x_opt)[1]

        def performance(x_opt):
            return performance_pair(x_opt)[0]

        def performance_jac(x_opt):
            return performance_pair(x_opt)[1]

        constraints = [
            NonlinearConstraint(
                fun=kl_step,
                lb=-np.inf,
                ub=self.kl_upper_bound,
                jac=kl_step_jac,
                keep_feasible=False,
            ),
            NonlinearConstraint(
                #  1e-4 of slack, as the reference has: the solver complains
                #  when the start point sits exactly on a boundary.
                fun=performance,
                lb=self.success_rate - 1e-4,
                ub=np.inf,
                jac=performance_jac,
                keep_feasible=self.hard_constraint,
            ),
        ]

        x0 = (
            inv_sigmoid_bounds(
                torch.tensor([current[0], current[1]], dtype=torch.float64),
                lo_b,
                hi_b,
            )
            .numpy()
            .astype(float)
        )
        try:
            result = minimize(
                objective,
                x0,
                method="trust-constr",
                jac=True,
                constraints=constraints,
                options={"gtol": 1e-8, "xtol": 1e-10, "maxiter": 1000},
            )
        except Exception as err:  # pragma: no cover - solver dependent
            print(f"[doraemon] solver raised {type(err).__name__}: {err}")
            return current[0], current[1], False

        candidate = result.x
        ok = bool(result.success)
        feasible = (
            kl_step(candidate) <= self.kl_upper_bound + 1e-6
            and performance(candidate) >= self.success_rate - 1e-3
        )
        if not (ok and feasible):
            #  The reference keeps the old parameters unless the result BOTH
            #  satisfies every constraint and improves the objective.
            if not (feasible and result.fun < objective(x0)[0]):
                print(
                    f"[doraemon] optimisation not usable (success={ok}, "
                    f"feasible={feasible}); keeping the current distribution."
                )
                return current[0], current[1], False
        _, ab = _ab(candidate)
        return float(ab[0]), float(ab[1]), ok

    # ------------------------------------------------------------------

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        state = self._state
        print(
            f"DORAEMON {group}: sigma ~ Beta(a, b) on [{self.dr_low}, "
            f"{self.dr_high}], starting at Beta({self.init_a:g}, "
            f"{self.init_b:g}) (entropy "
            f"{float(beta_entropy(self.init_a, self.init_b, self.dr_low, self.dr_high)):.4f}) "
            f"and driven toward Beta({self.target_a:g}, {self.target_b:g}) "
            f"(entropy "
            f"{float(beta_entropy(self.target_a, self.target_b, self.dr_low, self.dr_high)):.4f}); "
            f"{self.n_iters} outer iterations of ~{self._iter_frames()} frames; "
            f"constraint E[return >= {self.success_return}] >= "
            f"{self.success_rate}; trust region KL <= {self.kl_upper_bound}; "
            f"train_until_lb={self.train_until_lb} "
            f"hard_constraint={self.hard_constraint}"
        )
        print(
            "[doraemon] SET success_return FROM YOUR OWN B0 ROW. It is the "
            "return at which an episode counts as solved, and it is the only "
            "host-dependent number here. Watch doraemon_success: pinned at 0 "
            "means the threshold is unreachable and the distribution will "
            "never widen; pinned at 1 means it is trivial and DORAEMON "
            "degenerates into the uniform in one step."
        )
        loss_module = DoraemonLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            **_compat.coefficient_kwargs(
                DoraemonLoss,
                entropy_coef=self.entropy_coef,
                critic_coef=self.critic_coef,
            ),
            loss_critic_type=self.loss_critic_type,
            normalize_advantage=False,
            state=state,
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
        return loss_module, False


@dataclass
class DoraemonConfig(MappoConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Doraemon`."""

    dr_low: float = MISSING
    dr_high: float = MISSING
    init_a: float = MISSING
    init_b: float = MISSING
    target_a: float = MISSING
    target_b: float = MISSING
    success_return: float = MISSING
    success_rate: float = MISSING
    kl_upper_bound: float = MISSING
    n_iters: int = MISSING
    train_until_lb: bool = MISSING
    hard_constraint: bool = MISSING
    min_bound: float = MISSING
    max_bound: float = MISSING
    max_episodes: int = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Doraemon
