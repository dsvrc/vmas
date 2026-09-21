#  QCD+ / RR -- prior-free black-box non-stationary RL by DETECT-AND-RESTART.
#
#      Gerogiannis, Huang, Veeravalli, "Is Prior-Free Black-Box Non-Stationary
#      Reinforcement Learning Feasible?", arXiv 2410.13772.
#      Detector: Besson, Kaufmann, Maillard, Seznec, "Efficient Change-Point
#      Detection for Tackling Piecewise-Stationary Bandits", JMLR 2022 -- the
#      Bernoulli GLR that 2410.13772's Algorithm 3 plugs in, with the threshold
#      beta(n, delta) = log(4 n sqrt(n) / delta) that paper names.
#
#  The paper's argument, in three steps:
#
#    1. MASTER (Wei & Luo) is the state of the art in BLACK-BOX non-stationary
#       RL: no prior knowledge of when or how often the environment changes.
#    2. Theorem 4: MASTER's two non-stationarity tests compare a quantity
#       bounded by 1 against a threshold of order 54 (log2 T + 1) log(T/delta),
#       so they CANNOT fire below T ~ 1.24e9 rounds.  Below that horizon --
#       i.e. in every experiment anyone runs -- MASTER is restarting at random.
#    3. So the baseline worth running is the one that does fire: quickest
#       change detection plus a full restart.  In their 5-armed piecewise
#       stationary bandits, MASTER declares 0 changes on every problem and the
#       QCD methods declare 8-150.
#
#  What that gives this repo is a NON-STATIONARITY baseline that assumes
#  nothing about the disturbance: it does not observe the driver (unlike LCPO,
#  B6), it does not identify a latent (unlike RMA, B8) and it does not estimate
#  a coupling (unlike PACT).  It watches the reward stream, decides the world
#  changed, and throws the learner away.  If that wins, every method in this
#  paper that models the disturbance is priced against a method that models
#  nothing.
#
#  Three arms, one algorithm:
#
#      detector=glr      Algorithm 3 (QCD+): Bernoulli GLR on the reward
#                        stream, full restart on alarm.
#      detector=random   Algorithm 2 (RR): restart at i.i.d. Geometric times.
#                        The paper's own order-optimal baseline, and what
#                        Theorem 4 says MASTER degenerates into.
#      detector=none     no restarts.  Stock MAPPO, reached through this file,
#                        so the three rows differ by ONE flag.
#
#  See `baselines/docs/qcd.md`.

from __future__ import annotations

import collections
from dataclasses import dataclass, MISSING
from typing import Dict, Optional, Tuple, Type

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.objectives import ClipPPOLoss, LossModule, ValueEstimators

from benchmarl.algorithms import _compat
from benchmarl.algorithms._baseline_math import (
    GlrDetector,
    master_can_fire,
    master_min_horizon,
    RandomRestartSchedule,
)
from benchmarl.algorithms.common import Algorithm
from benchmarl.algorithms.mappo import Mappo, MappoConfig


class QcdState:
    """Everything one group's detector-and-restart wrapper remembers.

    Held outside the loss so the callback (which sees the collected batch) and
    the loss (which reports the diagnostics) are looking at the same object.
    """

    def __init__(
        self,
        detector: Optional[GlrDetector],
        schedule: Optional[RandomRestartSchedule],
        reward_low: float,
        reward_high: float,
    ):
        self.detector = detector
        self.schedule = schedule
        self.reward_low = float(reward_low)
        self.reward_high = float(reward_high)
        self.span = max(self.reward_high - self.reward_low, 1e-9)

        self.n_restarts = 0
        self.pending_restart = False
        self.samples_seen = 0
        self.steps_since_restart = 0
        self.n_clipped = 0
        self.last_stat = 0.0
        self.last_threshold = float("inf")

    def normalise(self, values: torch.Tensor) -> torch.Tensor:
        """The DECLARED affine map onto ``[0, 1]`` the GLR is defined on.

        The Bernoulli GLR tests a stream of ``[0, 1]``-valued observations, and
        a bandit's rewards already are.  A VMAS return is not, so it is mapped
        by a declared range and clipped, and the fraction that hit the clip is
        logged as ``qcd_clip_frac``.  A range that is wrong makes the detector
        blind (everything saturates) or deaf (everything crowds the middle) --
        so it is reported rather than assumed, in the same spirit as LCPO's
        ``ood_threshold``, which had to be rescaled off the paper's value for
        exactly this reason.
        """
        scaled = (values - self.reward_low) / self.span
        self.n_clipped += int(((scaled < 0.0) | (scaled > 1.0)).sum())
        return scaled.clamp(0.0, 1.0)

    def observe(self, values: torch.Tensor) -> bool:
        """Feed one batch of reward samples IN TIME ORDER.  True on an alarm."""
        fired = False
        if self.detector is not None:
            for value in self.normalise(values).tolist():
                if self.detector.observe(value):
                    fired = True
            self.last_stat = self.detector.last_stat
            self.last_threshold = self.detector.last_threshold
        if self.schedule is not None:
            for _ in range(values.numel()):
                if self.schedule.step():
                    fired = True
        self.samples_seen += int(values.numel())
        self.steps_since_restart += int(values.numel())
        if fired:
            self.pending_restart = True
        return fired

    def clip_frac(self) -> float:
        return self.n_clipped / max(self.samples_seen, 1)


class QcdLoss(ClipPPOLoss):
    """MAPPO's loss, plus the wrapper's diagnostics.

    Adds nothing to the objective: 2410.13772's algorithms are BLACK-BOX
    wrappers -- the base learner runs untouched between restarts, which is what
    makes the row a fair price for "assume nothing, restart when surprised".
    """

    #  Redeclared so torchrl's convert_to_functional does not warn: it checks
    #  the SUBCLASS's own __annotations__.  Same list torchrl's own losses
    #  carry.
    actor_network: TensorDictModule
    critic_network: TensorDictModule

    def __init__(self, *args, state: QcdState, **kwargs):
        super().__init__(*args, **kwargs)
        self.qcd_state = state

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        td_out = super().forward(tensordict)
        device = td_out.device
        state = self.qcd_state

        def _log(name, value):
            td_out.set(
                name,
                torch.as_tensor(float(value), device=device, dtype=torch.float32),
            )

        _log("qcd_restarts", state.n_restarts)
        _log("qcd_stat", state.last_stat)
        _log(
            "qcd_threshold",
            state.last_threshold if state.last_threshold < float("inf") else -1.0,
        )
        _log("qcd_steps_since_restart", state.steps_since_restart)
        _log("qcd_clip_frac", state.clip_frac())
        return td_out


_RESTART_CALLBACK_CLS = None


def _qcd_callback(algorithm: "Qcd"):
    """Watch the reward stream; restart the learner between iterations.

    Built here rather than at module level because its base class lives in
    ``benchmarl.experiment``, which imports back out of
    ``benchmarl.algorithms`` -- see ``_compat.callback_base``.
    """
    global _RESTART_CALLBACK_CLS
    if _RESTART_CALLBACK_CLS is None:

        class _QcdCallback(_compat.callback_base()):
            def __init__(self, algorithm):
                super().__init__()
                self._algorithm = algorithm

            def on_batch_collected(self, batch: TensorDictBase):
                self._algorithm.observe_batch(batch)

            def on_train_end(self, training_td: TensorDictBase, group: str):
                #  AFTER the batch has been trained on, not before.  The batch
                #  was collected by the pre-restart policy and carries its
                #  log-probabilities; training on it is one ordinary PPO
                #  update, whereas restarting first would leave the importance
                #  ratio comparing a fresh network against a trained one.  The
                #  next collection then runs on the restarted weights, which is
                #  what Algorithm 3 does.
                self._algorithm.apply_restart(group)

        _RESTART_CALLBACK_CLS = _QcdCallback
    return _RESTART_CALLBACK_CLS(algorithm)


class Qcd(Mappo):
    """Detect-and-restart around BenchMARL's MAPPO.

    Args:
        detector (str): ``"glr"`` is Algorithm 3 (QCD+), ``"random"`` is
            Algorithm 2 (RR), ``"none"`` disables restarts entirely and leaves
            stock MAPPO.
        delta (float): the GLR's false-alarm probability in
            ``beta(n, delta) = log(4 n sqrt(n) / delta)``. ``0`` means
            ``1/sqrt(T)`` with ``T`` the number of detector samples in the
            whole run, which is the usual prior-free choice and is what the
            paper's "order ``1/poly(T)``" allows.
        min_samples (int): the detector will not declare a change before this
            many post-restart samples. Two is the minimum a GLR is defined on.
        max_samples (int): cap on the post-restart history the statistic is
            computed over.
        reward_low, reward_high (float): the DECLARED range the per-step reward
            is mapped onto ``[0, 1]`` by. Watch ``qcd_clip_frac``.
        rr_interval (int): RR's mean restart interval in detector samples;
            ``eta_r = 1 / rr_interval``.
        restart_critic (bool): restart the value function as well as the
            policy. ``True`` is the black-box reading -- the base learner is
            restarted, and its critic is part of it.
        warmup_samples (int): detector samples to discard at the start of
            training. Nothing has been learned yet, so a restart there costs
            nothing and only spends the detector's history.

    All other arguments are :class:`~benchmarl.algorithms.Mappo`'s.
    """

    def __init__(
        self,
        detector: str,
        delta: float,
        min_samples: int,
        max_samples: int,
        reward_low: float,
        reward_high: float,
        rr_interval: int,
        restart_critic: bool,
        warmup_samples: int,
        **kwargs,
    ):
        self.detector_kind = str(detector)
        if self.detector_kind not in ("glr", "random", "none"):
            raise ValueError(
                "detector must be 'glr' (Algorithm 3, QCD+), 'random' "
                f"(Algorithm 2, RR) or 'none'; got {detector!r}"
            )
        self.delta = float(delta)
        self.min_samples = int(min_samples)
        self.max_samples = int(max_samples)
        self.reward_low = float(reward_low)
        self.reward_high = float(reward_high)
        self.rr_interval = int(rr_interval)
        self.restart_critic = bool(restart_critic)
        self.warmup_samples = int(warmup_samples)
        super().__init__(**kwargs)

        if self.has_rnn:
            raise NotImplementedError(
                "QCD+ here does not support recurrent models: a restart would "
                "have to reset the recurrent state carried in the buffer as "
                "well as the parameters, and 'what did the learner remember "
                "across a restart' would stop being answerable."
            )
        if self.reward_high <= self.reward_low:
            raise ValueError(
                f"reward_low={self.reward_low} reward_high={self.reward_high}: "
                "the GLR is defined on a [0, 1]-valued stream, so the declared "
                "range must be non-empty."
            )

        self._generator = torch.Generator().manual_seed(int(self.experiment.seed))
        self._states: Dict[str, QcdState] = {}
        self._snapshots: Dict[str, Dict] = {}
        self._losses: Dict[str, QcdLoss] = {}
        _compat.attach_callback(self.experiment, _qcd_callback(self))

    # ------------------------------------------------------------------
    #  the detector's sample budget, so delta = 1/sqrt(T) means something
    # ------------------------------------------------------------------

    def _n_detector_samples(self) -> int:
        """How many samples the detector will see over the whole run.

        One per environment STEP of the vectorised env: the batch is
        ``(n_envs, T)`` and the stream is the reward averaged over the parallel
        copies at each ``t``, so a run of ``F`` frames at ``E`` parallel
        workers gives ``F / E`` samples.
        """
        frames = int(self.experiment_config.get_max_n_frames(self.on_policy))
        envs = max(int(self.experiment_config.n_envs_per_worker(self.on_policy)), 1)
        return max(frames // envs, 2)

    def _delta(self) -> float:
        if self.delta > 0.0:
            return self.delta
        return 1.0 / (self._n_detector_samples() ** 0.5)

    # ------------------------------------------------------------------

    def _state(self, group: str) -> QcdState:
        if group not in self._states:
            delta = self._delta()
            detector = (
                GlrDetector(
                    delta=delta,
                    min_samples=self.min_samples,
                    max_samples=self.max_samples,
                )
                if self.detector_kind == "glr"
                else None
            )
            schedule = (
                RandomRestartSchedule(
                    eta=1.0 / max(self.rr_interval, 1), generator=self._generator
                )
                if self.detector_kind == "random"
                else None
            )
            self._states[group] = QcdState(
                detector=detector,
                schedule=schedule,
                reward_low=self.reward_low,
                reward_high=self.reward_high,
            )
        return self._states[group]

    def observe_batch(self, batch: TensorDictBase) -> None:
        """Feed each group's per-step reward stream to its detector."""
        for group in self.group_map.keys():
            state = self._state(group)
            reward = batch.get(("next", group, "reward"), None)
            if reward is None:
                continue
            #  batch.batch_size is (n_envs, T): average over the parallel
            #  copies, which are i.i.d. replicas of the same process, and over
            #  the agents, whose reward is the team's on every simple_ns host.
            #  What is left is R_t, one sample per round, in time order --
            #  which is the stream Algorithm 3 observes.
            per_step = reward.mean(dim=0).reshape(reward.shape[1], -1).mean(-1)
            if state.samples_seen < self.warmup_samples:
                remaining = self.warmup_samples - state.samples_seen
                state.samples_seen += int(min(remaining, per_step.numel()))
                state.steps_since_restart += int(min(remaining, per_step.numel()))
                if per_step.numel() <= remaining:
                    continue
                per_step = per_step[remaining:]
            if state.observe(per_step.detach().cpu()):
                print(
                    f"[qcd] {group}: CHANGE DECLARED at "
                    f"{self.experiment.total_frames} frames "
                    f"(statistic {state.last_stat:.3f} > threshold "
                    f"{state.last_threshold:.3f}); restarting the learner "
                    f"after this iteration"
                    if self.detector_kind == "glr"
                    else f"[qcd] {group}: RANDOM RESTART at "
                    f"{self.experiment.total_frames} frames"
                )

    # ------------------------------------------------------------------
    #  the restart itself
    # ------------------------------------------------------------------

    def _snapshot(self, group: str, loss: QcdLoss) -> None:
        """The learner's initial parameters, kept for the restart.

        ``H_B <- {}`` in Algorithms 2 and 3 means "forget everything this
        instance learned".  For a tabular bandit that is the empirical means;
        for a parametric learner it is the parameter vector it started from,
        plus the optimiser's accumulated moments, which are as much learned
        state as the weights are.
        """
        entry = {
            "actor": [
                (key, value.detach().clone())
                for key, value in loss.actor_network_params.items(True, True)
            ]
        }
        if self.restart_critic:
            entry["critic"] = [
                (key, value.detach().clone())
                for key, value in loss.critic_network_params.items(True, True)
            ]
        self._snapshots[group] = entry

    def apply_restart(self, group: str) -> None:
        state = self._states.get(group)
        if state is None or not state.pending_restart:
            return
        state.pending_restart = False
        loss = self._losses[group]
        snapshot = self._snapshots[group]

        with torch.no_grad():
            live = dict(loss.actor_network_params.items(True, True))
            for key, value in snapshot["actor"]:
                target = live.get(key)
                if target is not None and target.shape == value.shape:
                    target.data.copy_(value)
            if self.restart_critic:
                live = dict(loss.critic_network_params.items(True, True))
                for key, value in snapshot["critic"]:
                    target = live.get(key)
                    if target is not None and target.shape == value.shape:
                        target.data.copy_(value)

        #  Adam's first and second moments are learned state too: leaving them
        #  in place would push the fresh parameters along the OLD gradient for
        #  hundreds of steps, which is not a restart.
        for name, optimizer in self.experiment.optimizers[group].items():
            if name == "loss_critic" and not self.restart_critic:
                continue
            #  defaultdict(dict), not type(...)(): a plain defaultdict copy
            #  loses the factory and the next `optimizer.state[p]` raises.
            optimizer.state = collections.defaultdict(dict)

        state.n_restarts += 1
        state.steps_since_restart = 0
        print(
            f"[qcd] {group}: RESTART {state.n_restarts} applied at "
            f"{self.experiment.total_frames} frames "
            f"(policy{' + critic' if self.restart_critic else ''}, "
            "optimiser moments cleared)"
        )

    # ------------------------------------------------------------------

    def _get_loss(
        self, group: str, policy_for_loss: TensorDictModule, continuous: bool
    ) -> Tuple[LossModule, bool]:
        state = self._state(group)
        delta = self._delta()
        n_samples = self._n_detector_samples()
        print(
            f"QCD {group}: detector={self.detector_kind}"
            + (
                f", Bernoulli GLR with delta={delta:.3e} "
                f"(threshold log(4 n sqrt(n) / delta)), reward mapped from "
                f"[{self.reward_low}, {self.reward_high}] onto [0, 1]"
                if self.detector_kind == "glr"
                else (
                    f", Geometric(1/{self.rr_interval}) restart times"
                    if self.detector_kind == "random"
                    else ", NO restarts (stock MAPPO through this file)"
                )
            )
            + f"; ~{n_samples} detector samples over the run, restart resets "
            f"the policy{' and the critic' if self.restart_critic else ''} to "
            "its initial parameters and clears the optimiser moments"
        )
        #  2410.13772's Theorem 4, evaluated at THIS horizon.  It is the
        #  paper's headline and it costs two logarithms, so the run states it
        #  rather than leaving a reader to look it up.
        print(
            f"[qcd] MASTER's own non-stationarity tests could fire at "
            f"T={n_samples} rounds: {master_can_fire(n_samples)} "
            f"(Theorem 4: they need T >= {master_min_horizon():.3e}). "
            "That is why the arm that is run here is quickest change "
            "detection and not MASTER."
        )

        loss_module = QcdLoss(
            actor=policy_for_loss,
            critic=self.get_critic(group),
            clip_epsilon=self.clip_epsilon,
            **_compat.coefficient_kwargs(
                QcdLoss, entropy_coef=self.entropy_coef, critic_coef=self.critic_coef
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
        self._losses[group] = loss_module
        self._snapshot(group, loss_module)
        return loss_module, False


@dataclass
class QcdConfig(MappoConfig):
    """Configuration dataclass for :class:`~benchmarl.algorithms.Qcd`."""

    detector: str = MISSING
    delta: float = MISSING
    min_samples: int = MISSING
    max_samples: int = MISSING
    reward_low: float = MISSING
    reward_high: float = MISSING
    rr_interval: int = MISSING
    restart_critic: bool = MISSING
    warmup_samples: int = MISSING

    @staticmethod
    def associated_class() -> Type[Algorithm]:
        return Qcd
