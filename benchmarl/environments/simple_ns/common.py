#  BenchMARL wiring for the simple VMAS hosts under interaction-mediated,
#  INVERTIBLE non-stationarity.
#
#      python simple_ns/run.py algorithm=ippo task=simple_ns/transport
#
#  P-10.1: every arm is launched through the same entry point, with severity
#  supplied from OUTSIDE the method.  The baselines are the ones their authors
#  shipped; the only new object is the environment they run in.

from __future__ import annotations

import copy
import csv
import os
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from tensordict import TensorDictBase
from torchrl.envs import EnvBase
from torchrl.envs.libs.vmas import VmasEnv

from benchmarl.environments.common import Task
from benchmarl.environments.vmas.common import VmasClass
from benchmarl.utils import DEVICE_TYPING

__all__ = ["SimpleNsClass", "SimpleNsTask", "InertLayerError"]


class InertLayerError(RuntimeError):
    """NS-3.3: the layer is configured live but reached nothing.

    A silently inert disturbance is the one failure mode indistinguishable from
    a clean null result -- in exactly the arm you most need to trust.
    """


#: task name -> stock vmas scenario it wraps.  See simple_ns/hosts.py.
HOST_OF_TASK = {
    "transport": "transport",
    "sampling": "sampling",
    "balance": "balance",
    "navigation": "navigation",
}


class SimpleNsClass(VmasClass):
    """A stock VMAS scenario + the exertion dial (+ PACT), as a BenchMARL task.

    The dial and the method are the same objects for every host -- the layer needs
    only positions and a force action -- so a difference between two hosts is a
    difference in the task, not in the disturbance or in the method.
    """

    @property
    def host(self) -> str:
        name = self.name.lower()
        if name not in HOST_OF_TASK:
            raise ValueError(
                f"no host registered for simple_ns task {name!r}; "
                f"known: {sorted(HOST_OF_TASK)}"
            )
        return HOST_OF_TASK[name]

    def _scenario_kwargs(self) -> Dict[str, Any]:
        config = copy.deepcopy(self.config)
        for required in ("ns_severity", "pact_enabled"):
            if required not in config:
                raise ValueError(
                    f"task config for {self.name} is missing {required!r}; a "
                    "partially overridden config would silently run a different "
                    "environment"
                )
        return config

    @property
    def pact_enabled(self) -> bool:
        return bool(self.config.get("pact_enabled", False))

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        from simple_ns.hosts import make_scenario

        config = self._scenario_kwargs()
        pact = self.pact_enabled
        host = self.host
        return lambda: VmasEnv(
            # a scenario INSTANCE: nothing is copied into the installed vmas, so
            # a vmas upgrade cannot silently revert the non-stationarity, and the
            # stock scenario and this one coexist in one process
            scenario=make_scenario(pact, host=host),
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            categorical_actions=True,
            clamp_actions=True,
            **config,
        )

    def supports_discrete_actions(self) -> bool:
        """Yes, with one caveat that belongs in the caption rather than here.

        The disturbance is an additive force applied in ``process_action``, below
        the action interface, so discretising the policy's choice does not
        discretise the channel.  What discretisation DOES change is the task
        interface, so absolute returns are not comparable between a discrete and
        a continuous algorithm: compare blind against pact WITHIN an algorithm.

        Note also that the compensation is a continuous correction subtracted
        below the interface, so a discrete-action arm still receives a continuous
        correction.  That is the honest reading of "the actuator is compensated,
        the policy is not", and it is why the discrete arms are a robustness
        check rather than the headline.
        """
        return True

    @staticmethod
    def env_name() -> str:
        return "simple_ns"

    # ------------------------------------------------------------------
    #  II.10 -- the instrument panel
    # ------------------------------------------------------------------

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for group in self._groups_with_info(batch):
            info = batch.get(("next", group, "info"))
            out.update(self._ns_diagnostics(info))
            out.update(self._pact_diagnostics(info))
        # the domain metric belongs in the same row as the diagnostics, or you
        # cannot tell whether a good return came with a good estimate
        try:
            rew = batch.get(("next", "agents", "reward"))
            out["domain/reward_mean"] = float(rew.to(torch.float32).mean())
        except Exception:  # noqa: BLE001 -- group naming varies by task
            pass
        self._write_debug_row(out)
        return out

    # ------------------------------------------------------------------
    #  the debug CSV
    # ------------------------------------------------------------------

    def _write_debug_row(self, row: Dict[str, float]) -> None:
        """Append one row per collection iteration to a standalone CSV.

        BenchMARL's own csv logger writes one file per scalar, which is fine for
        plotting a curve and useless for asking "on the iteration where the
        return moved, what was the estimator doing".  Everything needed to judge
        whether PACT is working -- and, if it is not, WHICH link failed -- belongs
        in one row.

        Read it in this order, because the first question that answers "no" is
        the one to fix:

          ns/live_frac          did the disturbance happen at all?  0 means the
                                dial never fired and nothing below matters.
          ns/load_mean          how big was it, in action-range units?
          pact/fit_gain         does the REDUCTION hold?  Scored against an
                                intercept-only null, so this is specifically
                                whether the PEER CHANNELS explained anything.
                                Near 0 with a live disturbance means the model
                                class is wrong here and no amount of training
                                helps (II.9 gate 6).
          pact/beta_cos         is beta being RECOVERED?  Cosine against the true
                                beta*, which is known exactly on this instance.
                                High fit_gain with low beta_cos means it predicts
                                without identifying -- the design matrix is
                                degenerate and you may use beta but not decompose
                                it (II.9 gate 7).
          pact/n_updates        is the estimator being fed?  Flat means the rows
                                are being skipped as dead (P-4.2).
          pact/n_diverged       did the covariance blow up?  Non-zero means mu is
                                too aggressive for the excitation in this run.
          pact/confidence       is the gate open?
          pact/trust_applied    ... and does applied trust track the policy's
          pact/trust_policy     trust?  Divergence here is the P-5.2 failure.
          pact/cancelled_frac   how much of the disturbance was actually removed.
          pact/corr_vs_d        correction magnitude over disturbance magnitude.
                                Should approach 1.  Much above 1 is over-
                                correction; near 0 means trust or the estimate is
                                dead.
          ns/clipped_frac       is the actuator saturating?  Past ~30% the row is
                                about the action box, not the method.
          domain/reward_mean    and only then, did it win.

        The path comes from SIMPLE_NS_DEBUG_CSV, which simple_ns/run.py sets from
        experiment.save_folder so each arm gets its own file.
        """
        path = os.environ.get("SIMPLE_NS_DEBUG_CSV")
        if not path:
            #  Say so ONCE.  A diagnostic that silently does not exist is worse
            #  than no diagnostic: the first run of this looked like the writer
            #  was broken when the variable simply was not set, because the run
            #  predated simple_ns/run.py setting it.
            if not getattr(self, "_debug_warned", False):
                self._debug_warned = True
                print(
                    "[simple_ns] SIMPLE_NS_DEBUG_CSV is unset, so no debug CSV "
                    "will be written. simple_ns/run.py sets it from "
                    "experiment.save_folder -- if you are not launching through "
                    "it, export it yourself. The same columns are still in the "
                    "experiment's own logger under ns/* and pact/*.",
                    flush=True,
                )
            return
        if not row:
            return
        try:
            f = Path(path)
            f.parent.mkdir(parents=True, exist_ok=True)
            self._debug_n = getattr(self, "_debug_n", 0) + 1
            record = {
                "iteration": self._debug_n,
                "wall_time": round(time.time() - getattr(self, "_debug_t0", time.time()), 2),
                "arm": "pact" if self.pact_enabled else "blind",
                "sigma": self.config.get("ns_severity"),
                "direct": self.config.get("ns_direct"),
                "channels": self.config.get("pact_channels"),
                "oracle": self.config.get("pact_oracle"),
                "mu": self.config.get("pact_mu"),
                **{k.replace("/", "_"): v for k, v in sorted(row.items())},
            }
            if not hasattr(self, "_debug_t0"):
                self._debug_t0 = time.time()
            new = not f.exists()
            with f.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(record))
                if new:
                    w.writeheader()
                w.writerow(record)
        except Exception as exc:  # noqa: BLE001 -- diagnostics must never kill a run
            #  Never fatal, but never silent either.  Reported once, with the
            #  reason, so a missing file is a message rather than a mystery.
            if not getattr(self, "_debug_failed", False):
                self._debug_failed = True
                print(
                    f"[simple_ns] debug CSV disabled: {type(exc).__name__}: {exc} "
                    f"(path={path!r}). The run continues; ns/* and pact/* are "
                    "still in the experiment's own logger.",
                    flush=True,
                )

    @staticmethod
    def _groups_with_info(batch: TensorDictBase) -> List[str]:
        keys = batch.get("next").keys(include_nested=True, leaves_only=False)
        return [
            k
            for k in keys
            if isinstance(k, str)
            and ("next", k, "info") in batch.keys(include_nested=True)
        ]

    @staticmethod
    def _get(info: TensorDictBase, key: str) -> Optional[torch.Tensor]:
        if key not in info.keys():
            return None
        v = info.get(key).to(torch.float32)
        return v.squeeze(-1) if v.shape[-1] == 1 else v

    def _ns_diagnostics(self, info: TensorDictBase) -> Dict[str, float]:
        out: Dict[str, float] = {}
        load = self._get(info, "ns_load")
        if load is None:
            return out
        flat = load.abs().reshape(-1)
        out["ns/load_mean"] = float(flat.mean())
        out["ns/load_p95"] = float(flat.quantile(0.95))
        for key, name in (
            ("ns_dmag", "disturbance_mag"),
            ("ns_A", "driver_mean"),
            ("ns_y", "sensor_mean"),
            ("ns_x_std", "channel_std"),
        ):
            v = self._get(info, key)
            if v is not None:
                out[f"ns/{name}"] = float(v.reshape(-1).mean())
        clip = self._get(info, "ns_clipped")
        if clip is not None:
            # The actuator saturating is what bounds sigma*: past it, even a
            # controller handed the true disturbance cannot recover, and the row
            # is about the environment rather than the method.  Watch it.
            out["ns/clipped_frac"] = float(clip.reshape(-1).mean())
        out["ns/live_frac"] = float((flat > 0).to(torch.float32).mean())
        self._check_layer_fired(out["ns/live_frac"], info)
        return out

    def _check_layer_fired(self, live_frac: float, info: TensorDictBase) -> None:
        if float(self.config.get("ns_severity", 0.0)) <= 0.0:
            return
        if live_frac > 0.0:
            return
        driver = self._get(info, "ns_A")
        if driver is not None and float(driver.reshape(-1).max()) == 0.0:
            return  # legitimately an all-placebo batch
        raise InertLayerError(
            f"ns_severity={self.config.get('ns_severity')} but NOT ONE action "
            "was disturbed in this batch, with placebo ruled out. The layer is "
            "not reaching the physics -- a wiring bug, not a null result."
        )

    def _pact_diagnostics(self, info: TensorDictBase) -> Dict[str, float]:
        out: Dict[str, float] = {}
        trust = self._get(info, "pact_trust")
        if trust is None:
            return out
        # P-5.3: policy-set and applied trust are SEPARATE columns.  They diverge
        # precisely when the confidence gate is misbehaving.
        out["pact/trust_applied"] = float(trust.reshape(-1).mean())
        out["pact/trust_policy"] = float(self.config.get("pact_trust", 0.0))
        for key, name in (
            ("pact_conf", "confidence"),
            ("pact_pred", "pred_mean"),
            ("pact_corr", "correction_mag"),
            ("pact_updates", "n_updates"),
            ("pact_skipped", "n_skipped"),
            ("pact_bounded", "n_bounded"),
            ("pact_diverged", "n_diverged"),
            # II.10's two headline columns: does the reduction hold, and is beta
            # actually being recovered (the truth is known on this instance)
            ("pact_fit_gain", "fit_gain"),
            ("pact_beta_cos", "beta_cos"),
            ("pact_beta_relerr", "beta_relerr"),
            ("pact_corr_vs_d", "corr_vs_d"),
        ):
            v = self._get(info, key)
            if v is not None:
                out[f"pact/{name}"] = float(v.reshape(-1).mean())
        # the headline diagnostic for an INVERTIBLE channel: how much of the
        # disturbance the estimate actually removed
        resid = self._get(info, "pact_residual")
        load = self._get(info, "ns_load")
        if resid is not None and load is not None:
            num = float(resid.reshape(-1).mean())
            den = float(load.abs().reshape(-1).mean())
            out["pact/residual_mean"] = num
            if den > 1e-12:
                out["pact/cancelled_frac"] = 1.0 - num / den
        return out


class SimpleNsTask(Task):
    """Enum for the simple VMAS hosts under the exertion layer."""

    TRANSPORT = None
    SAMPLING = None
    BALANCE = None
    NAVIGATION = None

    @staticmethod
    def associated_class():
        return SimpleNsClass
