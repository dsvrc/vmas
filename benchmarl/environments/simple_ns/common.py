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
        return out

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
