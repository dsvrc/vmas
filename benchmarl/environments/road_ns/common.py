#  BenchMARL wiring for road_traffic under Coupling-Under-Drift.
#
#      python road_ns/run.py algorithm=ippo task=road_ns/road_traffic
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

__all__ = ["RoadNsClass", "RoadNsTask", "InertLayerError"]


class InertLayerError(RuntimeError):
    """NS-3.3: the severity layer is configured live but reached nothing.

    A silently inert disturbance is the one failure mode indistinguishable from
    a clean null result -- in exactly the arm you most need to trust.
    """


class RoadNsClass(VmasClass):
    """``vmas/road_traffic`` + the dial (+ PACT), as a BenchMARL task."""

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
        from road_ns.scenario import make_scenario

        config = self._scenario_kwargs()
        pact = self.pact_enabled
        return lambda: VmasEnv(
            # a scenario INSTANCE: nothing is copied into the installed vmas, so
            # a vmas upgrade cannot silently revert the non-stationarity
            scenario=make_scenario(pact),
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            categorical_actions=True,
            clamp_actions=True,
            **config,
        )

    def supports_discrete_actions(self) -> bool:
        """Yes -- qmix, vdn and iql are ordinary arms here.

        A.7's warning is about a *discrete channel*, i.e. a harm that is a
        permutation.  This harm is not: it is a continuous multiplicative factor
        on the velocity command, applied in ``process_action`` **below the
        action interface**.  Discretising the policy's choice does not
        discretise the channel, so the method is unchanged.

        Nor is there a headroom problem.  The pace shift is multiplicative and
        sits near 1, so ``v * shift`` is representable whatever grid ``v`` came
        off -- unlike a divisive inverse, which would leave the action box on
        any non-zero command and turn the correction into a constant bias.

        What discretisation DOES change is the task interface, so absolute
        returns are not comparable between a discrete and a continuous
        algorithm.  Compare blind against pact WITHIN an algorithm.
        """
        return True

    @staticmethod
    def env_name() -> str:
        return "road_ns"

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
        u = self._get(info, "ns_u")
        if u is None:
            return out
        flat = u.reshape(-1)
        out["ns/u_mean"] = float(flat.mean())
        out["ns/u_p95"] = float(flat.quantile(0.95))
        for key, name in (
            ("ns_harm", "harm_mean"),
            ("ns_g", "g_mean"),
            ("ns_A", "driver_mean"),
            ("ns_excess", "excess_mean"),
        ):
            v = self._get(info, key)
            if v is not None:
                out[f"ns/{name}"] = float(v.reshape(-1).mean())

        # Is the dial live?  harm == 1.0 exactly means it reached nothing.
        h = self._get(info, "ns_harm")
        if h is not None:
            hf = h.reshape(-1)
            out["ns/harmed_frac"] = float((hf > 1.0).to(torch.float32).mean())
            out["ns/dry_frac"] = float((hf == 1.0).to(torch.float32).mean())
            self._check_layer_fired(out["ns/harmed_frac"], info)
        return out

    def _check_layer_fired(self, harmed_frac: float, info: TensorDictBase) -> None:
        if float(self.config.get("ns_severity", 0.0)) <= 0.0:
            return
        if harmed_frac > 0.0:
            return
        driver = self._get(info, "ns_A")
        if driver is not None and float(driver.reshape(-1).max()) == 0.0:
            return  # legitimately an all-placebo batch
        raise InertLayerError(
            f"ns_severity={self.config.get('ns_severity')} but NOT ONE record "
            "was harmed in this batch, with placebo ruled out. The layer is not "
            "reaching the physics -- a wiring bug, not a null result."
        )

    def _pact_diagnostics(self, info: TensorDictBase) -> Dict[str, float]:
        out: Dict[str, float] = {}
        trust = self._get(info, "pact_trust")
        if trust is None:
            return out
        # P-5.3: policy-set and applied trust are SEPARATE columns.  They
        # diverge precisely when the confidence gate is misbehaving.
        out["pact/trust_applied"] = float(trust.reshape(-1).mean())
        out["pact/trust_policy"] = float(self.config.get("pact_trust", 0.0))
        for key, name in (
            ("pact_conf", "confidence"),
            ("pact_pred", "pred_mean"),
            ("pact_updates", "n_updates"),
            ("pact_skipped", "n_skipped"),
            ("pact_herd", "herd_index"),
        ):
            v = self._get(info, key)
            if v is not None:
                out[f"pact/{name}"] = float(v.reshape(-1).mean())
        shift = self._get(info, "pact_shift")
        if shift is not None:
            s = shift.reshape(-1)
            out["pact/shift_abs"] = float((s - 1.0).abs().mean())
            out["pact/shift_active_frac"] = float((s != 1.0).to(torch.float32).mean())
        return out


class RoadNsTask(Task):
    """Enum for road_traffic under the drift layer."""

    ROAD_TRAFFIC = None

    @staticmethod
    def associated_class():
        return RoadNsClass
