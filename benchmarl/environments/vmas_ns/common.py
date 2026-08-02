#  BenchMARL task wiring for the non-stationary VMAS family (``vmas_ns``).
#
#  Run it exactly like any other BenchMARL task:
#
#      python pact/run.py algorithm=ippo task=vmas_ns/navigation_pcw
#
#  (pact/run.py rather than benchmarl/run.py so the imported package and the
#  hydra configs are guaranteed to come from the same checkout -- see pact/run.py.)
#
#  The scenario is passed to ``VmasEnv`` as an *instance*, so nothing has to be
#  copied into the installed ``vmas`` package.

from __future__ import annotations

import copy
import math
from typing import Any, Callable, Dict, List, Optional

import torch
from tensordict import TensorDictBase
from torchrl.envs import EnvBase, Transform
from torchrl.envs.libs.vmas import VmasEnv

from benchmarl.environments.common import Task
from benchmarl.environments.vmas.common import VmasClass
from benchmarl.environments.vmas_ns.pact import PactTransform, Phase1ProbeTransform
from benchmarl.environments.vmas_ns.pcw_core import PcwParams, per_step_cosine
from benchmarl.utils import DEVICE_TYPING

#: Config keys consumed by the non-stationarity (passed to the scenario).
NS_CONFIG_KEYS = (
    "ns_severity",
    "ns_gain",
    "ns_rho",
    "ns_driver_period",
    "ns_phase_spread",
    "ns_freeze_driver",
)

#: Config keys consumed by PACT (env-side wrapper; never reach the scenario).
PACT_CONFIG_KEYS = (
    "pact_enabled",
    "pact_oracle",
    "pact_beta_max",
    "pact_beta_mode",
    "pact_beta_ema",
    "pact_obs_features",
    "pact_gate",
    "pact_gate_tol",
)


class PactGateError(RuntimeError):
    """Raised when the Phase-2 arithmetic gate fails.

    The gate certifies index order, reset masking and the one-step timing
    contract.  It is gait-independent and exact, so a failure is a wiring bug --
    there is nothing to tune and no point continuing the run.
    """


class VmasNsClass(VmasClass):
    """VMAS tasks carrying a category-C non-stationarity, plus the PACT wrapper."""

    # ------------------------------------------------------------------
    # environment construction
    # ------------------------------------------------------------------

    def _scenario_kwargs(self) -> Dict[str, Any]:
        """Task config minus the keys PACT owns; the rest goes to the scenario."""
        config = copy.deepcopy(self.config)
        for key in PACT_CONFIG_KEYS:
            config.pop(key, None)
        missing = [key for key in NS_CONFIG_KEYS if key not in config]
        if missing:
            raise ValueError(
                f"task config for {self.name} is missing non-stationarity keys {missing}; "
                "a partially-overridden config would silently run a different environment"
            )
        return config

    def _pcw_params(self) -> PcwParams:
        freeze = self.config.get("ns_freeze_driver", None)
        return PcwParams(
            severity=float(self.config.get("ns_severity", 0.0)),
            gain=float(self.config.get("ns_gain", 4.0)),
            rho=float(self.config.get("ns_rho", 0.8)),
            driver_period=int(self.config.get("ns_driver_period", 2000)),
            phase_spread=bool(self.config.get("ns_phase_spread", True)),
            freeze_driver=None if freeze is None else float(freeze),
        )

    def _make_scenario(self):
        if self.name.lower() != "navigation_pcw":
            raise ValueError(f"unknown vmas_ns task {self.name!r}")
        from benchmarl.environments.vmas_ns.scenario import Scenario

        return Scenario()

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device: DEVICE_TYPING,
    ) -> Callable[[], EnvBase]:
        config = self._scenario_kwargs()
        scenario_factory = self._make_scenario
        return lambda: VmasEnv(
            # A scenario *instance*: no file has to be dropped into the
            # installed vmas package, so there is no "same-file-different-module"
            # trap where knobs get set on the wrong copy.
            scenario=scenario_factory(),
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            categorical_actions=True,
            clamp_actions=True,
            **config,
        )

    def supports_discrete_actions(self) -> bool:
        # The compensation law is a rotation of a continuous thrust vector.
        return False

    @staticmethod
    def env_name() -> str:
        return "vmas_ns"

    # ------------------------------------------------------------------
    # PACT wiring
    # ------------------------------------------------------------------

    @property
    def pact_enabled(self) -> bool:
        return bool(self.config.get("pact_enabled", False))

    def group_and_agents(self, env: EnvBase):
        group_map = self.group_map(env)
        if len(group_map) != 1:
            raise ValueError(
                f"navigation_pcw expects a single agent group, got {list(group_map)}"
            )
        group = next(iter(group_map))
        return group, len(group_map[group])

    def action_bounds(self, env: EnvBase, group: str):
        spec = env.full_action_spec_unbatched[(group, "action")]
        return spec.space.low.clone(), spec.space.high.clone()

    def beta_max(self) -> float:
        configured = self.config.get("pact_beta_max", None)
        if configured is not None:
            return float(configured)
        # Convention from the PACT checklist: beta_max ~ 1.3x the peak driver
        # value, so the gain can reach c with headroom but cannot run away.
        peak = self._pcw_params().peak_c
        return max(1.3 * peak, 0.1)

    def get_env_transforms(self, env: EnvBase) -> List[Transform]:
        if not self.pact_enabled:
            return []
        group, n_agents = self.group_and_agents(env)
        low, high = self.action_bounds(env, group)
        return [
            PactTransform(
                group=group,
                n_agents=n_agents,
                params=self._pcw_params(),
                beta_max=self.beta_max(),
                action_low=low,
                action_high=high,
                beta_mode=str(self.config.get("pact_beta_mode", "affine")),
                beta_ema=float(self.config.get("pact_beta_ema", 0.3)),
                oracle=bool(self.config.get("pact_oracle", False)),
                obs_features=bool(self.config.get("pact_obs_features", True)),
            )
        ]

    def make_phase1_probe(self, env: EnvBase, beta: float) -> Phase1ProbeTransform:
        """Build the Phase-1 scripted-compensation probe for this task."""
        group, n_agents = self.group_and_agents(env)
        low, high = self.action_bounds(env, group)
        return Phase1ProbeTransform(
            group=group,
            n_agents=n_agents,
            params=self._pcw_params(),
            beta=beta,
            action_low=low,
            action_high=high,
        )

    # ------------------------------------------------------------------
    # diagnostics -- runs on every collection, via the stock run.py
    # ------------------------------------------------------------------

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for group in self.group_map_keys(batch):
            info_key = ("next", group, "info")
            if info_key not in batch.keys(include_nested=True):
                continue
            info = batch.get(info_key)
            out.update(self._ns_diagnostics(info))
            out.update(self._pact_diagnostics(info))
        return out

    @staticmethod
    def group_map_keys(batch: TensorDictBase) -> List[str]:
        """Group names present in a collected batch (those carrying an ``info``)."""
        keys = batch.get("next").keys(include_nested=True, leaves_only=False)
        return [
            key
            for key in keys
            if isinstance(key, str)
            and ("next", key, "info") in batch.keys(include_nested=True)
        ]

    @staticmethod
    def _get(info: TensorDictBase, key: str) -> Optional[torch.Tensor]:
        if key not in info.keys():
            return None
        value = info.get(key).to(torch.float32)
        return value.squeeze(-1) if value.shape[-1] == 1 else value

    def _ns_diagnostics(self, info: TensorDictBase) -> Dict[str, float]:
        """Calibration read-outs.  These are the numbers that tell you whether
        ``ns_gain`` is sized to the policy's *actual* operating scale."""
        out: Dict[str, float] = {}
        abs_theta = self._get(info, "ns_abs_theta")
        if abs_theta is not None:
            flat = abs_theta.reshape(-1)
            out["ns/abs_theta_mean"] = float(flat.mean())
            out["ns/abs_theta_p95"] = float(flat.quantile(0.95))
            out["ns/abs_theta_max"] = float(flat.max())
            out["ns/abs_theta_gt_90deg_frac"] = float(
                (flat > (math.pi / 2)).to(torch.float32).mean()
            )
        x2 = self._get(info, "ns_x2")
        if x2 is not None:
            out["ns/abs_x2_rms"] = float(x2.pow(2).mean().sqrt())
        for key, name in (("ns_c", "c"), ("ns_A", "A")):
            value = self._get(info, key)
            if value is not None:
                out[f"ns/{name}_mean"] = float(value.mean())
        return out

    def _pact_diagnostics(self, info: TensorDictBase) -> Dict[str, float]:
        out: Dict[str, float] = {}
        pact_x2 = self._get(info, "pact_x2")
        env_x2 = self._get(info, "ns_x2")
        if pact_x2 is None or env_x2 is None:
            return out

        # ---- the one hard gate ----------------------------------------
        # Per-step cosine between the agent-vectors, NOT a correlation pooled
        # across the driver's range: a pooled correlation reads ~0.95 off the
        # varying-c fan even when every point is exact.
        cos, valid = per_step_cosine(pact_x2, env_x2)
        valid_frac = float(valid.to(torch.float32).mean())
        out["pact/gate_valid_frac"] = valid_frac
        if valid.any():
            cos_valid = cos[valid]
            out["pact/gate_cosine_mean"] = float(cos_valid.mean())
            out["pact/gate_cosine_min"] = float(cos_valid.min())
            rel_err = (pact_x2 - env_x2).abs() / env_x2.abs().clamp_min(1e-6)
            out["pact/gate_rel_err_max"] = float(rel_err[valid].max())
            self._check_gate(out["pact/gate_cosine_mean"], valid_frac)

        # ---- beta tracking: the actual research question ---------------
        beta = self._get(info, "pact_beta")
        c_true = self._get(info, "ns_c")
        driver = self._get(info, "ns_A")
        if beta is not None:
            out["pact/beta_mean"] = float(beta.mean())
            if c_true is not None:
                out["pact/beta_minus_c_mean"] = float((beta - c_true).mean())
                out["pact/beta_abs_err_mean"] = float((beta - c_true).abs().mean())
            if driver is not None:
                # Success looks like beta -> c at the peak and beta -> 0 in the
                # trough.  A beta that is flat across phases is phase-blind and
                # wants recurrence / a CTDE critic, not more tuning.
                peak = driver > 0.5
                if peak.any():
                    out["pact/beta_peak"] = float(beta[peak].mean())
                    if c_true is not None:
                        out["pact/c_peak"] = float(c_true[peak].mean())
                trough = ~peak
                if trough.any():
                    out["pact/beta_trough"] = float(beta[trough].mean())
                    if c_true is not None:
                        out["pact/c_trough"] = float(c_true[trough].mean())

        # ---- residual actually felt by the plant ----------------------
        theta_hat = self._get(info, "pact_theta_hat")
        theta_true = self._get(info, "ns_theta_applied")
        if theta_hat is not None and theta_true is not None:
            out["pact/residual_abs_mean"] = float((theta_true - theta_hat).abs().mean())
            out["pact/uncompensated_abs_mean"] = float(theta_true.abs().mean())

        sat = self._get(info, "pact_sat")
        if sat is not None:
            out["pact/sat_frac"] = float(sat.mean())
        return out

    def _check_gate(self, cosine: float, valid_frac: float) -> None:
        if not self.pact_enabled or not bool(self.config.get("pact_gate", True)):
            return
        tol = float(self.config.get("pact_gate_tol", 0.999))
        # Below ~10% valid steps the batch is nearly all episode-starts where the
        # accumulator is still zero; the statistic is not yet meaningful.
        if valid_frac < 0.1 or cosine >= tol:
            return
        raise PactGateError(
            f"PACT arithmetic gate failed: mean per-step cosine between the computed "
            f"waveform and the environment's accumulator is {cosine:.6f} < {tol}. "
            "This certifies leak wiring only (index order, reset masking, one-step "
            "timing) and is gait-independent, so this is a wiring bug. Check that "
            "ns_rho/ns_gain match between task config and scenario, that the "
            "PACT transform's reset is reached on partial resets, and that the "
            "observation still starts with [pos_x, pos_y]."
        )


class VmasNsTask(Task):
    """Enum for non-stationary VMAS tasks."""

    NAVIGATION_PCW = None

    @staticmethod
    def associated_class():
        return VmasNsClass
