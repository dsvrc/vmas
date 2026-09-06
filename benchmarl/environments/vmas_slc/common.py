#  BenchMARL task wiring for the Shared Link Contention family (``vmas_slc``).
#
#      python pact2/run.py algorithm=ippo task=vmas_slc/sampling
#
#  (``pact2/run.py`` rather than ``benchmarl/run.py`` so the imported package and
#  the hydra configs are guaranteed to come from the same checkout.)
#
#  The scenario is handed to ``VmasEnv`` as an *instance*, so nothing has to be
#  copied into the installed ``vmas`` package and a vmas upgrade cannot silently
#  revert the non-stationarity.

from __future__ import annotations

import copy
import csv
import math
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from tensordict import TensorDictBase
from torchrl.envs import EnvBase
from torchrl.envs.libs.vmas import VmasEnv

from benchmarl.environments.common import Task
from benchmarl.environments.vmas.common import VmasClass
from benchmarl.environments.vmas_slc.scenario import (
    make_slc_scenario,
    PACT_KWARGS,
    SLC_KWARGS,
    STOCK_SCENARIOS,
)
from benchmarl.utils import DEVICE_TYPING

__all__ = [
    "VmasSlcClass",
    "VmasSlcTask",
    "SlcDialError",
    "write_diagnostics_row",
]


def write_diagnostics_row(path: Path, state: Dict[str, Any], row: Dict[str, float]) -> None:
    """Append one diagnostics row, one per collection iteration.

    **Never append across schema changes.**  Two runs with different column
    counts in one file misaligned every field in the second segment on POWER and
    produced an impossible ``cond_psi`` of 0.02, costing a full analysis pass.  A
    header mismatch rolls the old file aside instead of appending to it.

    Deliberately a free function so it can be tested without torchrl.  ``state``
    is a mutable dict carrying ``n`` (the row counter) and ``fields`` (the
    header, resolved on the first call).
    """
    fields = ["iteration"] + sorted(row)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if state.get("fields") is None:
            if path.exists():
                with path.open("r", newline="") as fh:
                    existing = next(csv.reader(fh), None)
                if existing is not None and existing != fields:
                    rolled = path.with_name(
                        f"{path.stem}.{int(path.stat().st_mtime)}{path.suffix}"
                    )
                    path.rename(rolled)
                    print(
                        f"[slc] diagnostics schema changed "
                        f"({len(existing)} -> {len(fields)} columns); rolled the "
                        f"old file to {rolled}"
                    )
            fresh = not path.exists()
            with path.open("a", newline="") as fh:
                if fresh:
                    csv.DictWriter(fh, fieldnames=fields).writeheader()
            state["fields"] = fields
            print(f"[slc] diagnostics -> {path}")
        elif fields != state["fields"]:
            # cannot happen within one run; a silent misalignment is exactly the
            # failure this guard exists for
            raise RuntimeError(
                f"diagnostics schema changed mid-run: {state['fields']} -> {fields}"
            )
        with path.open("a", newline="") as fh:
            csv.DictWriter(
                fh, fieldnames=state["fields"], extrasaction="ignore"
            ).writerow({"iteration": state.get("n", 0), **row})
        state["n"] = state.get("n", 0) + 1
    except OSError as exc:
        # a logging failure must never take down a training run
        print(f"[slc] could not write diagnostics to {path}: {exc}")


class SlcDialError(RuntimeError):
    """Raised when the severity dial is configured live but provably does
    nothing.

    A silently discarded dial produced five rows of pure scenario noise on
    POWER, and the failure is invisible: every curve looks healthy.  This is a
    wiring bug, not something to tune, so the run stops.
    """


class VmasSlcClass(VmasClass):
    """VMAS tasks under Shared Link Contention, with PACT as an optional layer.

    The dial lives **below** the method in the class hierarchy and is read from
    the task config, so every arm -- MAPPO, IPPO, MASAC, anything -- runs the
    identical physics.
    """

    # ------------------------------------------------------------------
    #  construction
    # ------------------------------------------------------------------

    #: Task-config keys owned by this class, never forwarded to the scenario.
    TASK_ONLY_KEYS = ("slc_diag_csv",)

    def _split_config(self):
        config = copy.deepcopy(self.config)
        missing = [k for k in ("slc_severity", "pact_enabled") if k not in config]
        if missing:
            raise ValueError(
                f"task config for {self.name} is missing {missing}; a partially "
                "overridden config would silently run a different environment"
            )
        for key in self.TASK_ONLY_KEYS:
            config.pop(key, None)
        return config

    @property
    def stock_name(self) -> str:
        name = self.name.lower()
        if name not in STOCK_SCENARIOS:
            raise ValueError(
                f"unknown vmas_slc task {self.name!r}; expected one of "
                f"{STOCK_SCENARIOS}"
            )
        return name

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
        config = self._split_config()
        stock = self.stock_name
        pact = self.pact_enabled
        return lambda: VmasEnv(
            scenario=make_slc_scenario(stock, pact),
            num_envs=num_envs,
            continuous_actions=continuous_actions,
            seed=seed,
            device=device,
            categorical_actions=True,
            clamp_actions=True,
            **config,
        )

    def supports_discrete_actions(self) -> bool:
        """Yes -- but read ``sat_frac`` before believing a discrete PACT arm.

        A.7 says a *discrete channel* makes trust a threshold rather than a
        scale.  That is about the **harm** being a permutation, and SLC's harm
        is not: it is a continuous multiplicative gain on the force vector,
        applied in ``process_action`` **below the action interface**.  The
        policy's action set does not make the channel discrete, so the method is
        unchanged and QMIX/VDN/IQL can run as ordinary arms.

        What *does* change is whether the inverse fits.  At VMAS's default
        3-way discretisation the policy commands land on ``{-u_range, 0,
        +u_range}``, so ``a / (1 - c)`` for any non-zero command immediately
        leaves the action box and the correction is lost to the rail -- a
        rail-pinned delta is a constant bias, not a compensation.  Give the
        discrete arms headroom with ``slc_discrete_nvec`` (7 or 9) and report
        ``slc/sat_frac`` alongside the result, or run them blind-only.
        """
        return True

    @staticmethod
    def env_name() -> str:
        return "vmas_slc"

    # ------------------------------------------------------------------
    #  diagnostics -- run on every collection through the stock run.py
    # ------------------------------------------------------------------

    def log_info(self, batch: TensorDictBase) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for group in self._groups_with_info(batch):
            info = batch.get(("next", group, "info"))
            out.update(self._dial_diagnostics(info))
            out.update(self._pact_diagnostics(info))
            out.update(self._return_diagnostics(batch, group))
        self._write_diagnostics_row(out)
        return out

    @staticmethod
    def _return_diagnostics(batch: TensorDictBase, group: str) -> Dict[str, float]:
        """Enough of the return to make the diagnostics CSV self-contained.

        This is per-step reward, NOT the episode return the training curve
        reports -- it is here so a row can be read on its own, not so it can be
        quoted as a result.
        """
        key = ("next", group, "reward")
        if key not in batch.keys(include_nested=True):
            return {}
        r = batch.get(key).to(torch.float32)
        return {"reward/step_mean": float(r.mean()), "reward/step_sum": float(r.sum())}

    # -- persistence --------------------------------------------------------

    def _diag_path(self) -> Path:
        configured = self.config.get("slc_diag_csv", "") or ""
        if configured:
            return Path(configured)
        # hydra chdirs into the run's own output directory, so cwd is already
        # per-run and two arms cannot collide.
        return Path(os.getcwd()) / "slc_pact_diagnostics.csv"

    def _write_diagnostics_row(self, row: Dict[str, float]) -> None:
        if not row:
            return
        state = getattr(self, "_diag_state", None)
        if state is None:
            state = {"n": 0, "fields": None}
            self._diag_state = state
        write_diagnostics_row(self._diag_path(), state, row)

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

    # -- the dial -----------------------------------------------------------

    def _dial_diagnostics(self, info: TensorDictBase) -> Dict[str, float]:
        out: Dict[str, float] = {}
        u = self._get(info, "slc_u")
        if u is None:
            return out
        flat = u.reshape(-1)
        out["slc/u_mean"] = float(flat.mean())
        out["slc/u_p95"] = float(flat.quantile(0.95))
        out["slc/u_max"] = float(flat.max())

        for key, name in (
            ("slc_c", "harm"),
            ("slc_A_eff", "A_eff"),
            ("slc_excess", "excess"),
            ("slc_sat", "sat_frac"),
            ("slc_quiet", "quiet_frac"),
        ):
            v = self._get(info, key)
            if v is not None:
                out[f"slc/{name}_mean"] = float(v.reshape(-1).mean())

        # A.5's counter-check: a perfectly constant Phi is unidentifiable.  POWER
        # ran 0.28 and called that healthy; anything under 0.05 means the
        # exertion functional is not varying enough to identify anything.
        phi = self._get(info, "slc_phi")
        if phi is not None:
            f = phi.reshape(-1)
            out["slc/phi_std_over_mean"] = float(
                f.std(unbiased=False) / f.mean().clamp_min(1e-9)
            )

        # Is the severity live?  ``dial_ratio`` should sit below 1 and
        # ``dial_skip`` -- the fraction of steps where g is EXACTLY 1 -- should be
        # ~0 on a busy shift and exactly 1 on the placebo shift.
        g = self._get(info, "slc_g")
        if g is not None:
            gf = g.reshape(-1)
            out["slc/dial_ratio"] = float(gf.mean())
            skip = float((gf >= 1.0).to(torch.float32).mean())
            out["slc/dial_skip"] = skip
            self._check_dial_live(skip, info)
        return out

    def _check_dial_live(self, dial_skip: float, info: TensorDictBase) -> None:
        severity = float(self.config.get("slc_severity", 0.0))
        p_quiet = float(self.config.get("slc_p_quiet", 0.0))
        if severity <= 0.0 or p_quiet >= 1.0:
            return
        if dial_skip < 0.999:
            return
        quiet = self._get(info, "slc_quiet")
        if quiet is not None and float(quiet.reshape(-1).mean()) > 0.999:
            return  # legitimately all-placebo this batch
        raise SlcDialError(
            f"slc_severity={severity} is configured but the capacity dial read "
            f"exactly 1.0 on {dial_skip:.4f} of steps with quiet shifts ruled "
            "out.  The dial is not reaching the physics.  Check that the task "
            "config keys actually arrive at the scenario (VmasEnv forwards "
            "unknown kwargs) and that slc_severity was not shadowed by a "
            "method-level block -- severity is TASK physics and must reach "
            "every arm."
        )

    # -- the method ---------------------------------------------------------

    def _pact_diagnostics(self, info: TensorDictBase) -> Dict[str, float]:
        out: Dict[str, float] = {}
        trust = self._get(info, "pact_applied_trust")
        if trust is None:
            return out

        # READ THESE TWO BEFORE ANY OTHER NUMBER.
        out["pact/applied_trust"] = float(trust.reshape(-1).mean())
        delta = self._get(info, "pact_delta_abs")
        if delta is not None:
            d = delta.reshape(-1)
            out["pact/delta_abs"] = float(d.mean())
            out["pact/delta_nonzero_frac"] = float((d > 0).to(torch.float32).mean())
        clip = self._get(info, "pact_delta_clip")
        if clip is not None:
            out["pact/delta_clip_frac"] = float(clip.reshape(-1).mean())

        # The local / coordination split.  The feedforward needs no peer
        # information, so it supports NO coordination claim -- report both or the
        # claim is not honest.  POWER measured 79% / 21%.
        ff = self._get(info, "pact_ff_abs")
        peer = self._get(info, "pact_peer_abs")
        own = self._get(info, "pact_own_abs")
        base = self._get(info, "pact_base_abs")
        if ff is not None and peer is not None:
            f, p = float(ff.reshape(-1).mean()), float(peer.reshape(-1).mean())
            o = float(own.reshape(-1).mean()) if own is not None else 0.0
            b = float(base.reshape(-1).mean()) if base is not None else 0.0
            out["pact/base_abs"] = b
            out["pact/ff_abs"] = f
            out["pact/peer_abs"] = p
            out["pact/own_abs"] = o
            total = f + p + o + b
            # Guard every ratio with NaN, never an epsilon: at the driver trough
            # the disturbance is genuinely ~0 and the ratio is meaningless.
            out["pact/peer_share"] = (p / total) if total > 0 else float("nan")

        fit = self._get(info, "pact_fit_gain")
        if fit is not None:
            f = fit.reshape(-1)
            f = f[torch.isfinite(f)]
            if f.numel():
                out["pact/fit_gain_now"] = float(f.mean())

        err = self._get(info, "pact_u_err")
        u = self._get(info, "slc_u")
        if err is not None:
            out["pact/u_err_abs"] = float(err.reshape(-1).abs().mean())
            if u is not None:
                base = float(u.reshape(-1).std(unbiased=False))
                out["pact/u_err_rel"] = (
                    float(err.reshape(-1).abs().mean()) / base
                    if base > 0
                    else float("nan")
                )

        for key, name in (
            ("pact_trP", "trP"),
            ("pact_clamp", "clamp_frac"),
            ("pact_n_updates", "n_updates"),
            ("pact_own_gain_coef", "own_gain_coef"),
            ("pact_u_hat", "u_hat"),
        ):
            v = self._get(info, key)
            if v is not None:
                out[f"pact/{name}"] = float(v.reshape(-1).mean())

        # cond: can theta be DECOMPOSED, not merely predicted?  Non-finite is a
        # VALUE and is reported as one rather than being silently dropped.
        cond = self._get(info, "pact_cond")
        if cond is not None:
            c = cond.reshape(-1)
            finite = c[torch.isfinite(c)]
            out["pact/cond_psi"] = float(finite.max()) if finite.numel() else float("inf")
            out["pact/cond_nonfinite_frac"] = float(
                (~torch.isfinite(c)).to(torch.float32).mean()
            )

        state = self._get(info, "pact_state")
        if state is not None:
            s = state.reshape(-1)
            out["pact/frac_alive"] = float((s >= 2.0).to(torch.float32).mean())
            out["pact/frac_inert"] = float((s <= 0.0).to(torch.float32).mean())
        return out


class VmasSlcTask(Task):
    """Enum for VMAS tasks carrying Shared Link Contention.

    Chosen for the reasons in ``pact2/README.md`` section 3: ``Phi`` cannot be
    driven to its floor without forfeiting reward, ``n_agents`` is free so the
    C.4 N-scaling prediction is testable, and every one of them runs at N=1 for
    the irreducibility certificate.
    """

    SAMPLING = None
    DISCOVERY = None
    NAVIGATION = None

    @staticmethod
    def associated_class():
        return VmasSlcClass
