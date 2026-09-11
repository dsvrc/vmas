#!/usr/bin/env python
#  Config-plumbing consistency for every simple_ns task.  Run this FIRST, always.
#
#      python simple_ns/check_plumbing.py
#
#  torch only -- no torchrl, no vmas -- so it runs before anything heavy is
#  imported and catches the failures that otherwise only appear on the cluster,
#  minutes into a launch.
#
#  It exists because two of them already happened:
#
#    * `benchmarl/environments/__init__.py` listed SimpleNsTask in `tasks`
#      without importing it, so the very first launch died with a NameError
#      inside `import benchmarl`.  The import list and the task list are edited
#      in different places and nothing tied them together.
#    * A key present in a yaml and absent from the TaskConfig dataclass (or the
#      reverse) is a hydra struct error at run time, and a key the scenario never
#      pops is a knob nothing reads -- which reads exactly like a setting that
#      works.
#
#  Four hosts share one dial and one method, so any of these drifts silently
#  turns "same disturbance, different task" into "different experiments".

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from simple_ns.driver import DialParams  # noqa: E402
from pact1.core import PactParams  # noqa: E402

LAYER = ROOT / "simple_ns" / "layer.py"
ENV_INIT = ROOT / "benchmarl" / "environments" / "__init__.py"
ENV_DIR = ROOT / "benchmarl" / "environments" / "simple_ns"
YAML_DIR = ROOT / "benchmarl" / "conf" / "task" / "simple_ns"

HOSTS = ["transport", "sampling", "balance", "navigation"]

problems: List[str] = []


def report(label: str, items) -> None:
    items = sorted(items)
    print(f"  {label}: {items if items else 'none'}")
    if items:
        problems.append(label)


def tuple_keys(name: str, text: str) -> Set[str]:
    m = re.search(name + r"\s*=\s*\((.*?)\n\)", text, re.S)
    return set(re.findall(r'"([a-z0-9_]+)"', m.group(1))) if m else set()


def yaml_scalars(text: str) -> Dict[str, object]:
    out: Dict[str, object] = {}
    for k, raw in re.findall(r"^([a-z0-9_]+):\s*([^\s#]+)", text, re.M):
        v = raw.strip().strip('"').strip("'")
        if v.lower() in ("true", "false"):
            out[k] = v.lower() == "true"
        else:
            try:
                out[k] = float(v)
            except ValueError:
                out[k] = v
    return out


# ---------------------------------------------------------------------------
#  0. every name in `tasks` must actually be imported
# ---------------------------------------------------------------------------
print("== registry ==")
tree = ast.parse(ENV_INIT.read_text(encoding="utf-8"))
bound: Set[str] = set()
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        bound.update(a.asname or a.name for a in node.names)
listed: List[str] = []
for node in tree.body:
    if isinstance(node, ast.Assign) and getattr(node.targets[0], "id", None) == "tasks":
        listed = [e.id for e in node.value.elts]
report("task enums listed but never imported", [t for t in listed if t not in bound])
report("simple_ns missing from the task list", [] if "SimpleNsTask" in listed else ["SimpleNsTask"])

layer = LAYER.read_text(encoding="utf-8")
NS_K = tuple_keys("NS_KWARGS", layer)
PACT_K = tuple_keys("PACT_KWARGS", layer)

#  Two keys are DELIBERATE per-instance overrides and must not be compared
#  against the dataclass default:
#
#    ns_severity  the experimental variable; transport's operating point is 2.0,
#                 calibrated on simple_ns/calibrate.py before any training
#    pact_mu      swept, not inherited: URB's 0.999 has a 1000-step memory
#                 against a 100-step driver cycle
#
#  They are checked for PRESENCE and for cross-host consistency instead, which is
#  what actually matters -- a missing key silently falls back to the dataclass
#  default and the run is then not the run you think it is.
DECLARED_OVERRIDES = {"ns_severity", "pact_mu"}

DIAL_PAIRS = [
    ("period", "ns_period"),
    ("wet_fraction", "ns_wet_fraction"),
    ("loss_at_sigma1", "ns_loss_at_sigma1"),
    ("rho", "ns_rho"),
    ("n_types", "ns_n_types"),
    ("recv_spread", "ns_recv_spread"),
    ("send_spread", "ns_send_spread"),
    ("kernel_lambda", "ns_kernel_lambda"),
    ("y_clip", "ns_y_clip"),
]


def same(want, got) -> bool:
    if isinstance(want, bool) or isinstance(got, (bool, str)) or isinstance(want, str):
        return want == got
    return abs(float(got) - float(want)) < 1e-12


shared: Dict[str, Dict[str, object]] = {}

for host in HOSTS:
    print(f"\n== {host} ==")
    dc_path, y_path = ENV_DIR / f"{host}.py", YAML_DIR / f"{host}.yaml"
    if not dc_path.exists() or not y_path.exists():
        report("missing task config files", [str(p) for p in (dc_path, y_path) if not p.exists()])
        continue
    dc_text, y_text = dc_path.read_text(encoding="utf-8"), y_path.read_text(encoding="utf-8")
    dc_keys = set(re.findall(r"^\s{4}(\w+):\s*\w", dc_text, re.M))
    y_keys = set(re.findall(r"^([a-z0-9_]+):", y_text, re.M)) - {"defaults"}

    # the hydra defaults line has to name the ConfigStore entry, which benchmarl
    # registers as "<env>_<task>_config"
    want_default = f"simple_ns_{host}_config"
    report("hydra defaults entry wrong",
           [] if want_default in y_text else [f"expected {want_default}"])

    report("scenario keys with no dataclass field", (NS_K | PACT_K) - dc_keys)
    report("yaml keys with no dataclass field", y_keys - dc_keys)
    report("dataclass fields with no yaml value", dc_keys - y_keys)
    report(
        "ns_/pact_ fields the layer never pops",
        {k for k in dc_keys if k.startswith(("ns_", "pact_")) and k not in (NS_K | PACT_K)},
    )

    yv = yaml_scalars(y_text)
    report("declared overrides missing from the yaml",
           [k for k in DECLARED_OVERRIDES if k not in yv])
    mismatch, compared = [], 0
    for field, key in DIAL_PAIRS:
        if key not in yv:
            continue
        compared += 1
        want = getattr(DialParams(), field)
        if not same(want, yv[key]):
            mismatch.append(f"{key}: DialParams.{field}={want!r} yaml={yv[key]!r}")
    if compared < len(DIAL_PAIRS):
        mismatch.append(f"only {compared} of {len(DIAL_PAIRS)} compared -- the yaml parse missed keys")
    report(f"dial defaults that disagree with the yaml ({compared} compared)", mismatch)

    shared[host] = {k: v for k, v in yv.items() if k.startswith(("ns_", "pact_"))}

# ---------------------------------------------------------------------------
#  the four hosts must declare the SAME dial and the SAME method, except for the
#  severity, which is calibrated per host.  If they drift, "one disturbance,
#  four tasks" stops being true and the cross-host comparison measures nothing.
# ---------------------------------------------------------------------------
print("\n== across hosts ==")
#  Severity is calibrated per host; everything else in the dial and the method
#  must be identical, or "one disturbance, four tasks" is not true.
PER_HOST_OK = {"ns_severity"}
ref = HOSTS[0]
drift = []
for h in HOSTS[1:]:
    if h not in shared or ref not in shared:
        continue
    for k in sorted(set(shared[ref]) | set(shared[h])):
        if k in PER_HOST_OK:
            continue
        if shared[ref].get(k) != shared[h].get(k):
            drift.append(f"{k}: {ref}={shared[ref].get(k)!r} {h}={shared[h].get(k)!r}")
report("dial/method settings that differ between hosts", drift)

print()
print("PLUMBING OK" if not problems else f"PROBLEMS in {len(problems)} place(s)")
raise SystemExit(1 if problems else 0)
