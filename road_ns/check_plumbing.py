#!/usr/bin/env python
#  Config-plumbing consistency, for EVERY road_ns task.  Run this FIRST, always.
#
#      python road_ns/check_plumbing.py
#
#  A key that exists in one place and not another is exactly how a partially
#  overridden config silently runs a different environment.  This has bitten
#  this project once already, when Phase 0 read a dataclass default while
#  training read a different yaml value -- every offline number described an
#  environment that never trained, and nothing looked wrong.
#
#  Two hosts now carry the same medium, which doubles the surface: a key added
#  to the dial has to reach BOTH task configs or the two hosts quietly run
#  different physics, and the whole point of having two is that they do not.
#
#  torch only; no torchrl, so it runs before anything heavy is imported.

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Dict, Set

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from road_ns.dial import DialParams  # noqa: E402
from pact1.core import PactParams  # noqa: E402

SCEN = ROOT / "road_ns" / "scenario.py"
FLOW = ROOT / "road_ns" / "flow.py"
ENV_DIR = ROOT / "benchmarl" / "environments" / "road_ns"
YAML_DIR = ROOT / "benchmarl" / "conf" / "task" / "road_ns"

#: task -> the host module whose own kwargs it must also declare
TASKS = {"road_traffic": None, "lanelet_flow": FLOW}

problems: list[str] = []


def report(label: str, items) -> None:
    items = sorted(items)
    print(f"  {label}: {items if items else 'none'}")
    if items:
        problems.append(label)


def tuple_keys(name: str, text: str) -> Set[str]:
    m = re.search(name + r"\s*=\s*\((.*?)\n\)", text, re.S)
    return set(re.findall(r'"([a-z0-9_]+)"', m.group(1))) if m else set()


def parse_yaml_scalars(text: str) -> Dict[str, object]:
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


scen = SCEN.read_text(encoding="utf-8")
NS_K = tuple_keys("NS_KWARGS", scen)
PACT_K = tuple_keys("PACT_KWARGS", scen)

# dial / method defaults that the yaml must agree with
DIAL_PAIRS = [
    ("severity", "ns_severity"),
    ("period", "ns_period"),
    ("wet_fraction", "ns_wet_fraction"),
    ("alpha", "ns_alpha"),
    ("mean_preserve", "ns_mean_preserve"),
]
PACT_PAIRS = [
    ("mu", "pact_mu"),
    ("p0", "pact_p0"),
    ("kappa", "pact_kappa"),
    ("y_clip", "pact_y_clip"),
    ("shift_mode", "pact_shift_mode"),
    ("shift_clip", "pact_shift_clip"),
]


def same(want, got) -> bool:
    if isinstance(want, bool) or isinstance(got, (bool, str)) or isinstance(want, str):
        return want == got
    return abs(float(got) - float(want)) < 1e-12


for task, host_mod in TASKS.items():
    print(f"\n== {task} ==")
    dc_text = (ENV_DIR / f"{task}.py").read_text(encoding="utf-8")
    yaml_text = (YAML_DIR / f"{task}.yaml").read_text(encoding="utf-8")
    dc_keys = set(re.findall(r"^\s{4}(\w+):\s*\w", dc_text, re.M))
    yaml_keys = set(re.findall(r"^([a-z0-9_]+):", yaml_text, re.M)) - {"defaults"}

    host_k = tuple_keys("FLOW_KWARGS", host_mod.read_text(encoding="utf-8")) if host_mod else set()

    report("scenario keys with no dataclass field", (NS_K | PACT_K) - dc_keys)
    report("yaml keys with no dataclass field", yaml_keys - dc_keys)
    report("dataclass fields with no yaml value", dc_keys - yaml_keys)
    # every ns_/pact_ dataclass field must be consumed by the scenario, or it is
    # a knob nothing reads
    report(
        "ns_/pact_ fields the scenario never pops",
        {k for k in dc_keys if k.startswith(("ns_", "pact_")) and k not in (NS_K | PACT_K)},
    )
    if host_k:
        # the host's own kwargs must be declared too, minus the ones the task
        # class supplies itself (num_envs, seed, ...) and the shared geometry
        report(
            "host kwargs the task config never sets",
            {k for k in host_k if k not in dc_keys and k not in ("dt", "max_speed")},
        )

    yv = parse_yaml_scalars(yaml_text)
    mismatch, compared = [], 0
    for obj, pairs in ((DialParams(), DIAL_PAIRS), (PactParams(), PACT_PAIRS)):
        for field, key in pairs:
            if key not in yv:
                continue
            compared += 1
            want, got = getattr(obj, field), yv[key]
            if not same(want, got):
                mismatch.append(f"{key}: {type(obj).__name__}.{field}={want!r} yaml={got!r}")
    if compared < len(DIAL_PAIRS) + len(PACT_PAIRS):
        mismatch.append(
            f"only {compared} of {len(DIAL_PAIRS) + len(PACT_PAIRS)} defaults compared "
            "-- the yaml parse missed keys"
        )
    report(f"defaults that disagree with the yaml ({compared} compared)", mismatch)

# The two tasks must declare the SAME dial and the SAME method.  If they drift,
# "same medium, different vehicle model" stops being true and the comparison
# between the hosts measures something nobody chose.
shared: Dict[str, Dict[str, object]] = {}
for task in TASKS:
    yv = parse_yaml_scalars((YAML_DIR / f"{task}.yaml").read_text(encoding="utf-8"))
    shared[task] = {k: v for k, v in yv.items() if k.startswith(("ns_", "pact_"))}
a, b = list(TASKS)
drift = [
    f"{k}: {a}={shared[a].get(k)!r} {b}={shared[b].get(k)!r}"
    for k in sorted(set(shared[a]) | set(shared[b]))
    if shared[a].get(k) != shared[b].get(k)
]
print("\n== across hosts ==")
report("dial/method settings that differ between the two tasks", drift)

print()
print("PLUMBING OK" if not problems else f"PROBLEMS in {len(problems)} place(s)")
raise SystemExit(1 if problems else 0)
