#!/usr/bin/env python
#  Config-plumbing consistency.  Run this FIRST, always.
#
#      python road_ns/check_plumbing.py
#
#  A key that exists in one place and not another is exactly how a partially
#  overridden config silently runs a different environment.  This has bitten
#  this project once already, when Phase 0 read a dataclass default while
#  training read a different yaml value -- every offline number described an
#  environment that never trained, and nothing looked wrong.
#
#  torch only; no torchrl, so it runs before anything heavy is imported.

from __future__ import annotations

import dataclasses
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from road_ns.dial import DialParams  # noqa: E402
from pact1.core import PactParams  # noqa: E402

SCEN = ROOT / "road_ns" / "scenario.py"
DC = ROOT / "benchmarl" / "environments" / "road_ns" / "road_traffic.py"
YAML = ROOT / "benchmarl" / "conf" / "task" / "road_ns" / "road_traffic.yaml"

problems: list[str] = []


def report(label: str, items) -> None:
    items = sorted(items)
    print(f"{label}: {items if items else 'none'}")
    if items:
        problems.append(label)


def tuple_keys(name: str, text: str) -> set:
    m = re.search(name + r"\s*=\s*\((.*?)\n\)", text, re.S)
    return set(re.findall(r'"([a-z0-9_]+)"', m.group(1))) if m else set()


scen = SCEN.read_text(encoding="utf-8")
NS_K = tuple_keys("NS_KWARGS", scen)
PACT_K = tuple_keys("PACT_KWARGS", scen)

dc_keys = set(re.findall(r"^\s{4}(\w+):\s*\w", DC.read_text(encoding="utf-8"), re.M))
yaml_text = YAML.read_text(encoding="utf-8")
yaml_keys = set(re.findall(r"^([a-z0-9_]+):", yaml_text, re.M)) - {"defaults"}

report("scenario keys with no dataclass field", (NS_K | PACT_K) - dc_keys)
report("yaml keys with no dataclass field", yaml_keys - dc_keys)
report("dataclass fields with no yaml value", dc_keys - yaml_keys)

# every ns_/pact_ dataclass field must be consumed by the scenario, or it is a
# knob nothing reads
unconsumed = {
    k for k in dc_keys if k.startswith(("ns_", "pact_")) and k not in (NS_K | PACT_K)
}
report("ns_/pact_ fields the scenario never pops", unconsumed)


# --- the defaults must agree with the yaml --------------------------------
def parse_yaml_scalars(text: str) -> dict:
    out = {}
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


yv = parse_yaml_scalars(yaml_text)
# map dataclass field -> yaml key
pairs = [
    ("severity", "ns_severity"),
    ("period", "ns_period"),
    ("wet_fraction", "ns_wet_fraction"),
    ("alpha", "ns_alpha"),
    ("mean_preserve", "ns_mean_preserve"),
]
mismatch, compared = [], 0
for field, key in pairs:
    if key not in yv:
        continue
    want = getattr(DialParams(), field)
    got = yv[key]
    compared += 1
    same = (
        abs(float(got) - float(want)) < 1e-12
        if isinstance(want, (int, float)) and not isinstance(want, bool)
        else got == want
    )
    if not same:
        mismatch.append(f"{key}: DialParams.{field}={want!r} yaml={got!r}")

for field, key in (("mu", "pact_mu"), ("p0", "pact_p0"), ("kappa", "pact_kappa"),
                   ("y_clip", "pact_y_clip")):
    if key not in yv:
        continue
    want = getattr(PactParams(), field)
    compared += 1
    if abs(float(yv[key]) - float(want)) > 1e-12:
        mismatch.append(f"{key}: PactParams.{field}={want!r} yaml={yv[key]!r}")

if compared < len(pairs):
    mismatch.append(f"only {compared} defaults compared -- the yaml parse missed keys")
report(f"defaults that disagree with the yaml ({compared} compared)", mismatch)

print()
print("PLUMBING OK" if not problems else f"PROBLEMS in {len(problems)} place(s)")
raise SystemExit(1 if problems else 0)
