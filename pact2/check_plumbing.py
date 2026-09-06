"""Config-plumbing consistency check: a key that exists in one place and not
another is exactly how a partially-overridden config silently runs a different
environment.  No torchrl needed."""

import dataclasses
import re
from pathlib import Path

from pact2._bootstrap import load_cores

slc, pact = load_cores()
SRC = Path("benchmarl/environments/vmas_slc")


def grab(name, text):
    m = re.search(name + r"\s*=\s*\((.*?)\n\)", text, re.S)
    return set(re.findall(r'"([a-z0-9_]+)"', m.group(1)))


scen = (SRC / "scenario.py").read_text()
SLC_K = grab("SLC_KWARGS", scen)
PACT_K = grab("PACT_KWARGS", scen)

slc_fields = {f.name for f in dataclasses.fields(slc.SlcParams) if f.init}
pact_fields = {f.name for f in dataclasses.fields(pact.PactParams)}

problems = []


def report(label, items):
    print(f"{label}: {sorted(items) if items else 'none'}")
    if items:
        problems.append(label)


report("SlcParams fields not reachable from a task config",
       slc_fields - {k[4:] for k in SLC_K})
# trace_gate_p0 is a constant of the trace-gate ABLATION, deliberately not a
# task knob: it exists to reproduce a failure, not to be tuned.
report("PactParams fields not reachable from a task config",
       pact_fields - {k[5:] for k in PACT_K} - {"trace_gate_p0"})
report("SLC_KWARGS with no matching SlcParams field",
       {k for k in SLC_K if k[4:] not in slc_fields})
report("PACT_KWARGS with no matching PactParams field",
       {k for k in PACT_K if k[5:] not in pact_fields})

base = (SRC / "_config_base.py").read_text()
base_keys = set(re.findall(r"^\s*(slc_\w+|pact_\w+):", base, re.M))
# Keys the TASK CLASS owns and strips before the config reaches the scenario --
# diagnostics plumbing, not environment physics.
task_only = set(
    re.findall(
        r'TASK_ONLY_KEYS\s*=\s*\((.*?)\)',
        (SRC / "common.py").read_text(),
        re.S,
    )[0].replace('"', " ").replace(",", " ").split()
)
report(
    "config-base keys missing from the kwarg lists",
    base_keys - SLC_K - PACT_K - task_only,
)
report("kwarg-list keys missing from the config base", (SLC_K | PACT_K) - base_keys)

for y in sorted(Path("benchmarl/conf/task/vmas_slc").glob("*.yaml")):
    ykeys = set(re.findall(r"^([a-z_]\w*):", y.read_text(), re.M)) - {"defaults"}
    dc = SRC / (y.stem + ".py")
    dkeys = set(re.findall(r"^\s*(\w+):\s*\w", dc.read_text(), re.M)) | base_keys
    report(f"{y.name} keys with no dataclass field", ykeys - dkeys)
    report(f"{y.name} dataclass fields with no yaml value", dkeys - ykeys)


# ---------------------------------------------------------------------------
#  Dataclass defaults must equal the shipped yaml.
#
#  Phase 0 (ceiling.py, calibrate.py, selfcheck.py) constructs SlcParams and
#  PactParams directly and therefore reads the DATACLASS defaults; training
#  reads the YAML.  When those diverge the calibration is measured against a
#  different environment than the one that trains, and nothing looks wrong --
#  it happened once here, with aclr 1e-3 vs 0.25, and it silently invalidated
#  every Part C number.
# ---------------------------------------------------------------------------

# pact_enabled is the ARM SWITCH: False in the yaml so the default task is
# blind, True in the dataclass so a hand-built compensator is usable.  The
# disagreement is the point.
#   pact_trace_gate_p0 is a constant of the trace-gate ablation and has no yaml
#   entry at all -- it exists to reproduce a failure, not to be tuned.
DEFAULT_EXEMPT = {"pact_enabled", "pact_trace_gate_p0"}


def _parse(text):
    out = {}
    for key, raw in re.findall(r"^([a-z0-9_]+):\s*([^\s#]+)", text, re.M):
        v = raw.strip()
        if v in ("true", "True", "false", "False"):
            out[key] = v.lower() == "true"
        else:
            try:
                out[key] = float(v)
            except ValueError:
                out[key] = v
    return out


yaml_vals = _parse(Path("benchmarl/conf/task/vmas_slc/sampling.yaml").read_text())
defaults = {}
for prefix, cls in (("slc_", slc.SlcParams), ("pact_", pact.PactParams)):
    for f in dataclasses.fields(cls):
        if f.init and f.default is not dataclasses.MISSING:
            defaults[prefix + f.name] = f.default

mismatched = []
compared = 0
for key, want in sorted(defaults.items()):
    if key in DEFAULT_EXEMPT or key not in yaml_vals:
        continue
    compared += 1
    got = yaml_vals[key]
    same = (
        abs(got - want) < 1e-12
        if isinstance(want, (int, float))
        and not isinstance(want, bool)
        and isinstance(got, (int, float))
        else got == want
    )
    if not same:
        mismatched.append(f"{key}: dataclass={want!r} yaml={got!r}")
# A comparison that silently matched nothing would pass vacuously, which is the
# same failure one level up.
expected = len(defaults) - len(DEFAULT_EXEMPT)
if compared < expected:
    mismatched.append(
        f"only {compared}/{expected} defaults were compared -- the yaml parse "
        "missed keys, so this check would pass vacuously"
    )
report(
    f"dataclass defaults that disagree with sampling.yaml ({compared} compared)",
    mismatched,
)

print()
print("PLUMBING OK" if not problems else f"PROBLEMS in {len(problems)} place(s)")
raise SystemExit(1 if problems else 0)
