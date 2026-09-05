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
report("config-base keys missing from the kwarg lists", base_keys - SLC_K - PACT_K)
report("kwarg-list keys missing from the config base", (SLC_K | PACT_K) - base_keys)

for y in sorted(Path("benchmarl/conf/task/vmas_slc").glob("*.yaml")):
    ykeys = set(re.findall(r"^([a-z_]\w*):", y.read_text(), re.M)) - {"defaults"}
    dc = SRC / (y.stem + ".py")
    dkeys = set(re.findall(r"^\s*(\w+):\s*\w", dc.read_text(), re.M)) | base_keys
    report(f"{y.name} keys with no dataclass field", ykeys - dkeys)
    report(f"{y.name} dataclass fields with no yaml value", dkeys - ykeys)

print()
print("PLUMBING OK" if not problems else f"PROBLEMS in {len(problems)} place(s)")
raise SystemExit(1 if problems else 0)
