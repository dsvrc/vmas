#!/usr/bin/env python
#  Offline verification for the BASELINES.md baselines.  Run this FIRST, always.
#
#      python baselines/verify.py
#
#  torch only -- no torchrl, no tensordict, no vmas -- so it runs on a laptop
#  that cannot start a training job, and it catches the two classes of failure
#  that otherwise only appear minutes into a cluster launch:
#
#    * PLUMBING.  A yaml key with no dataclass field (or the reverse) is a hydra
#      struct error at run time; a config field the algorithm's __init__ never
#      names is a TypeError; an algorithm in the registry that is never imported
#      is a NameError inside `import benchmarl`.  All three are one-line
#      mistakes that cost a queue slot each.
#
#    * ARITHMETIC.  The published formulas -- HAPPO's factor, the ESO's pole
#      placement, LIAM's "everyone but me" index, the mean-field average, LCPO's
#      conjugate gradients and its out-of-distribution reservoir -- live in two
#      torch-only modules precisely so they can be checked here against their
#      definitions rather than against themselves.  A wrong observer gain does
#      not crash; it just tracks badly, and no training curve tells you which.
#
#  Exit code 0 means every check passed.

from __future__ import annotations

import ast
import importlib.util
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

import torch

ROOT = Path(__file__).resolve().parents[1]
ALGO_DIR = ROOT / "benchmarl" / "algorithms"
CONF_DIR = ROOT / "benchmarl" / "conf" / "algorithm"
TASK_DIR = ROOT / "benchmarl" / "conf" / "task" / "simple_ns"

#: algorithm name -> (the BASELINES.md section it answers, its doc)
BASELINES = {
    "happo": ("B1 trust region / sequential update", "happo.md"),
    "hasac": ("B1 off-policy, maximum entropy", "hasac.md"),
    "mfac": ("B4 mean-field MARL", "mfac.md"),
    "liam": ("B5 agent modelling", "liam.md"),
    "lcpo": ("B6 non-stationary RL, observed context", "lcpo.md"),
    "rma": ("B8 meta-RL / online system identification", "rma_osi.md"),
    "ernie": ("B9 robust MARL, adversarial regularisation", "ernie.md"),
}

#: rows that are a configuration or an environment flag rather than an
#: algorithm, and the "what was not done" pages.  Every one of them still owes
#: a checklist: a baseline with no doc is a baseline nobody can check.
DOC_ONLY = ["rnn.md", "gnn.md", "dr_sigma.md", "eso_dob.md", "rls_raw.md",
            "lilac.md", "skipped.md"]

failures: List[str] = []
checks = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global checks
    checks += 1
    if ok:
        print(f"[PASS] {label}" + (f"\n       {detail}" if detail else ""))
    else:
        print(f"[FAIL] {label}" + (f"\n       {detail}" if detail else ""))
        failures.append(label)


def load_module(path: Path, name: str):
    """Import a file directly, bypassing its package __init__.

    ``benchmarl.algorithms.__init__`` imports torchrl; these two modules do not.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# ===========================================================================
print("\n== syntax, on THIS interpreter ==")
# ===========================================================================
#  The cluster's python is not this laptop's.  Nothing here can be imported
#  without torchrl, but everything here can be PARSED, and a SyntaxError is the
#  one failure that costs a queue slot before a single frame is collected.
#  It has already happened once: a backslash inside an f-string expression is
#  Python 3.12 syntax (PEP 701) and the cluster runs older, so this file itself
#  would not parse there.

SOURCES = (
    sorted(ALGO_DIR.glob("*.py"))
    + sorted((ROOT / "benchmarl" / "environments" / "simple_ns").glob("*.py"))
    + [
        ROOT / "simple_ns" / "baselines.py",
        ROOT / "simple_ns" / "observer.py",
        ROOT / "simple_ns" / "layer.py",
        ROOT / "simple_ns" / "hosts.py",
        ROOT / "simple_ns" / "run.py",
    ]
)
bad = []
for _path in SOURCES:
    if not _path.exists():
        continue
    try:
        compile(_path.read_text(encoding="utf-8"), str(_path), "exec")
    except SyntaxError as err:
        bad.append("{}:{} {}".format(_path.name, err.lineno, err.msg))
check(
    "every baseline source parses on python {}.{}".format(*sys.version_info[:2]),
    not bad,
    "\n       ".join(bad) if bad else "{} files".format(len(SOURCES)),
)

#  benchmarl/__init__.py imports benchmarl.algorithms FIRST, and
#  benchmarl.experiment / benchmarl.environments both import names back OUT of
#  benchmarl.algorithms.  So a module under benchmarl/algorithms/ that imports
#  either of them at import time closes a cycle and `import benchmarl` dies
#  with "cannot import name 'IppoConfig' from partially initialized module".
#  It has already happened once.  mappo_ctde.py documents the same rule for
#  benchmarl.environments; the fix is always to import inside the function that
#  needs it.  This is invisible to a syntax check and to anything that cannot
#  import torchrl, so it is checked structurally.
CYCLE_PRONE = ("benchmarl.experiment", "benchmarl.environments")


def _import_time_nodes(body):
    """Statements that run when the module is imported.

    Recurses into ``if`` / ``try`` / ``class`` bodies -- all of which execute at
    import time -- but deliberately NOT into function bodies, which are exactly
    where a lazy import is supposed to live.
    """
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            yield node
        elif isinstance(node, (ast.If, ast.Try, ast.ClassDef)):
            yield from _import_time_nodes(node.body)
            yield from _import_time_nodes(getattr(node, "orelse", []))
            yield from _import_time_nodes(getattr(node, "finalbody", []))
            for handler in getattr(node, "handlers", []):
                yield from _import_time_nodes(handler.body)


cycles = []
for _path in sorted(ALGO_DIR.glob("*.py")):
    tree = ast.parse(_path.read_text(encoding="utf-8"))
    for node in _import_time_nodes(tree.body):
        names = (
            [node.module or ""]
            if isinstance(node, ast.ImportFrom)
            else [alias.name for alias in node.names]
        )
        for name in names:
            if any(name.startswith(prefix) for prefix in CYCLE_PRONE):
                cycles.append("{}:{} imports {}".format(_path.name, node.lineno, name))
check(
    "no algorithm imports benchmarl.experiment/environments at import time",
    not cycles,
    "\n       ".join(cycles) if cycles else "checked every module-level import",
)

#  The other half of the same rule.  `from benchmarl.algorithms import X` at
#  import time is an ATTRIBUTE lookup on a package that is still half-built, so
#  it only works when X is a SUBMODULE -- python falls back to importing it.
#  Ask for a class defined in __init__ and you get the same ImportError.
attr_imports = []
for _path in sorted(ALGO_DIR.glob("*.py")):
    tree = ast.parse(_path.read_text(encoding="utf-8"))
    for node in _import_time_nodes(tree.body):
        if isinstance(node, ast.ImportFrom) and node.module == "benchmarl.algorithms":
            for alias in node.names:
                if not (ALGO_DIR / "{}.py".format(alias.name)).exists():
                    attr_imports.append(
                        "{}:{} imports the name '{}', which is not a submodule".format(
                            _path.name, node.lineno, alias.name
                        )
                    )
check(
    "every `from benchmarl.algorithms import X` names a submodule",
    not attr_imports,
    "\n       ".join(attr_imports)
    if attr_imports
    else "the package is half-built at that point; only submodules resolve",
)

#  `Experiment.callbacks` is whatever it was constructed with, and the two
#  entry points differ: a direct Experiment(...) gets the [] default, while
#  load_experiment_from_hydra -- every launcher in this repo -- defaults to
#  `callbacks=()`.  So `.append` works in a notebook and is an AttributeError
#  on the cluster.  _compat.attach_callback handles both.
#  Matched on the AST, not on the text: the helper that exists to prevent this
#  naturally mentions `.callbacks.append` in its docstring, and a substring
#  search would flag the fix as the bug.
appends = []
for _path in sorted(ALGO_DIR.glob("*.py")):
    for node in ast.walk(ast.parse(_path.read_text(encoding="utf-8"))):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "append"
            and isinstance(node.func.value, ast.Attribute)
            and node.func.value.attr == "callbacks"
        ):
            appends.append("{}:{}".format(_path.name, node.lineno))
check(
    "no algorithm calls experiment.callbacks.append",
    not appends,
    "\n       ".join(appends)
    if appends
    else "use _compat.attach_callback: hydra hands Experiment a TUPLE",
)

#  The torchrl internals the baselines reach into moved between releases, and
#  the cluster's torchrl is older than setup.py's pin: `_has_critic` does not
#  exist before 0.8, `_log_weight` / `_get_entropy` grew an `adv_shape` argument
#  later still.  Every one of them is an AttributeError or TypeError one step
#  into the first optimizer loop -- after the queue wait, after setup -- so each
#  is reached through the shim in _compat.py, the one file allowed to name them.
TORCHRL_PRIVATE = ("_has_critic", "_log_weight", "_get_entropy", "_clip_bounds")
direct = []
for _path in sorted(ALGO_DIR.glob("*.py")):
    if _path.name == "_compat.py":
        continue
    for node in ast.walk(ast.parse(_path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Attribute) and node.attr in TORCHRL_PRIVATE:
            direct.append("{}:{} reads .{}".format(_path.name, node.lineno, node.attr))
check(
    "no algorithm reads a torchrl-private loss attribute directly",
    not direct,
    "\n       ".join(direct)
    if direct
    else "go through _compat: {}".format(", ".join(TORCHRL_PRIVATE)),
)

#  The constructor side of the same problem.  torchrl renamed `entropy_coef`
#  to `entropy_coeff` (0.9) and `critic_coef` to `critic_coeff` (0.10), and
#  PPOLoss.__init__ ends in a **kwargs it never checks, so the spelling the
#  installed torchrl does not know is silently DROPPED and the loss runs on
#  torchrl's defaults: entropy 0.01 where every yaml here says 0.0.  The
#  cluster's torchrl predates both renames; this code was written after them.
#  _compat.coefficient_kwargs reads the right name off the signature, and it
#  is the only call allowed to spell either coefficient as a keyword.
COEF_KEYWORDS = ("entropy_coef", "entropy_coeff", "critic_coef", "critic_coeff")
literal = []
for _path in sorted(ALGO_DIR.glob("*.py")):
    if _path.name == "_compat.py":
        continue
    for node in ast.walk(ast.parse(_path.read_text(encoding="utf-8"))):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Attribute) and func.attr == "coefficient_kwargs":
            continue
        for kw in node.keywords:
            if kw.arg in COEF_KEYWORDS:
                literal.append("{}:{} passes {}=".format(_path.name, kw.lineno, kw.arg))
check(
    "no algorithm passes a PPO coefficient to a loss by name",
    not literal,
    "\n       ".join(literal)
    if literal
    else "all seven ClipPPOLoss sites go through _compat.coefficient_kwargs",
)


# ===========================================================================
print("\n== registry and imports ==")
# ===========================================================================

init_src = (ALGO_DIR / "__init__.py").read_text(encoding="utf-8")
tree = ast.parse(init_src)
imported: Set[str] = set()
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        imported.update(alias.asname or alias.name for alias in node.names)

registry: Dict[str, str] = {}
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and getattr(
        node.targets[0], "id", None
    ) == "algorithm_config_registry":
        for key, value in zip(node.value.keys, node.value.values):
            registry[key.value] = value.id

check(
    "every baseline is in algorithm_config_registry",
    all(name in registry for name in BASELINES),
    f"missing: {sorted(set(BASELINES) - set(registry))}",
)
check(
    "every registry entry is imported in __init__",
    all(cls in imported for cls in registry.values()),
    f"missing: {sorted(c for c in registry.values() if c not in imported)}",
)
check(
    "every baseline has a conf/algorithm yaml",
    all((CONF_DIR / f"{name}.yaml").exists() for name in BASELINES),
    f"missing: {[n for n in BASELINES if not (CONF_DIR / f'{n}.yaml').exists()]}",
)
DOCS = ROOT / "baselines" / "docs"
wanted = [doc for _, doc in BASELINES.values()] + DOC_ONLY
check(
    "every baseline has a checklist in baselines/docs",
    all((DOCS / doc).exists() for doc in wanted),
    f"missing: {[d for d in wanted if not (DOCS / d).exists()]}",
)


# ===========================================================================
print("\n== config dataclasses against yaml ==")
# ===========================================================================


def dataclass_fields(module_src: str) -> Dict[str, Dict]:
    """``{ClassName: {"bases": [...], "fields": [...]}}`` for every dataclass."""
    out: Dict[str, Dict] = {}
    for node in ast.parse(module_src).body:
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(
            (isinstance(d, ast.Name) and d.id == "dataclass") for d in node.decorator_list
        ):
            continue
        fields = [
            item.target.id
            for item in node.body
            if isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name)
        ]
        bases = [b.id for b in node.bases if isinstance(b, ast.Name)]
        out[node.name] = {"bases": bases, "fields": fields}
    return out


def init_params(module_src: str, class_name: str):
    """``(named parameters, has **kwargs)`` of ``class_name.__init__``."""
    for node in ast.parse(module_src).body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    args = [a.arg for a in item.args.args if a.arg != "self"]
                    args += [a.arg for a in item.args.kwonlyargs]
                    return set(args), item.args.kwarg is not None
    return set(), False


all_dataclasses: Dict[str, Dict] = {}
sources: Dict[str, str] = {}
for path in sorted(ALGO_DIR.glob("*.py")):
    src = path.read_text(encoding="utf-8")
    sources[path.stem] = src
    all_dataclasses.update(dataclass_fields(src))


def resolved_fields(name: str) -> List[str]:
    info = all_dataclasses.get(name)
    if info is None:
        return []
    out: List[str] = []
    for base in info["bases"]:
        out += resolved_fields(base)
    for field in info["fields"]:
        if field not in out:
            out.append(field)
    return out


def yaml_keys(path: Path) -> List[str]:
    keys, in_defaults = [], False
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("defaults:"):
            in_defaults = True
            continue
        if in_defaults:
            if line.startswith(("  -", "  ")) or not line.strip():
                continue
            in_defaults = False
        match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*):", line)
        if match:
            keys.append(match.group(1))
    return keys


for name, (section, _doc) in BASELINES.items():
    yaml_path = CONF_DIR / f"{name}.yaml"
    if not yaml_path.exists():
        continue
    config_class = registry.get(name)
    fields = set(resolved_fields(config_class))
    keys = set(yaml_keys(yaml_path))
    check(
        f"{name}: yaml keys == {config_class} fields",
        fields == keys,
        f"yaml only: {sorted(keys - fields)}   dataclass only: {sorted(fields - keys)}",
    )

    first_default = None
    for line in yaml_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("- ") and first_default is None:
            first_default = stripped[2:]
            break
    check(
        f"{name}: hydra defaults entry is {name}_config",
        first_default == f"{name}_config",
        f"got {first_default!r}",
    )

    own = set(all_dataclasses[config_class]["fields"])
    algo_class = config_class[: -len("Config")]
    params, has_kwargs = init_params(sources[name], algo_class)
    check(
        f"{name}: {algo_class}.__init__ names every new config field",
        own <= params and has_kwargs,
        f"not named: {sorted(own - params)}   has **kwargs: {has_kwargs}",
    )


# ===========================================================================
print("\n== environment-side arms (B9, B10) ==")
# ===========================================================================

layer_src = (ROOT / "simple_ns" / "layer.py").read_text(encoding="utf-8")
hosts_src = (ROOT / "simple_ns" / "hosts.py").read_text(encoding="utf-8")


def tuple_keys(name: str, text: str) -> Set[str]:
    match = re.search(name + r"\s*=\s*\((.*?)\n\)", text, re.S)
    return set(re.findall(r'"([a-z0-9_]+)"', match.group(1))) if match else set()


ns_keys = tuple_keys("NS_KWARGS", layer_src)
baseline_keys = tuple_keys("BASELINE_KWARGS", layer_src)
new_env_keys = {
    "ns_observe_driver",
    "ns_observe_prev_action",
    "ns_dr_enabled",
    "ns_dr_low",
    "ns_dr_high",
    "ns_baseline",
}
check(
    "the new dial keys are popped by ExertionMixin",
    new_env_keys <= ns_keys,
    f"not popped: {sorted(new_env_keys - ns_keys)}",
)
check(
    "the B10 arm keys are popped by ExertionMixin",
    baseline_keys == {"eso_bandwidth", "rls_raw_use_operator"},
    f"got {sorted(baseline_keys)}",
)

arm_names = set(re.findall(r'"(eso|rls_raw)":', hosts_src))
#  Counted into locals first: a backslash inside an f-string expression is
#  Python 3.12 syntax (PEP 701) and this file has to parse on the cluster's
#  interpreter, which is older.  Same rule everywhere else in this tree.
n_eso = hosts_src.count('"eso":')
n_rls = hosts_src.count('"rls_raw":')
check(
    "hosts.py registers both B10 arms for every host",
    arm_names == {"eso", "rls_raw"} and n_eso == 4 and n_rls == 4,
    "arms {}, eso entries {}, rls_raw entries {}".format(
        sorted(arm_names), n_eso, n_rls
    ),
)

for host in ("balance", "transport", "sampling", "navigation"):
    task_yaml = (TASK_DIR / f"{host}.yaml").read_text(encoding="utf-8")
    missing = [key for key in sorted(new_env_keys | baseline_keys) if f"{key}:" not in task_yaml]
    check(f"{host}.yaml carries every new key", not missing, f"missing: {missing}")
    check(
        f"{host}.yaml leaves every new knob OFF",
        "ns_observe_driver: false" in task_yaml
        and "ns_observe_prev_action: false" in task_yaml
        and "ns_dr_enabled: false" in task_yaml
        and "ns_baseline: none" in task_yaml,
        "an existing arm would not be bit-identical",
    )


# ===========================================================================
print("\n== arithmetic ==")
# ===========================================================================

math_mod = load_module(ALGO_DIR / "_baseline_math.py", "_bl_math")
observer = load_module(ROOT / "simple_ns" / "observer.py", "_bl_observer")

# -- HAPPO -----------------------------------------------------------------
bounds = math_mod.block_bounds(675, 4)
sizes = [bounds[k + 1] - bounds[k] for k in range(4)]
check(
    "HAPPO: the optimiser budget splits evenly across agents",
    sum(sizes) == 675 and max(sizes) - min(sizes) <= 1 and min(sizes) > 0,
    f"675 calls, 4 agents -> blocks {sizes}",
)
sizes_7 = [
    math_mod.block_bounds(100, 7)[k + 1] - math_mod.block_bounds(100, 7)[k]
    for k in range(7)
]
check(
    "HAPPO: an indivisible budget still gives every agent a block",
    sum(sizes_7) == 100 and min(sizes_7) > 0,
    f"100 calls, 7 agents -> blocks {sizes_7}",
)

torch.manual_seed(0)
log_weight = torch.randn(32, 4, 1) * 0.1
mask = torch.tensor([1.0, 0.0, 1.0, 0.0])
factor = math_mod.happo_log_factor(log_weight, mask).exp()
brute = (log_weight[:, 0, 0].exp() * log_weight[:, 2, 0].exp()).reshape(32, 1, 1)
check(
    "HAPPO: the factor is the product of the updated agents' ratios",
    torch.allclose(factor, brute, atol=1e-6),
    f"max |diff| = {float((factor - brute).abs().max()):.2e} against the "
    "explicit product over agents {0, 2}",
)
check(
    "HAPPO: the factor is exactly 1 for the first agent in the order",
    torch.allclose(
        math_mod.happo_log_factor(log_weight, torch.zeros(4)).exp(),
        torch.ones(32, 1, 1),
    ),
    "M starts at 1, as `factor = np.ones(...)` in HARL's runner",
)

#  The identity the implementation rests on: scaling the advantage by M > 0 is
#  the same as scaling HARL's `min(surr1, surr2)` by M.
adv = torch.randn(32, 4, 1)
ratio = torch.rand(32, 4, 1) + 0.5
clipped = ratio.clamp(0.8, 1.2)
M = torch.rand(32, 1, 1) + 0.5
lhs = torch.stack([ratio * (adv * M), clipped * (adv * M)], -1).min(-1).values
rhs = M * torch.stack([ratio * adv, clipped * adv], -1).min(-1).values
check(
    "HAPPO: min(r*M*A, clip*M*A) == M * min(r*A, clip*A) for M > 0",
    torch.allclose(lhs, rhs, atol=1e-6),
    "so multiplying the advantage by the factor IS HARL's "
    "`factor_batch * torch.min(surr1, surr2)`",
)

# -- ESO / DOB -------------------------------------------------------------
for w in (0.1, 0.3, 0.7):
    beta1, beta2 = observer.eso_gains(w)
    #  the observer's characteristic polynomial is
    #  z^2 - (2 - beta1) z + (1 - beta1 + beta2); both roots must be 1 - w
    b = -(2 - beta1)
    c = 1 - beta1 + beta2
    root_sum, root_product = -b, c
    check(
        f"ESO: bandwidth w={w} places both observer poles at 1-w={1 - w:.2f}",
        math.isclose(root_sum, 2 * (1 - w), rel_tol=1e-9)
        and math.isclose(root_product, (1 - w) ** 2, rel_tol=1e-9),
        f"beta1={beta1:.4f} beta2={beta2:.4f} -> pole sum {root_sum:.6f} "
        f"(want {2 * (1 - w):.6f}), product {root_product:.6f} "
        f"(want {(1 - w) ** 2:.6f})",
    )

z1 = torch.zeros(1)
z2 = torch.zeros(1)
beta1, beta2 = observer.eso_gains(0.3)
target = torch.tensor([0.25])
for _ in range(200):
    z1, z2, pred = observer.eso_step(z1, z2, target, beta1, beta2)
check(
    "ESO: converges to a constant disturbance with zero steady-state error",
    torch.allclose(z1, target, atol=1e-4) and z2.abs().item() < 1e-4,
    f"after 200 steps on a constant 0.25: z1={float(z1):.6f} z2={float(z2):.2e}",
)

z1 = torch.zeros(1)
z2 = torch.zeros(1)
errors = []
for t in range(300):
    y = torch.tensor([0.001 * t])
    z1, z2, pred = observer.eso_step(z1, z2, y, beta1, beta2)
    errors.append(float(pred - 0.001 * (t + 1)))
check(
    "ESO: the second state removes the lag on a ramp",
    abs(errors[-1]) < 1e-3,
    f"one-step-ahead error on a unit-slope ramp settles at {errors[-1]:.2e}; a "
    "first-order filter would settle at a constant offset",
)

# -- LIAM ------------------------------------------------------------------
n_agents = 4
values = torch.arange(n_agents * 2, dtype=torch.float32).reshape(1, n_agents, 2)
others = math_mod.others_view(values, n_agents)
expected_row0 = torch.tensor([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
check(
    "LIAM: the decoder target is every agent but the modelling one",
    others.shape == (1, n_agents, (n_agents - 1) * 2)
    and torch.allclose(others[0, 0], expected_row0),
    f"row 0 = {others[0, 0].tolist()} (agent 0's own value 0,1 is absent)",
)
own = values.reshape(n_agents, 2)
diag_leak = any(
    torch.isclose(others[0, i].reshape(n_agents - 1, 2), own[i]).all(-1).any()
    and not any(
        torch.isclose(own[j], own[i]).all() for j in range(n_agents) if j != i
    )
    for i in range(n_agents)
)
check(
    "LIAM: no agent appears in its own target row",
    not diag_leak,
    "the zero diagonal is structural, as it is everywhere else in this repo",
)

# -- MF-AC -----------------------------------------------------------------
actions = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]])
team = math_mod.mean_action(actions, include_self=True)
peers = math_mod.mean_action(actions, include_self=False)
check(
    "MF-AC: include_self=True is the reference's team mean",
    torch.allclose(team, torch.full((1, 4, 1), 2.5)),
    f"mean of 1,2,3,4 = 2.5 for every agent -> {team.reshape(-1).tolist()}",
)
check(
    "MF-AC: include_self=False is the paper's mean over N(j)",
    torch.allclose(peers.reshape(-1), torch.tensor([3.0, 8 / 3, 7 / 3, 2.0])),
    f"agent 0 sees mean(2,3,4)=3 -> {[round(v, 4) for v in peers.reshape(-1).tolist()]}",
)
check(
    "MF-AC: a lone agent has no peers to average",
    torch.allclose(
        math_mod.mean_action(torch.ones(1, 1, 1), include_self=False), torch.zeros(1, 1, 1)
    ),
    "N=1 gives exactly zero, so the mean-field channel is category-C clean too",
)

# -- LCPO ------------------------------------------------------------------
torch.manual_seed(1)
dim = 12
A = torch.randn(dim, dim)
A = A @ A.T + dim * torch.eye(dim)  # symmetric positive definite
b = torch.randn(dim)
x = math_mod.conjugate_gradients(lambda v: A @ v, b, dim)
check(
    "LCPO: conjugate gradients solves A x = b in at most dim iterations",
    torch.allclose(A @ x, b, atol=1e-4),
    f"residual {float((A @ x - b).norm()):.2e} after {dim} iterations",
)

roots = math_mod.get_qu(1.0, -3.0, 2.0)
check(
    "LCPO: get_qu returns the roots of the dual's quadratic",
    math.isclose(max(roots), 2.0, rel_tol=1e-9)
    and math.isclose(min(roots), 1.0, rel_tol=1e-9),
    f"s^2 - 3s + 2 -> {roots}",
)

loc = torch.zeros(1, 3)
scale = torch.ones(1, 3)
check(
    "LCPO: the Gaussian KL is zero between identical policies",
    float(math_mod.gaussian_kl(loc, scale, loc, scale)) == 0.0,
    "the trust region is inactive when nothing moved",
)
kl_known = math_mod.gaussian_kl(
    torch.zeros(1, 1), torch.ones(1, 1), torch.ones(1, 1), torch.ones(1, 1)
)
check(
    "LCPO: KL(N(0,1) || N(1,1)) == 0.5",
    math.isclose(float(kl_known), 0.5, rel_tol=1e-6),
    f"got {float(kl_known):.6f}",
)
probs_old = torch.tensor([[0.5, 0.5]])
probs_new = torch.tensor([[0.25, 0.75]])
expected_kl = 0.5 * math.log(0.5 / 0.25) + 0.5 * math.log(0.5 / 0.75)
check(
    "LCPO: the categorical KL matches its definition",
    math.isclose(float(math_mod.categorical_kl(probs_old, probs_new)), expected_kl, rel_tol=1e-5),
    f"got {float(math_mod.categorical_kl(probs_old, probs_new)):.6f}, want {expected_kl:.6f}",
)

generator = torch.Generator().manual_seed(0)
sampler = math_mod.OutOfDistributionSampler(
    n_agents=2,
    obs_dim=3,
    window=8,
    capacity=64,
    context_slice=slice(2, 3),
    threshold=0.25,
    device=torch.device("cpu"),
    generator=generator,
)
check(
    "LCPO: an empty reservoir returns nothing, so the A2C branch is taken",
    sampler.get(4) is None,
    "`OutOfDSampler.get` returns [] before anything has been stored",
)
near = torch.zeros(32, 2, 3)
sampler.add(near)
check(
    "LCPO: states from the CURRENT context are not out of distribution",
    sampler.get(4) is None,
    "everything stored has the same context as the recent window",
)
far = torch.zeros(32, 2, 3)
far[..., 2] = 5.0
sampler.add(far)
recent_is_far = torch.zeros(32, 2, 3)
recent_is_far[..., 2] = 5.0
batch = sampler.get(4)
check(
    "LCPO: once the context moves, the old states become out of distribution",
    batch is not None and batch.shape == (4, 2, 3),
    f"got {None if batch is None else tuple(batch.shape)} after the window "
    "filled with context 5.0 and the reservoir still holds context 0.0",
)
if batch is not None:
    check(
        "LCPO: every returned state really is distant",
        bool((batch[..., 2] != 5.0).all()),
        "no state from the current context leaked into the OOD batch",
    )


# ===========================================================================
print("\n" + "-" * 78)
if failures:
    print(f"{checks - len(failures)}/{checks} checks passed; FAILED: {failures}")
    sys.exit(1)
print(f"{checks}/{checks} checks passed")
