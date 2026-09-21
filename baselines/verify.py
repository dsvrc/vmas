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
    # ---- the EXTRA baselines; see baselines/README_EXTRA.md -------------
    "qcd": ("X1 prior-free NS-RL: detect and restart", "qcd.md"),
    "dedafp": ("X2 deep fictitious play for continuous MFGs", "dedafp.md"),
    "ipga": ("X3 independent learning in performative MPGs", "ipga.md"),
    "wisdom": ("X4 wavelet predictive representations", "wisdom.md"),
    "doraemon": ("X5 domain randomisation by entropy maximisation", "doraemon.md"),
    "m3w": ("X6 MoE world model, with planning", "m3w.md"),
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
        ROOT / "simple_ns" / "dr_state.py",
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
print("\n== torchrl, if it is installed ==")
# ===========================================================================
#  Every loss here subclasses a torchrl loss and calls into methods torchrl
#  marks private, and those names MOVE between releases.  The cluster's `vmas`
#  env is torchrl 0.7.x while setup.py pins >=0.10, so the two sides disagree
#  about, among others, `qvalue_v2_loss` (0.11) vs `_qvalue_v2_loss` (0.7) and
#  `entropy_coeff` (0.9+) vs `entropy_coef`.  `_compat` shims each one; this
#  section checks the shims can actually find something on the torchrl that is
#  installed HERE.  It is skipped, loudly, when torchrl is absent.

try:
    import torchrl
    from torchrl.objectives import ClipPPOLoss, SACLoss
except Exception as _err:  # noqa: BLE001 -- any import failure means "skip"
    print(
        "[SKIP] torchrl is not importable here ({}: {}), so its API cannot be "
        "checked.\n       Run this on the machine that will train -- these are "
        "the checks that\n       catch a version mismatch before it costs a "
        "queue slot.".format(type(_err).__name__, _err)
    )
else:
    import torch as _torch

    print(
        "       torchrl {}, torch {}".format(torchrl.__version__, _torch.__version__)
    )
    compat = load_module(ALGO_DIR / "_compat.py", "_bl_compat")

    #  Alias groups: at least one spelling of each must exist.
    for cls, groups in (
        (
            ClipPPOLoss,
            [
                ("_log_weight",),
                ("_get_entropy",),
                ("_clip_bounds",),
                ("loss_critic",),
            ],
        ),
        (
            SACLoss,
            [
                ("qvalue_v2_loss", "_qvalue_v2_loss"),
                ("_compute_target_v2",),
                ("_alpha_loss",),
                ("_alpha",),
            ],
        ),
    ):
        absent = [g for g in groups if not any(hasattr(cls, n) for n in g)]
        check(
            "{}: every internal the baselines reach into exists".format(cls.__name__),
            not absent,
            "missing every spelling of: {}".format(absent)
            if absent
            else "checked {} name groups".format(len(groups)),
        )

    #  The constructor coefficient names, which is the one that failed SILENTLY:
    #  before the shim, `entropy_coeff=` went into PPOLoss's unchecked **kwargs
    #  on 0.7.x and the configured value was simply dropped.
    try:
        coeffs = compat.coefficient_kwargs(ClipPPOLoss, 0.0, 1.0)
    except Exception as err:  # noqa: BLE001
        coeffs, coeff_err = {}, err
    else:
        coeff_err = None
    check(
        "the PPO coefficient kwargs resolve to names this torchrl accepts",
        coeff_err is None and len(coeffs) == 2,
        "resolved {}".format(sorted(coeffs))
        if coeff_err is None
        else "{}: {}".format(type(coeff_err).__name__, coeff_err),
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
#: added for the EXTRA baselines, and inert at these defaults
extra_env_keys = {
    "ns_observe_prev_reward",   # X4 WISDOM, X6 M3W
    "ns_observe_time",          # X2 DEDA-FP
    "ns_time_horizon",
    "ns_dr_dist",               # X5 DORAEMON
    "ns_dr_a",
    "ns_dr_b",
}
check(
    "the new dial keys are popped by ExertionMixin",
    new_env_keys <= ns_keys,
    f"not popped: {sorted(new_env_keys - ns_keys)}",
)
check(
    "the EXTRA baselines' dial keys are popped by ExertionMixin",
    extra_env_keys <= ns_keys,
    f"not popped: {sorted(extra_env_keys - ns_keys)}",
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
    missing = [
        key
        for key in sorted(new_env_keys | baseline_keys | extra_env_keys)
        if f"{key}:" not in task_yaml
    ]
    check(f"{host}.yaml carries every new key", not missing, f"missing: {missing}")
    check(
        f"{host}.yaml leaves every new knob OFF",
        "ns_observe_driver: false" in task_yaml
        and "ns_observe_prev_action: false" in task_yaml
        and "ns_dr_enabled: false" in task_yaml
        and "ns_baseline: none" in task_yaml
        and "ns_observe_prev_reward: false" in task_yaml
        and "ns_observe_time: false" in task_yaml
        and "ns_dr_dist: uniform" in task_yaml,
        "an existing arm would not be bit-identical",
    )
    task_py = (
        ROOT / "benchmarl" / "environments" / "simple_ns" / f"{host}.py"
    ).read_text(encoding="utf-8")
    absent = [key for key in sorted(extra_env_keys) if f"{key}:" not in task_py]
    check(
        f"{host} TaskConfig declares every EXTRA key",
        not absent,
        f"missing from the dataclass (hydra would refuse the yaml): {absent}",
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
print("\n== arithmetic: the EXTRA baselines (X1 .. X6) ==")
# ===========================================================================

# -- X1  QCD+ / RR ---------------------------------------------------------
#  Theorem 4 is the paper's headline and it is a two-line inequality, so it is
#  checked against the number the paper quotes rather than restated.
_min_T = math_mod.master_min_horizon()
check(
    "QCD+: MASTER's tests cannot fire below T ~ 1.24e9 (2410.13772, Thm 4)",
    1.20e9 < _min_T < 1.30e9
    and not math_mod.master_can_fire(1e9)
    and math_mod.master_can_fire(1e10),
    f"smallest horizon at which sqrt(T) > 54 (log2 T + 1) log T is {_min_T:.4e}; "
    "the paper says 'T must be at least 1.24 billion'",
)
check(
    "QCD+: at this repo's horizons MASTER would declare ZERO changes",
    not math_mod.master_can_fire(3_000_000 // 300),
    "a 3M-frame run at 300 workers is 10000 detector rounds, which is seven "
    "orders of magnitude below the threshold -- which is why the arm that runs "
    "is quickest change detection and not MASTER",
)

_flat = torch.full((200,), 0.5)
_step = torch.cat([torch.full((100,), 0.2), torch.full((100,), 0.8)])
_stat_flat = float(math_mod.glr_statistic(_flat))
_stat_step = float(math_mod.glr_statistic(_step))
_thresh = math_mod.glr_threshold(200, 0.01)
check(
    "QCD+: the Bernoulli GLR separates a step change from a flat stream",
    _stat_flat < _thresh < _stat_step,
    f"flat {_stat_flat:.3f} < threshold {_thresh:.3f} < step change "
    f"{_stat_step:.3f}",
)
check(
    "QCD+: the GLR threshold is log(4 n sqrt(n) / delta)",
    math.isclose(
        math_mod.glr_threshold(64, 0.05),
        math.log(4 * 64 * 8 / 0.05),
        rel_tol=1e-12,
    ),
    "Besson et al. (JMLR 2022) Theorem 1, which 2410.13772 Algorithm 3 cites",
)
check(
    "QCD+: kl(p, p) = 0 and kl is finite at the boundary",
    float(math_mod.bernoulli_kl(torch.tensor(0.3), torch.tensor(0.3))) == 0.0
    and torch.isfinite(
        math_mod.bernoulli_kl(torch.tensor(0.0), torch.tensor(1.0))
    ).all(),
    "a segment of all-zeros must not fire the detector with an infinite "
    "statistic on its first sample",
)

_gen = torch.Generator().manual_seed(0)
_schedule = math_mod.RandomRestartSchedule(eta=0.01, generator=_gen)
_gaps, _last = [], 0
for _t in range(1, 200001):
    if _schedule.step():
        _gaps.append(_t - _last)
        _last = _t
_mean_gap = sum(_gaps) / max(len(_gaps), 1)
check(
    "RR: the restart gaps are Geometric(eta), mean 1/eta",
    90.0 < _mean_gap < 110.0 and _schedule.n_restarts == len(_gaps),
    f"eta=0.01 over 200k rounds gave {len(_gaps)} restarts, mean gap "
    f"{_mean_gap:.1f} (want ~100)",
)

# -- X2  DEDA-FP -----------------------------------------------------------
torch.manual_seed(0)
_x = torch.randn(9, 6)
_shift, _log_scale = torch.randn(9, 6), torch.randn(9, 6) * 0.3
_mask = torch.tensor([1.0, 1.0, 1.0, 0.0, 0.0, 0.0])
_y, _log_det = math_mod.affine_coupling(_x, _shift, _log_scale, _mask)
_back = math_mod.affine_coupling_inverse(_y, _shift, _log_scale, _mask)
check(
    "DEDA-FP: the coupling layer is exactly invertible",
    torch.allclose(_x, _back, atol=1e-5),
    f"max |x - f^-1(f(x))| = {float((_x - _back).abs().max()):.2e}",
)
_jac = torch.autograd.functional.jacobian(
    lambda v: math_mod.affine_coupling(v, _shift[0], _log_scale[0], _mask)[0],
    _x[0],
)
check(
    "DEDA-FP: the reported log-determinant IS the Jacobian's",
    math.isclose(
        float(_log_det[0]),
        float(torch.logdet(_jac).abs()),
        rel_tol=1e-4,
    ),
    f"reported {float(_log_det[0]):.6f}, autograd Jacobian "
    f"{float(torch.logdet(_jac).abs()):.6f} -- if these disagree the flow's "
    "density is not a density and the fit is not maximum likelihood",
)
check(
    "DEDA-FP: the base log-density integrates to 1 in one dimension",
    math.isclose(
        float(
            torch.trapz(
                math_mod.standard_normal_log_prob(
                    torch.linspace(-12, 12, 200001).unsqueeze(-1)
                ).exp(),
                torch.linspace(-12, 12, 200001),
            )
        ),
        1.0,
        rel_tol=1e-6,
    ),
    "the flow's base measure",
)
_loc, _scale = torch.zeros(4, 2), torch.ones(4, 2)
check(
    "DEDA-FP: the Gaussian NLL is the negative log-density",
    math.isclose(
        float(math_mod.gaussian_nll(_loc, _scale, torch.zeros(4, 2))[0]),
        2 * 0.5 * math.log(2 * math.pi),
        rel_tol=1e-6,
    ),
    "L_NLL at the mean of a unit Gaussian in 2 dimensions",
)

# -- X3  IPGA / INPG -------------------------------------------------------
_p_loc, _p_scale = torch.tensor([0.3]), torch.tensor([0.8])
_q_loc, _q_scale = torch.tensor([-0.4]), torch.tensor([1.1])
_closed = float(math_mod.gaussian_l2_sq(_p_loc, _p_scale, _q_loc, _q_scale))
_grid = torch.linspace(-40, 40, 400001)
_p = torch.exp(-0.5 * ((_grid - _p_loc) / _p_scale) ** 2) / (
    _p_scale * math.sqrt(2 * math.pi)
)
_q = torch.exp(-0.5 * ((_grid - _q_loc) / _q_scale) ** 2) / (
    _q_scale * math.sqrt(2 * math.pi)
)
_numeric = float(torch.trapz((_p - _q) ** 2, _grid))
check(
    "IPGA: the closed-form L2 distance between two Gaussians is the integral",
    math.isclose(_closed, _numeric, rel_tol=1e-5),
    f"closed form {_closed:.8f}, numerical quadrature {_numeric:.8f} -- this "
    "is the proximal term the paper writes as ||pi - pi'||_2^2",
)
check(
    "IPGA: the proximal term is zero between identical policies",
    float(math_mod.gaussian_l2_sq(_p_loc, _p_scale, _p_loc, _p_scale)) < 1e-12
    and float(
        math_mod.categorical_l2_sq(
            torch.tensor([0.25, 0.75]), torch.tensor([0.25, 0.75])
        )
    )
    == 0.0,
    "so the update reduces to plain policy gradient when nothing has moved",
)
_probs = torch.tensor([[0.5, 0.5]])
_adv = torch.tensor([[1.0, 0.0]])
_new = math_mod.inpg_multiplicative_update(_probs, _adv, eta=0.1, gamma=0.9)
_w = math.exp(0.1 / (1 - 0.9))
check(
    "INPG: the multiplicative update matches its closed form",
    torch.allclose(
        _new, torch.tensor([[_w / (_w + 1), 1.0 / (_w + 1)]]), atol=1e-6
    )
    and math.isclose(float(_new.sum()), 1.0, rel_tol=1e-6),
    f"pi prop-to pi exp(eta/(1-gamma) A) with eta=0.1, gamma=0.9 -> "
    f"{[round(v, 6) for v in _new.reshape(-1).tolist()]}",
)

# -- X4  WISDOM ------------------------------------------------------------
_mus = torch.tensor([[[1.0], [3.0]]])
_vars = torch.tensor([[[1.0], [1.0]]])
_mu, _var = math_mod.product_of_gaussians(_mus, _vars)
check(
    "WISDOM: the product of N(1,1) and N(3,1) is N(2, 1/2)",
    torch.allclose(_mu, torch.tensor([[2.0]]))
    and torch.allclose(_var, torch.tensor([[0.5]])),
    f"got mu={float(_mu):.4f} var={float(_var):.4f} -- PEARL's "
    "permutation-invariant posterior over the context window",
)
_single_mu = torch.randn(3, 1, 4)
_single_var = torch.rand(3, 1, 4) + 0.5
_out_mu, _out_var = math_mod.product_of_gaussians(_single_mu, _single_var)
check(
    "WISDOM: a one-element context gives back that element's Gaussian",
    torch.allclose(_out_mu, _single_mu.squeeze(-2), atol=1e-5)
    and torch.allclose(_out_var, _single_var.squeeze(-2), atol=1e-5),
    "the product over an empty rest is the identity",
)
check(
    "WISDOM: KL(N(0,1) || N(0,1)) is zero and KL(N(mu,1) || N(0,1)) is mu^2/2",
    float(
        math_mod.gaussian_kl_to_standard_normal(torch.zeros(1, 1), torch.ones(1, 1))
    )
    == 0.0
    and math.isclose(
        float(
            math_mod.gaussian_kl_to_standard_normal(
                torch.full((1, 1), 2.0), torch.ones(1, 1)
            )
        ),
        2.0,
        rel_tol=1e-6,
    ),
    "the encoder's only objective in the released tree; see docs/wisdom.md",
)

#  The wavelet is a LEARNED filter bank, so what can be checked is the
#  structure: with a delta low-pass and a zero high-pass, every approximation
#  band is the input and the mixing weights are all that is left.
_delta_h0 = torch.zeros(1, 1, 2)
_delta_h0[0, 0, -1] = 1.0
_zero_h1 = torch.zeros(1, 1, 2)
_w_only_lo = torch.zeros(1, 4)
_w_only_lo[0, 0] = 1.0            # weight on the FINAL approximation band
_signal = torch.randn(5, 1, 8)
_y, _res = math_mod.wavelet_forward_fading(
    _signal, _delta_h0, _zero_h1, _w_only_lo, depth=2, kernel_size=2
)
check(
    "WISDOM: with an identity low-pass the approximation band IS the signal",
    torch.allclose(_res, _signal, atol=1e-6)
    and torch.allclose(_y, _signal, atol=1e-6),
    "so forward_fading's padding, dilation and band bookkeeping line up",
)
_w_only_in = torch.zeros(1, 4)
_w_only_in[0, -1] = 1.0           # weight on the INPUT skip connection
_y_skip, _ = math_mod.wavelet_forward_fading(
    _signal, torch.zeros(1, 1, 2), _zero_h1, _w_only_in, depth=2, kernel_size=2
)
check(
    "WISDOM: the last mixing weight is the input's own skip connection",
    torch.allclose(_y_skip, _signal, atol=1e-6),
    "w[:, -1] multiplies x itself, as in the reference's `y += x * w[:, -1:]`",
)
#  The TD operator's fixed point is the discounted future of the latent: with
#  res_lo = identity, iterating the target must converge to sum_k gamma^k z.
_z = torch.ones(1, 3)
_res_lo = torch.zeros(1, 3)
for _ in range(400):
    _res_lo = math_mod.wavelet_td_target(_z, _res_lo, 0.9)
check(
    "WISDOM: the wavelet TD operator's fixed point is sum_k gamma^k z",
    torch.allclose(_res_lo, torch.full((1, 3), 10.0), atol=1e-4),
    f"iterating z + 0.9 * res_lo from 0 converges to {float(_res_lo[0, 0]):.4f} "
    "(want 1/(1-0.9) = 10) -- this is what makes the representation PREDICTIVE",
)

# -- X5  DORAEMON ----------------------------------------------------------
check(
    "DORAEMON: the maximum-entropy Beta on [low, high] is the uniform",
    math.isclose(
        float(math_mod.beta_entropy(1.0, 1.0, 0.0, 3.0)), math.log(3.0), rel_tol=1e-9
    )
    and float(math_mod.beta_entropy(100.0, 100.0, 0.0, 3.0))
    < float(math_mod.beta_entropy(1.0, 1.0, 0.0, 3.0)),
    f"H[Beta(1,1) on [0,3]] = {float(math_mod.beta_entropy(1.0, 1.0, 0.0, 3.0)):.6f} "
    f"= log 3; H[Beta(100,100)] = "
    f"{float(math_mod.beta_entropy(100.0, 100.0, 0.0, 3.0)):.6f}. Driving the "
    "distribution to Beta(1,1) IS maximising entropy.",
)
#  THE ONE THAT MATTERS.  In float32 this KL evaluates to -4e-5 -- negative,
#  for a divergence -- and a constraint function with that much noise makes the
#  trust-region solver's steps meaningless: measured, DORAEMON's distribution
#  moved by a KL of 2e-4 against a bound of 5e-2 and never widened.
_tiny = float(math_mod.beta_kl(100.0, 100.0, 99.994, 99.994))
check(
    "DORAEMON: the Beta KL is computed in double precision and is non-negative",
    0.0 < _tiny < 1e-8 and float(math_mod.beta_kl(3.0, 5.0, 3.0, 5.0)) == 0.0,
    f"KL(Beta(100,100) || Beta(99.994,99.994)) = {_tiny:.3e}; in float32 the "
    "same expression is -4.2e-05, and the outer loop stops moving",
)
_a = torch.tensor(4.0, dtype=torch.float64, requires_grad=True)
_kl = math_mod.beta_kl(_a, 4.0, 1.0, 1.0)
(_grad,) = torch.autograd.grad(_kl, _a)
check(
    "DORAEMON: the KL is differentiable in the Beta parameters",
    torch.isfinite(_grad).all() and float(_grad) != 0.0,
    "the reference supplies analytic jacobians for the objective and BOTH "
    "constraints; finite differences of a 1e-9 KL are noise",
)
#  The importance-sampling identity the performance constraint rests on:
#  E_p[f] estimated from samples of q, with weights p/q.
_gen = torch.Generator().manual_seed(3)
_samples = (
    torch.distributions.Beta(torch.tensor(2.0), torch.tensor(5.0)).sample((200000,))
    * 3.0
).double()
#  The shift is a SMALL one, because that is the only regime the estimator is
#  used in: DORAEMON's trust region exists precisely to keep the candidate
#  close enough to the sampling distribution for the reweighting to be sound.
_proposed = (2.5, 4.5)
_weights = math_mod.importance_ratio(_samples, _proposed, (2.0, 5.0), 0.0, 3.0)
_is_estimate = float(
    math_mod.doraemon_success_rate(_samples, 1.5, weights=_weights)
)
_direct = float(
    math_mod.doraemon_success_rate(
        (
            torch.distributions.Beta(
                torch.tensor(_proposed[0]), torch.tensor(_proposed[1])
            ).sample((200000,))
            * 3.0
        ).double(),
        1.5,
    )
)
check(
    "DORAEMON: the importance-sampled success rate matches direct sampling",
    abs(_is_estimate - _direct) < 0.01,
    f"P[sigma >= 1.5] under Beta{_proposed} is {_direct:.4f} directly and "
    f"{_is_estimate:.4f} reweighted from Beta(2,5) samples -- which is what "
    "lets a candidate distribution be tested without collecting any episodes",
)
check(
    "DORAEMON: the importance weights average to 1",
    abs(float(_weights.mean()) - 1.0) < 0.01,
    f"mean weight {float(_weights.mean()):.5f}; a reweighting that does not "
    "integrate to 1 biases every constraint evaluation",
)
_bounds = math_mod.sigmoid_bounds(
    math_mod.inv_sigmoid_bounds(torch.tensor([0.9, 37.0]), 0.8, 110.0), 0.8, 110.0
)
check(
    "DORAEMON: the sigmoid parameterisation round-trips",
    torch.allclose(_bounds, torch.tensor([0.9, 37.0], dtype=torch.float64), atol=1e-6),
    f"got {[round(float(v), 6) for v in _bounds]} -- the optimiser is "
    "unconstrained while a and b stay inside their bounds",
)

# -- X6  M3W ---------------------------------------------------------------
_x = torch.randn(7, 24)
_normed = math_mod.simnorm(_x, 8)
check(
    "M3W: SimNorm makes every group of 8 coordinates a simplex",
    torch.allclose(
        _normed.reshape(7, 3, 8).sum(-1), torch.ones(7, 3), atol=1e-6
    )
    and bool((_normed >= 0).all()),
    "which is what bounds the latent so a planning rollout cannot diverge",
)
_bins = torch.linspace(-10.0, 10.0, 101)
_values = torch.tensor([[0.7], [-3.2], [0.0], [55.0]])
_encoded = math_mod.two_hot_encode(_values, _bins, -10.0, 10.0)
_decoded = math_mod.two_hot_decode(torch.log(_encoded.clamp_min(1e-12)), _bins)
check(
    "M3W: the two-hot encoding is a distribution and round-trips exactly",
    torch.allclose(_encoded.sum(-1), torch.ones(4), atol=1e-6)
    and torch.allclose(_decoded, _values, atol=1e-3),
    f"{[round(float(v), 4) for v in _decoded.reshape(-1)]} from "
    f"{[float(v) for v in _values.reshape(-1)]}",
)
check(
    "M3W: symlog and symexp are inverses, including on the negative side",
    torch.allclose(
        math_mod.sym_exp(math_mod.sym_log(torch.tensor([-400.0, -1.0, 0.0, 7.5]))),
        torch.tensor([-400.0, -1.0, 0.0, 7.5]),
        atol=1e-3,
    ),
    "the two-hot bins live on the symlog scale, so [-10, 10] covers |r| up to e^10",
)
_r = torch.tensor([[1.0, 1.0, 1.0, 1.0]])
_ret, _tail = math_mod.nstep_return(_r, 0.5)
_ret_done, _tail_done = math_mod.nstep_return(
    _r, 0.5, torch.tensor([[0.0, 1.0, 0.0, 0.0]])
)
check(
    "M3W: the n-step return is the discounted sum, and a done truncates it",
    math.isclose(float(_ret), 1.875)
    and math.isclose(float(_tail), 0.0625)
    and math.isclose(float(_ret_done), 1.5)
    and float(_tail_done) == 0.0,
    "1 + .5 + .25 + .125 = 1.875 with a 0.5^4 bootstrap; terminating after "
    "the second reward gives 1.5 and no bootstrap",
)
#  The SoftMoE routing: with one expert whose map is the identity, every token
#  must come back as a convex combination of the tokens -- nothing is dropped.
_tokens = torch.randn(3, 4, 5)
_out_uniform = math_mod.soft_moe(_tokens, torch.zeros(5, 1, 1), [lambda v: v])
check(
    "M3W: SoftMoE with a flat router and an identity expert is the token mean",
    _out_uniform.shape == _tokens.shape
    and torch.allclose(
        _out_uniform, _tokens.mean(1, keepdim=True).expand_as(_tokens), atol=1e-5
    ),
    "phi = 0 makes the dispatch softmax uniform, so the single slot holds the "
    "plain average and every token reads it back",
)
_out_routed = math_mod.soft_moe(_tokens, torch.randn(5, 1, 1), [lambda v: v])
check(
    "M3W: one slot means every token reads back the SAME value, and no token "
    "is dropped",
    torch.allclose(
        _out_routed, _out_routed[:, :1].expand_as(_out_routed), atol=1e-5
    )
    and bool(torch.isfinite(_out_routed).all()),
    "a routed dispatch is a different convex combination of the tokens, but "
    "still a convex combination -- which is the property that separates SoftMoE "
    "from a top-k router",
)
check(
    "M3W: the router's balance term is zero at a perfectly even load",
    float(math_mod.cv_squared(torch.full((8,), 3.0))) < 1e-9
    and float(math_mod.cv_squared(torch.tensor([0.0, 0.0, 0.0, 12.0]))) > 1.0,
    "cv_squared is the variance over the squared mean: 0 when every expert "
    "carries the same mass, large when one carries all of it",
)
_elite_values = torch.tensor([[0.0, 10.0]])
_elite_actions = torch.tensor([[[[-1.0], [1.0]]]])   # (horizon=1, batch=1, K=2, d=1)
_mean, _std, _score = math_mod.mppi_update(
    _elite_values, _elite_actions, temperature=10.0, min_std=0.0, max_std=1.0
)
check(
    "M3W: MPPI's moment update concentrates on the best elite",
    math.isclose(float(_score.sum()), 1.0, rel_tol=1e-6)
    and float(_mean[0, 0, 0]) > 0.99
    and float(_std[0, 0, 0]) < 0.05,
    f"two elites worth 0 and 10 at temperature 10 give mean "
    f"{float(_mean[0, 0, 0]):.4f} and std {float(_std[0, 0, 0]):.4f}",
)
_flat_values = torch.zeros(1, 4)
_flat_actions = torch.arange(4, dtype=torch.float32).reshape(1, 1, 4, 1)
_mean_flat, _, _score_flat = math_mod.mppi_update(
    _flat_values, _flat_actions, temperature=1.0, min_std=0.0, max_std=1.0
)
check(
    "M3W: equal elite values give the plain mean",
    torch.allclose(_score_flat, torch.full((1, 4), 0.25), atol=1e-6)
    and math.isclose(float(_mean_flat[0, 0, 0]), 1.5, rel_tol=1e-6),
    "mean of 0,1,2,3 is 1.5",
)

# ===========================================================================
print("\n== scipy, which DORAEMON needs ==")
# ===========================================================================
try:
    from scipy.optimize import minimize as _minimize, NonlinearConstraint as _NLC
except Exception as _err:  # noqa: BLE001
    check(
        "scipy.optimize is importable (algorithm=doraemon needs it)",
        False,
        f"{type(_err).__name__}: {_err}. DORAEMON solves its constrained "
        "problem with trust-constr, which is what the reference uses. Every "
        "other row is unaffected.",
    )
else:
    import scipy as _scipy

    _probe = _minimize(
        lambda v: (float((v[0] - 2.0) ** 2), __import__("numpy").array([2 * (v[0] - 2.0)])),
        __import__("numpy").array([0.0]),
        method="trust-constr",
        jac=True,
        constraints=[_NLC(fun=lambda v: v[0], lb=-1e18, ub=1.0, jac=lambda v: __import__("numpy").array([1.0]))],
        options={"gtol": 1e-8, "xtol": 1e-10, "maxiter": 200},
    )
    check(
        "scipy's trust-constr solves a constrained problem with analytic jacobians",
        bool(_probe.success) and abs(float(_probe.x[0]) - 1.0) < 1e-3,
        f"scipy {_scipy.__version__}: min (x-2)^2 s.t. x <= 1 gave x = "
        f"{float(_probe.x[0]):.6f} (want 1.0)",
    )


# ===========================================================================
print("\n" + "-" * 78)
if failures:
    print(f"{checks - len(failures)}/{checks} checks passed; FAILED: {failures}")
    sys.exit(1)
print(f"{checks}/{checks} checks passed")
