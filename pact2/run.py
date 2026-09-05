#  Entry point for the Shared Link Contention tasks.
#
#      python pact2/run.py algorithm=ippo task=vmas_slc/sampling \
#             experiment.render=false experiment.checkpoint_at_end=true
#
#  Identical to benchmarl/run.py except that it guarantees the BenchMARL being
#  imported is *this* checkout.
#
#  Why it exists: `python benchmarl/run.py` puts `benchmarl/` on sys.path[0],
#  not the repo root, so `import benchmarl` resolves to whatever is installed in
#  site-packages while hydra reads yaml from the local `benchmarl/conf`.  When
#  those are different copies the task yaml is found but its ConfigStore schema
#  is not, and hydra fails with
#
#      In 'task/vmas_slc/sampling': Could not load
#      'task/vmas_slc_sampling_config'
#
#  which reads like a config bug and is actually an import-path bug.

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

import hydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import benchmarl  # noqa: E402,F401  (import registers the hydra schemas)
from benchmarl.hydra_config import load_experiment_from_hydra  # noqa: E402

_TASKS = ("vmas_slc/sampling", "vmas_slc/discovery", "vmas_slc/navigation")


def _check_import_is_this_checkout() -> None:
    imported = Path(benchmarl.__file__).resolve().parent
    expected = _REPO_ROOT / "benchmarl"
    if imported != expected:
        raise RuntimeError(
            f"benchmarl was imported from {imported}, not {expected}.\n"
            "Configs would be read from this checkout while the code came from "
            "elsewhere.  Run `pip install -e .` from the repo root, or unset any "
            "conflicting PYTHONPATH."
        )
    from benchmarl.environments import task_config_registry

    missing = [t for t in _TASKS if t not in task_config_registry]
    if missing:
        raise RuntimeError(
            f"{missing} not registered.  Check that "
            "benchmarl/environments/__init__.py lists VmasSlcTask in `tasks`."
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    hydra_choices = HydraConfig.get().runtime.choices
    task_name = hydra_choices.task
    algorithm_name = hydra_choices.algorithm

    print(f"\nAlgorithm: {algorithm_name}, Task: {task_name}")
    print(f"benchmarl: {Path(benchmarl.__file__).resolve().parent}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))

    experiment = load_experiment_from_hydra(cfg, task_name=task_name)
    experiment.run()


if __name__ == "__main__":
    _check_import_is_this_checkout()
    hydra_experiment()
