#  Entry point for the road_traffic drift tasks.  P-10.1: every arm launches
#  through THIS, with severity supplied from outside the method, so the
#  baselines are bit-for-bit the ones BenchMARL ships.
#
#      python road_ns/run.py algorithm=ippo task=road_ns/road_traffic
#
#  Identical to benchmarl/run.py except that it guarantees the BenchMARL being
#  imported is *this* checkout: `python benchmarl/run.py` puts benchmarl/ on
#  sys.path[0] rather than the repo root, so `import benchmarl` can resolve to
#  an installed copy while hydra reads yaml from here.  That failure reads like
#  a config error and is an import-path error.

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) in sys.path:
    sys.path.remove(str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT))

import hydra  # noqa: E402
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from omegaconf import DictConfig, OmegaConf  # noqa: E402

import benchmarl  # noqa: E402,F401  (registers the hydra schemas)
from benchmarl.hydra_config import load_experiment_from_hydra  # noqa: E402

_TASK = "road_ns/road_traffic"


def _check_import_is_this_checkout() -> None:
    imported = Path(benchmarl.__file__).resolve().parent
    expected = _REPO_ROOT / "benchmarl"
    if imported != expected:
        raise RuntimeError(
            f"benchmarl was imported from {imported}, not {expected}.\n"
            "Configs would come from this checkout while the code came from "
            "elsewhere.  Run `pip install -e .` from the repo root, or unset "
            "any conflicting PYTHONPATH."
        )
    from benchmarl.environments import task_config_registry

    if _TASK not in task_config_registry:
        raise RuntimeError(
            f"{_TASK} is not registered.  Check that "
            "benchmarl/environments/__init__.py lists RoadNsTask in `tasks`."
        )


@hydra.main(version_base=None, config_path="conf", config_name="config")
def hydra_experiment(cfg: DictConfig) -> None:
    choices = HydraConfig.get().runtime.choices
    print(f"\nAlgorithm: {choices.algorithm}, Task: {choices.task}")
    print(f"benchmarl: {Path(benchmarl.__file__).resolve().parent}")
    print("\nLoaded config:\n")
    print(OmegaConf.to_yaml(cfg))
    load_experiment_from_hydra(cfg, task_name=choices.task).run()


if __name__ == "__main__":
    _check_import_is_this_checkout()
    hydra_experiment()
