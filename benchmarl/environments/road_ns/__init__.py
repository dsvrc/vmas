#  road_traffic under Coupling-Under-Drift, plus PACT-1.
#
#  Exposed lazily: road_ns/ and pact1/ are torch-only by design so the
#  conformance suite, the ceiling decomposition and the estimator self-test run
#  on a laptop.  Importing this package must not drag torchrl in.

_LAZY = {"RoadNsClass": "common", "RoadNsTask": "common", "InertLayerError": "common"}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        return getattr(importlib.import_module(f"{__name__}.{_LAZY[name]}"), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
