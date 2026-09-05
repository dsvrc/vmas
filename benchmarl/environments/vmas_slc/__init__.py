#  VMAS tasks under Shared Link Contention (category-C), and PACT on top.
#
#  The submodules ``slc_core`` and ``pact_core`` are torch-only by design: they
#  must import on a machine with no simulator and no RL stack, because the
#  arithmetic self-check and the Part-C ceiling decomposition run there.  So the
#  torchrl-dependent wiring below is exposed lazily -- importing
#  ``benchmarl.environments.vmas_slc.slc_core`` must not drag torchrl in.

_LAZY = {
    "SlcDialError": "common",
    "VmasSlcClass": "common",
    "VmasSlcTask": "common",
}

__all__ = list(_LAZY)


def __getattr__(name):
    if name in _LAZY:
        import importlib

        module = importlib.import_module(f"{__name__}.{_LAZY[name]}")
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
