"""Forest backend selection for BBEATSx.

BBEATSx is written against the ``stochtree`` low-level API.  This module exposes
exactly those primitives, importing the compiled ``stochtree`` package when it is
available and otherwise falling back to the pure-numpy reference implementation in
:mod:`bbeatsx.backend._numpy_backend` (see that module's docstring for the
caveats).

``BACKEND`` is ``"stochtree"`` or ``"numpy-reference"`` and is surfaced in fitted
models so experiments record which sampler actually produced the draws.
"""

from __future__ import annotations

import os

_FORCE = os.environ.get("BBEATSX_BACKEND", "").strip().lower()

Dataset = Residual = RNG = None  # type: ignore
GlobalModelConfig = ForestModelConfig = None  # type: ignore
Forest = ForestContainer = ForestSampler = None  # type: ignore
GlobalVarianceModel = LeafVarianceModel = None  # type: ignore
BACKEND = None  # type: ignore


def _load_stochtree() -> bool:
    global Dataset, Residual, RNG, GlobalModelConfig, ForestModelConfig
    global Forest, ForestContainer, ForestSampler
    global GlobalVarianceModel, LeafVarianceModel, BACKEND
    try:
        from stochtree import (  # noqa: F401
            Dataset as _Dataset,
            Residual as _Residual,
            RNG as _RNG,
            GlobalModelConfig as _GlobalModelConfig,
            ForestModelConfig as _ForestModelConfig,
            Forest as _Forest,
            ForestContainer as _ForestContainer,
            ForestSampler as _ForestSampler,
            GlobalVarianceModel as _GlobalVarianceModel,
            LeafVarianceModel as _LeafVarianceModel,
        )
    except Exception:
        return False
    Dataset, Residual, RNG = _Dataset, _Residual, _RNG
    GlobalModelConfig, ForestModelConfig = _GlobalModelConfig, _ForestModelConfig
    Forest, ForestContainer, ForestSampler = _Forest, _ForestContainer, _ForestSampler
    GlobalVarianceModel, LeafVarianceModel = _GlobalVarianceModel, _LeafVarianceModel
    BACKEND = "stochtree"
    return True


def _load_numpy() -> None:
    global Dataset, Residual, RNG, GlobalModelConfig, ForestModelConfig
    global Forest, ForestContainer, ForestSampler
    global GlobalVarianceModel, LeafVarianceModel, BACKEND
    from . import _numpy_backend as nb
    Dataset, Residual, RNG = nb.Dataset, nb.Residual, nb.RNG
    GlobalModelConfig, ForestModelConfig = nb.GlobalModelConfig, nb.ForestModelConfig
    Forest, ForestContainer, ForestSampler = nb.Forest, nb.ForestContainer, nb.ForestSampler
    GlobalVarianceModel, LeafVarianceModel = nb.GlobalVarianceModel, nb.LeafVarianceModel
    BACKEND = nb.BACKEND_NAME


if _FORCE == "numpy":
    _load_numpy()
elif _FORCE == "stochtree":
    if not _load_stochtree():
        raise ImportError("BBEATSX_BACKEND=stochtree but stochtree is not importable")
else:
    if not _load_stochtree():
        _load_numpy()


__all__ = [
    "Dataset", "Residual", "RNG", "GlobalModelConfig", "ForestModelConfig",
    "Forest", "ForestContainer", "ForestSampler",
    "GlobalVarianceModel", "LeafVarianceModel", "BACKEND",
]
