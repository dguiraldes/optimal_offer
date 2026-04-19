from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

# Fixed prediction window: 24 h at 15-min resolution
HORIZON_HOURS = 24
RESOLUTION_MIN = 15
N_STEPS = HORIZON_HOURS * 60 // RESOLUTION_MIN  # 96

# Project root (two levels above src/scenarios/base.py)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]


class ScenarioModel(ABC):
    """
    A single modelling method for one resource.

    Each concrete model (e.g. ARIMAX for spot price) inherits from this
    and implements `generate`.  Models should be pre-trained / self-contained.
    """

    name: str  # short identifier, e.g. "arimax"

    @property
    @abstractmethod
    def required_inputs(self) -> list[str]:
        """Names of the keyword arguments that `generate` expects."""
        ...

    @abstractmethod
    def generate(self, n_scenarios: int, seed: int | None = None, **inputs) -> np.ndarray:
        """
        Produce scenario paths.

        Returns
        -------
        np.ndarray of shape (n_scenarios, N_STEPS)
        """
        ...

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r})"


class ScenarioGenerator:
    """
    Front-end for a resource's scenario models.

    Subclasses only need to populate the `_models` dict mapping
    model name -> ScenarioModel subclass, and optionally
    `_default_model_paths` mapping model name -> relative path
    (from project root) to a pre-trained checkpoint.

    Usage
    -----
    >>> gen = WindProductionGenerator(model="conditional_nvp")
    >>> gen.required_inputs
    ['wind_forecast']
    >>> scenarios = gen.generate(n_scenarios=200, wind_forecast=...)
    """

    _models: dict[str, type[ScenarioModel]] = {}
    _default_model_paths: dict[str, str] = {}

    def __init__(self, model: str, model_path: str | Path | None = None, **model_kwargs):
        if model not in self._models:
            available = list(self._models.keys())
            raise ValueError(
                f"Unknown model '{model}'. Available: {available}"
            )
        if model_path is None and model in self._default_model_paths:
            model_path = _PROJECT_ROOT / self._default_model_paths[model]
        if model_path is not None:
            model_kwargs["model_path"] = model_path
        self._model = self._models[model](**model_kwargs)

    @classmethod
    def available_models(cls) -> list[str]:
        """Return the names of all registered models."""
        return list(cls._models.keys())

    @property
    def required_inputs(self) -> list[str]:
        """Inputs required by the selected model's `generate` method."""
        return self._model.required_inputs

    def generate(self, n_scenarios: int, seed: int | None = None, **inputs) -> np.ndarray:
        """Generate scenario paths by delegating to the selected model.

        Parameters
        ----------
        n_scenarios : int
            Number of independent scenario paths to produce.
        seed : int or None
            Random seed for reproducibility.
        **inputs
            Model-specific keyword arguments.  Check
            ``self.required_inputs`` for the names expected by the
            current model, and the generator/model class docstring
            for detailed descriptions and examples.

        Returns
        -------
        np.ndarray of shape (n_scenarios, N_STEPS)
            Each row is one scenario path with ``N_STEPS = 96``
            quarter-hourly values covering a 24-hour delivery day.
        """
        return self._model.generate(n_scenarios, seed=seed, **inputs)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self._model.name!r})"
