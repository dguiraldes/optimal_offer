import numpy as np

from src.scenarios.base import ScenarioModel, N_STEPS


class ArimaxModel(ScenarioModel):
    """ARIMAX-based spot price scenario generator (placeholder)."""

    name = "arimax"

    @property
    def required_inputs(self) -> list[str]:
        return ["day_ahead_forecast"]

    def generate(self, n_scenarios: int, seed: int | None = None, **inputs) -> np.ndarray:
        # TODO: implement ARIMAX sampling around day-ahead forecast
        raise NotImplementedError
