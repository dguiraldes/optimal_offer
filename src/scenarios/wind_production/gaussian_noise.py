import numpy as np

from src.scenarios.base import ScenarioModel, N_STEPS


class GaussianNoiseModel(ScenarioModel):
    """Gaussian noise around wind power forecast (placeholder)."""

    name = "gaussian_noise"

    @property
    def required_inputs(self) -> list[str]:
        return ["wind_forecast"]

    def generate(self, n_scenarios: int, seed: int | None = None, **inputs) -> np.ndarray:
        # TODO: add Gaussian perturbations around forecast
        raise NotImplementedError
