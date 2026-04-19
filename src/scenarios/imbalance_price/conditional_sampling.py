import numpy as np

from src.scenarios.base import ScenarioModel, N_STEPS


class ConditionalSamplingModel(ScenarioModel):
    """Conditional sampling of imbalance price given direction (placeholder)."""

    name = "conditional_sampling"

    @property
    def required_inputs(self) -> list[str]:
        return ["imbalance_direction", "spot_price"]

    def generate(self, n_scenarios: int, seed: int | None = None, **inputs) -> np.ndarray:
        # TODO: sample imbalance prices conditioned on direction & spot
        raise NotImplementedError
