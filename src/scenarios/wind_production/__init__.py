from src.scenarios.base import ScenarioGenerator
from src.scenarios.wind_production.conditional_nvp import NormalizingFlowModel
# from src.scenarios.wind_production.gaussian_noise import GaussianNoiseModel


class WindProductionGenerator(ScenarioGenerator):
    """Scenario generator for offshore wind production."""

    _models = {
        NormalizingFlowModel.name: NormalizingFlowModel,
       # GaussianNoiseModel.name: GaussianNoiseModel,
    }

    _default_model_paths = {
        "conditional_nvp": "trained_models/wind_generation/conditional_nvp.pt",
    }
