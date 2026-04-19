from src.scenarios.base import ScenarioGenerator
from src.scenarios.imbalance_direction.markov_chain import MarkovChainModel


class ImbalanceDirectionGenerator(ScenarioGenerator):
    """Scenario generator for imbalance direction (up/down regulation)."""

    _models = {
        MarkovChainModel.name: MarkovChainModel,
    }

    _default_model_paths = {
        "markov_chain": "trained_models/imbalance_direction/markov_chain.npy",
    }
