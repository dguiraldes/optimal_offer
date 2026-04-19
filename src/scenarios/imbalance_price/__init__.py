from src.scenarios.base import ScenarioGenerator
from src.scenarios.imbalance_price.conditional_sampling import ConditionalSamplingModel
from src.scenarios.imbalance_price.spread_model import SpreadModel


class ImbalancePriceGenerator(ScenarioGenerator):
    """Scenario generator for imbalance prices.

    Available models
    ----------------
    ``"spread"``
        Log-spread regression conditioned on spot price, direction,
        time-of-day (sin/cos), and a 1-step autoregressive lag.

        Required inputs for ``generate()``:

        - **spot_price** : ``(n_scenarios, 96)`` — spot price scenarios.
        - **imbalance_direction** : ``(n_scenarios, 96)`` — direction
          scenarios with values in {-1, 0, 1}.
        - **last_spread** : ``float`` — last observed
          ``imbalance_price − spot_price``.
        - **delivery_date** : ``str`` or ``date`` — delivery date.

    ``"conditional_sampling"``
        Placeholder (not yet implemented).
    """

    _models = {
        SpreadModel.name: SpreadModel,
        ConditionalSamplingModel.name: ConditionalSamplingModel,
    }

    _default_model_paths = {
        "spread": "trained_models/imbalance_price/spread.npz",
    }
