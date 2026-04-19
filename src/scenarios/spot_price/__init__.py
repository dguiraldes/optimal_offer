from src.scenarios.base import ScenarioGenerator
from src.scenarios.spot_price.ar_model import ARModel
from src.scenarios.spot_price.arimax import ArimaxModel


class SpotPriceGenerator(ScenarioGenerator):
    """Scenario generator for day-ahead spot prices.

    Available models
    ----------------
    - ``"ar"`` – Autoregressive model with time-of-day features.

    Required inputs (model ``"ar"``)
    --------------------------------
    recent_prices : array-like, shape (>= 672,)
        Quarter-hourly (15-min) spot prices in EUR/MWh, ending at the
        last known time step (e.g. 10:45 UTC if the cut-off is 11:00).
        Must contain at least 672 values (= 7 days) to cover the
        largest AR lag.
    cutoff_utc : str, datetime, or pd.Timestamp
        UTC timestamp of the last known price observation.  Used to
        compute the sin/cos time-of-day features for the forecast
        horizon.
    n_bridge : int, optional
        Number of 15-min steps between ``cutoff_utc`` and midnight of
        the delivery day. Default ``52`` (11:00 UTC → 00:00 next day).

    Examples
    --------
    >>> gen = SpotPriceGenerator(model="ar")
    >>> gen.required_inputs
    ['recent_prices', 'cutoff_utc']
    >>>
    >>> # recent_prices: last 672+ quarter-hourly spot prices
    >>> scenarios = gen.generate(
    ...     n_scenarios=200,
    ...     seed=42,
    ...     recent_prices=recent_prices,
    ...     cutoff_utc="2026-03-15 10:45",
    ... )
    >>> scenarios.shape
    (200, 96)
    """

    _models = {
        ArimaxModel.name: ArimaxModel,
        ARModel.name: ARModel,
    }

    _default_model_paths = {
        "ar": "trained_models/spot_price/ar.npz",
    }
