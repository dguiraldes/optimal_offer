from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.scenarios.base import ScenarioModel, N_STEPS, RESOLUTION_MIN

# Lags in 15-min steps: short-term (15–60 min), daily (96 = 24 h),
# two-day (192 = 48 h), weekly (672 = 168 h).
DEFAULT_LAGS = [1, 2, 3, 4, 96, 192, 672]

_DK_TZ = ZoneInfo("Europe/Copenhagen")
_MINUTES_PER_DAY = 24 * 60


def _time_features_from_timestamps(timestamps_utc: pd.DatetimeIndex) -> np.ndarray:
    """Return (N, 2) array of [sin, cos] time-of-day in Danish time."""
    idx = pd.DatetimeIndex(timestamps_utc)
    if idx.tz is None:
        idx = idx.tz_localize("UTC")
    dk = idx.tz_convert(_DK_TZ)
    frac = (dk.hour * 60 + dk.minute) / _MINUTES_PER_DAY
    angle = 2 * np.pi * frac
    return np.column_stack([np.sin(angle), np.cos(angle)])


def _time_features_from_utc_minutes(minutes_since_midnight_utc: np.ndarray,
                                     utc_offset_hours: float) -> np.ndarray:
    """Return (N, 2) array of [sin, cos] given UTC minutes and offset."""
    dk_minutes = (minutes_since_midnight_utc + utc_offset_hours * 60) % _MINUTES_PER_DAY
    angle = 2 * np.pi * dk_minutes / _MINUTES_PER_DAY
    return np.column_stack([np.sin(angle), np.cos(angle)])


class ARModel(ScenarioModel):
    """
    Autoregressive model for spot price scenario generation.

    Fits an AR model with configurable lags on 15-min spot prices,
    augmented with sin/cos time-of-day features in Danish local time,
    and generates quarter-hourly scenarios directly.

    Parameters
    ----------
    model_path : str or Path, optional
        Path to a saved ``.npz`` file with fitted parameters.
    ar_lags : list[int], optional
        Lags (in 15-min steps) to include in the AR model.
        Defaults to ``[1, 2, 3, 4, 96, 192, 672]``.
    """

    name = "ar"

    def __init__(
        self,
        model_path: str | Path | None = None,
        ar_lags: list[int] | None = None,
        **kwargs,
    ):
        self.ar_lags = ar_lags or DEFAULT_LAGS
        self.ar_coefficients: np.ndarray | None = None
        self.time_coefficients: np.ndarray | None = None  # [sin, cos]
        self.intercept: float | None = None
        self.residual_std: float | None = None

        if model_path is not None:
            self._load(model_path)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        prices_15min: np.ndarray,
        timestamps_utc: pd.DatetimeIndex,
    ) -> ARModel:
        """
        Fit the AR model on 15-min spot prices via OLS.

        Parameters
        ----------
        prices_15min : array-like, shape (N,)
            Quarter-hourly spot prices (EUR/MWh).
        timestamps_utc : DatetimeIndex, shape (N,)
            UTC timestamps corresponding to each price observation.

        Returns
        -------
        self
        """
        prices = np.asarray(prices_15min, dtype=float)
        max_lag = max(self.ar_lags)
        n = len(prices)

        if n <= max_lag:
            raise ValueError(
                f"Need more than {max_lag} observations, got {n}."
            )

        y = prices[max_lag:]

        # AR lag features
        X_lags = np.column_stack(
            [prices[max_lag - lag : n - lag] for lag in self.ar_lags]
        )

        # Time-of-day features (sin/cos in Danish time)
        X_time = _time_features_from_timestamps(timestamps_utc[max_lag:])

        X = np.column_stack([np.ones(len(y)), X_lags, X_time])

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        self.intercept = float(beta[0])
        n_lags = len(self.ar_lags)
        self.ar_coefficients = beta[1 : 1 + n_lags]
        self.time_coefficients = beta[1 + n_lags :]  # [sin, cos]

        residuals = y - X @ beta
        self.residual_std = float(np.std(residuals))

        return self

    # ------------------------------------------------------------------
    # ScenarioModel interface
    # ------------------------------------------------------------------

    @property
    def required_inputs(self) -> list[str]:
        return ["recent_prices", "cutoff_utc"]

    def generate(
        self,
        n_scenarios: int,
        seed: int | None = None,
        **inputs,
    ) -> np.ndarray:
        """
        Generate spot price scenarios for the next delivery day.

        Parameters
        ----------
        n_scenarios : int
            Number of scenario paths.
        seed : int or None
            Random seed for reproducibility.
        recent_prices : array-like, shape (>= max_lag,)
            Quarter-hourly spot prices ending at the last known
            step (e.g. 10:45 when the cut-off is 11:00 UTC).
        cutoff_utc : str, datetime, or pd.Timestamp
            UTC timestamp of the last known price.
        n_bridge : int, optional
            Number of 15-min steps between the last known price
            and midnight of the delivery day.  Default ``52``
            (cut-off at 11:00 UTC → 13 h × 4).

        Returns
        -------
        np.ndarray, shape (n_scenarios, 96)
        """
        if self.ar_coefficients is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        recent = np.asarray(inputs["recent_prices"], dtype=float)
        cutoff = pd.Timestamp(inputs["cutoff_utc"])
        n_bridge = int(inputs.get("n_bridge", 52))
        max_lag = max(self.ar_lags)

        if len(recent) < max_lag:
            raise ValueError(
                f"Need at least {max_lag} recent quarter-hourly prices, "
                f"got {len(recent)}."
            )

        n_forecast = n_bridge + N_STEPS

        # Pre-compute time-of-day features for all forecast steps
        forecast_times = pd.date_range(
            start=cutoff + pd.Timedelta(minutes=RESOLUTION_MIN),
            periods=n_forecast,
            freq=f"{RESOLUTION_MIN}min",
        )
        time_feats = _time_features_from_timestamps(forecast_times)

        rng = np.random.default_rng(seed)
        scenarios = np.empty((n_scenarios, N_STEPS))

        for s in range(n_scenarios):
            buf = list(recent[-max_lag:])
            for t in range(n_forecast):
                x_ar = np.array([buf[-lag] for lag in self.ar_lags])
                y_hat = (
                    self.intercept
                    + self.ar_coefficients @ x_ar
                    + self.time_coefficients @ time_feats[t]
                )
                y = y_hat + rng.normal(0, self.residual_std)
                buf.append(y)
            scenarios[s] = buf[-N_STEPS:]

        return scenarios

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save fitted parameters to a ``.npz`` file."""
        if self.ar_coefficients is None:
            raise RuntimeError("Nothing to save – model not fitted.")
        np.savez(
            path,
            ar_lags=np.array(self.ar_lags),
            ar_coefficients=self.ar_coefficients,
            time_coefficients=self.time_coefficients,
            intercept=np.array([self.intercept]),
            residual_std=np.array([self.residual_std]),
        )

    def _load(self, path: str | Path) -> None:
        data = np.load(path)
        self.ar_lags = data["ar_lags"].tolist()
        self.ar_coefficients = data["ar_coefficients"]
        self.time_coefficients = data["time_coefficients"]
        self.intercept = float(data["intercept"][0])
        self.residual_std = float(data["residual_std"][0])

    @classmethod
    def load(cls, path: str | Path) -> ARModel:
        """Load a fitted model from a ``.npz`` file."""
        return cls(model_path=path)
