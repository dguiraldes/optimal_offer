from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.scenarios.base import ScenarioModel, N_STEPS, RESOLUTION_MIN

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


class SpreadModel(ScenarioModel):
    """
    Imbalance price model via log-spread regression.

    Models the spread between imbalance price and spot price as:

        imbalance_price = spot_price + direction × abs_spread

    where ``abs_spread = exp(ŷ) − 1`` and ``ŷ`` is predicted by OLS on:

        [intercept, spot_price, direction, sin_tod, cos_tod, lag1_log_spread]

    Constraints enforced at generation time:

    - direction =  0 → imbalance_price  = spot_price
    - direction =  1 → imbalance_price ≥ spot_price
    - direction = -1 → imbalance_price ≤ spot_price

    Parameters
    ----------
    model_path : str or Path, optional
        Path to a saved ``.npz`` file with fitted parameters.
    """

    name = "spread"

    def __init__(self, model_path: str | Path | None = None, **kwargs):
        self.coefficients: np.ndarray | None = None
        self.residual_std: float | None = None
        self.log_spread_cap: float | None = None

        if model_path is not None:
            self._load(model_path)

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(
        self,
        imbalance_prices: np.ndarray,
        spot_prices: np.ndarray,
        directions: np.ndarray,
        timestamps_utc: pd.DatetimeIndex,
    ) -> SpreadModel:
        """
        Fit the spread model via OLS on log-transformed absolute spread.

        Only observations with non-zero direction are used for fitting.
        The target is ``log(|imbalance_price − spot_price| + 1)``.

        Parameters
        ----------
        imbalance_prices : array, shape (N,)
            Historical imbalance prices (EUR/MWh) at 15-min resolution.
        spot_prices : array, shape (N,)
            Corresponding spot prices (EUR/MWh).
        directions : array, shape (N,)
            Dominating direction at each step (-1, 0, 1).
        timestamps_utc : DatetimeIndex, shape (N,)
            UTC timestamps for each observation.

        Returns
        -------
        self
        """
        imb = np.asarray(imbalance_prices, dtype=float)
        spot = np.asarray(spot_prices, dtype=float)
        dirs = np.asarray(directions, dtype=float)
        n = len(imb)

        # Compute log-absolute-spread for every step.
        # For direction=0 steps, enforce spread=0 (used only for lag).
        spread = imb - spot
        model_spread = np.where(dirs != 0, spread, 0.0)
        log_abs_spread = np.log(np.abs(model_spread) + 1.0)

        # Keep only non-zero-direction rows that have a lag available.
        valid_mask = (dirs != 0) & (np.arange(n) > 0)
        idx = np.where(valid_mask)[0]

        y = log_abs_spread[idx]

        X = np.column_stack([
            np.ones(len(idx)),                                      # intercept
            spot[idx],                                               # spot price
            dirs[idx],                                               # direction ±1
            _time_features_from_timestamps(timestamps_utc[idx]),     # sin, cos
            log_abs_spread[idx - 1],                                 # lag-1
        ])

        beta = np.linalg.lstsq(X, y, rcond=None)[0]
        self.coefficients = beta

        residuals = y - X @ beta
        self.residual_std = float(np.std(residuals))

        # Cap for generation: 99.9th percentile of training log-spread
        self.log_spread_cap = float(np.percentile(y, 99.9))

        return self

    # ------------------------------------------------------------------
    # ScenarioModel interface
    # ------------------------------------------------------------------

    @property
    def required_inputs(self) -> list[str]:
        return ["spot_price", "imbalance_direction", "last_spread", "delivery_date"]

    def generate(
        self,
        n_scenarios: int,
        seed: int | None = None,
        **inputs,
    ) -> np.ndarray:
        """
        Generate imbalance price scenarios for a delivery day.

        Parameters
        ----------
        n_scenarios : int
            Number of scenario paths to produce.
        seed : int or None
            Random seed for reproducibility.
        spot_price : np.ndarray, shape (n_scenarios, N_STEPS)
            Spot price scenarios for the delivery day.
        imbalance_direction : np.ndarray, shape (n_scenarios, N_STEPS)
            Direction scenarios for the delivery day (values in {-1, 0, 1}).
        last_spread : float
            Last observed spread (imbalance_price − spot_price) before
            the delivery day, used to initialise the lag-1 feature.
        delivery_date : str or date
            Delivery date (used to compute sin/cos time-of-day features).

        Returns
        -------
        np.ndarray, shape (n_scenarios, N_STEPS)
        """
        if self.coefficients is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")

        spot = np.asarray(inputs["spot_price"], dtype=float)
        dirs = np.asarray(inputs["imbalance_direction"], dtype=float)
        last_spread = float(inputs["last_spread"])
        delivery_date = pd.Timestamp(inputs["delivery_date"])

        # Pre-compute time-of-day features for the 96 delivery steps.
        delivery_start = pd.Timestamp(delivery_date).tz_localize("UTC")
        timestamps = pd.date_range(
            start=delivery_start,
            periods=N_STEPS,
            freq=f"{RESOLUTION_MIN}min",
        )
        time_feats = _time_features_from_timestamps(timestamps)  # (96, 2)

        rng = np.random.default_rng(seed)
        result = np.empty((n_scenarios, N_STEPS))

        for s in range(n_scenarios):
            log_spread_prev = np.log(np.abs(last_spread) + 1.0)

            for t in range(N_STEPS):
                d = dirs[s, t]

                if d == 0:
                    result[s, t] = spot[s, t]
                    log_spread_prev = 0.0
                else:
                    x = np.array([
                        1.0,
                        spot[s, t],
                        d,
                        time_feats[t, 0],
                        time_feats[t, 1],
                        log_spread_prev,
                    ])
                    y_hat = (self.coefficients @ x
                             + rng.normal(0, self.residual_std))
                    y_hat = min(y_hat, self.log_spread_cap)
                    abs_spread = max(0.0, np.exp(y_hat) - 1.0)
                    result[s, t] = spot[s, t] + d * abs_spread
                    log_spread_prev = np.log(abs_spread + 1.0)

        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Save fitted parameters to a ``.npz`` file."""
        if self.coefficients is None:
            raise RuntimeError("Nothing to save – model not fitted.")
        np.savez(
            path,
            coefficients=self.coefficients,
            residual_std=np.array([self.residual_std]),
            log_spread_cap=np.array([self.log_spread_cap]),
        )

    def _load(self, path: str | Path) -> None:
        data = np.load(path)
        self.coefficients = data["coefficients"]
        self.residual_std = float(data["residual_std"][0])
        self.log_spread_cap = float(data["log_spread_cap"][0])

    @classmethod
    def load(cls, path: str | Path) -> SpreadModel:
        """Load a fitted model from a ``.npz`` file."""
        return cls(model_path=path)
