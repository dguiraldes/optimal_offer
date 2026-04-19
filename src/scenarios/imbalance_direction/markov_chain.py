from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.scenarios.base import ScenarioModel, N_STEPS, RESOLUTION_MIN

# Imbalance direction states: down-regulation, balanced, up-regulation
STATES = np.array([-1, 0, 1])
_STATE_IX = {s: i for i, s in enumerate(STATES)}  # state -> row/col index


class MarkovChainModel(ScenarioModel):
    """
    Time-of-day-dependent Markov chain for imbalance direction.

    The model estimates one 3×3 transition matrix per hour of the day
    (24 matrices total).  Each row gives P(next_state | current_state)
    for that hour.

    Parameters
    ----------
    transition_matrices : np.ndarray, shape (24, 3, 3), optional
        Pre-computed transition matrices.  If not supplied the model
        must be fitted with :meth:`fit` before calling :meth:`generate`.
    """

    name = "markov_chain"

    def __init__(
        self,
        model_path: str | Path | None = None,
        transition_matrices: np.ndarray | None = None,
        **kwargs,
    ):
        if model_path is not None:
            transition_matrices = np.load(model_path)
        if transition_matrices is not None:
            self._validate_matrices(transition_matrices)
            self.transition_matrices = transition_matrices
        else:
            self.transition_matrices = None

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit(self, df: pd.DataFrame) -> "MarkovChainModel":
        """
        Estimate hourly transition matrices from historical data.

        Parameters
        ----------
        df : DataFrame
            Must contain columns ``"TimeUTC"`` (or a datetime index)
            and ``"DominatingDirection"`` with values in {-1, 0, 1}.

        Returns
        -------
        self
        """
        clean = df.dropna(subset=["DominatingDirection"])
        series = clean["DominatingDirection"].values.astype(int)

        # Determine hour for each row from TimeUTC (or index)
        if "TimeUTC" in clean.columns:
            hours = pd.to_datetime(clean["TimeUTC"]).dt.hour.values
        else:
            hours = pd.to_datetime(clean.index).hour.values

        matrices = np.zeros((24, 3, 3), dtype=float)

        for t in range(len(series) - 1):
            cur, nxt = series[t], series[t + 1]
            h = hours[t]
            i, j = _STATE_IX[cur], _STATE_IX[nxt]
            matrices[h, i, j] += 1

        # Normalise rows → probabilities; handle zero-count rows with
        # a uniform fallback so sampling never fails.
        for h in range(24):
            for i in range(3):
                row_sum = matrices[h, i].sum()
                if row_sum > 0:
                    matrices[h, i] /= row_sum
                else:
                    matrices[h, i] = 1.0 / 3

        self.transition_matrices = matrices
        return self

    # ------------------------------------------------------------------
    # ScenarioModel interface
    # ------------------------------------------------------------------

    @property
    def required_inputs(self) -> list[str]:
        return ["initial_state", "start_hour"]

    def generate(
        self,
        n_scenarios: int,
        seed: int | None = None,
        **inputs,
    ) -> np.ndarray:
        """
        Recursively generate imbalance-direction scenarios.

        Parameters
        ----------
        n_scenarios : int
            Number of independent scenario paths to produce.
        seed : int or None
            Random seed for reproducibility.
        initial_state : int
            Last observed imbalance direction (-1, 0, or 1).
        start_hour : int
            Hour of day (0-23) for the first generated step.

        Returns
        -------
        np.ndarray of shape (n_scenarios, N_STEPS)
            Each entry is one of {-1, 0, 1}.
        """
        if self.transition_matrices is None:
            raise RuntimeError(
                "Model has not been fitted. Call .fit(df) first."
            )

        initial_state: int = inputs["initial_state"]
        start_hour: int = inputs["start_hour"]

        rng = np.random.default_rng(seed)
        scenarios = np.empty((n_scenarios, N_STEPS), dtype=int)

        for s in range(n_scenarios):
            state = initial_state
            for t in range(N_STEPS):
                # Current quarter's hour
                hour = (start_hour + (t * RESOLUTION_MIN) // 60) % 24
                row = self.transition_matrices[hour, _STATE_IX[state]]
                state = rng.choice(STATES, p=row)
                scenarios[s, t] = state

        return scenarios

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save transition matrices to a ``.npy`` file."""
        if self.transition_matrices is None:
            raise RuntimeError("Nothing to save – model not fitted.")
        np.save(path, self.transition_matrices)

    @classmethod
    def load(cls, path: str) -> "MarkovChainModel":
        """Load a fitted model from a ``.npy`` file."""
        matrices = np.load(path)
        return cls(transition_matrices=matrices)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_matrices(m: np.ndarray) -> None:
        if m.shape != (24, 3, 3):
            raise ValueError(
                f"Expected shape (24, 3, 3), got {m.shape}"
            )
        if not np.allclose(m.sum(axis=2), 1.0):
            raise ValueError("Rows of transition matrices must sum to 1.")
