"""Strategy evaluation utilities for day-ahead wind power bidding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from src.market.profit_evaluation import OnePriceEvaluator, TwoPriceEvaluator

SettlementType = Literal["one_price", "two_price"]

_PERIODS_PER_DAY = 96  # 15-min intervals in one day


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean Absolute Percentage Error, ignoring time steps where ``y_true`` is zero."""
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])))


class DayEvaluator:
    """Evaluates bidding strategies for a single trading day.

    Holds stochastic scenario data and optional real observations for one day,
    and exposes :meth:`evaluate_strategy` to score any day-ahead bid vector
    against both the scenario distribution and realised outcomes.

    Parameters
    ----------
    date:
        The trading date; used to build the time-series index ``ts_xaxis``.
    imbalance_settlement:
        Settlement scheme – ``"two_price"`` or ``"one_price"``.
    scenarios:
        Array of shape ``(4, n_scenarios, 96)`` with the following variable
        order along axis 0:

        * 0 – offshore wind power realisation  [MW]
        * 1 – day-ahead spot price             [EUR/MWh]
        * 2 – grid imbalance direction         [dimensionless]
        * 3 – imbalance settlement price       [EUR/MWh]

    df_real:
        Optional DataFrame (96 rows) with columns ``OffshoreWindPower``,
        ``SpotPriceEUR``, ``ImbalancePriceEUR``, ``DominatingDirection`` for
        out-of-sample realised performance evaluation.
    """

    def __init__(
        self,
        date: str | pd.Timestamp,
        imbalance_settlement: SettlementType,
        scenarios: np.ndarray,
        df_real: Optional[pd.DataFrame] = None,
    ) -> None:
        self.date = date
        self.imbalance_settlement = imbalance_settlement

        # Unpack scenario cube – each array has shape (n_scenarios, 96)
        self.p_real_scenarios: np.ndarray = scenarios[0]
        self.spot_price_scenarios: np.ndarray = scenarios[1]
        self.imbalance_direction_scenarios: np.ndarray = scenarios[2]
        self.imbalance_price_scenarios: np.ndarray = scenarios[3]

        # Cached mean scenario realisation – reused across strategy evaluations
        self._mean_p_real: np.ndarray = self.p_real_scenarios.mean(axis=0)

        if df_real is not None:
            self.p_real: Optional[np.ndarray] = df_real["OffshoreWindPower"].to_numpy()
            self.spot_price: Optional[np.ndarray] = df_real["SpotPriceEUR"].to_numpy()
            self.imbalance_price: Optional[np.ndarray] = df_real["ImbalancePriceEUR"].to_numpy()
            self.imbalance_direction: Optional[np.ndarray] = df_real["DominatingDirection"].to_numpy()
        else:
            self.p_real = self.spot_price = self.imbalance_price = self.imbalance_direction = None

        self._evaluator_cls = (
            TwoPriceEvaluator if imbalance_settlement == "two_price" else OnePriceEvaluator
        )
        self.ts_xaxis: pd.DatetimeIndex = pd.date_range(
            start=date, periods=_PERIODS_PER_DAY, freq="15min"
        )
        self.expected_perfect_profit: float = self._compute_perfect_profit()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _compute_perfect_profit(self) -> float:
        """Expected daily profit under perfect foresight (zero imbalance, pure DA revenue)."""
        evaluator = self._evaluator_cls(
            p_da=self.p_real_scenarios,
            p_real=self.p_real_scenarios,
            spot_price=self.spot_price_scenarios,
            imbalance_price=self.imbalance_price_scenarios,
            imbalance_direction=self.imbalance_direction_scenarios,
        )
        return float(evaluator.total_profit.sum(axis=1).mean())

    def _compute_scenario_metrics(self, p_DA: np.ndarray, alpha_CVaR: float) -> dict:
        """Compute all scenario-based performance metrics for a given bid.

        Parameters
        ----------
        p_DA:
            Day-ahead bid vector, shape ``(96,)`` [MW].
        alpha_CVaR:
            Confidence level for CVaR / VaR (e.g. ``0.95`` → 5 % tail).

        Returns
        -------
        dict
            Flat mapping of metric names to scalar values.
        """
        evaluator = self._evaluator_cls(
            p_da=p_DA,
            p_real=self.p_real_scenarios,
            spot_price=self.spot_price_scenarios,
            imbalance_price=self.imbalance_price_scenarios,
            imbalance_direction=self.imbalance_direction_scenarios,
        )
        profit_per_scenario: np.ndarray = evaluator.total_profit.sum(axis=1)
        expected_profit = float(profit_per_scenario.mean())
        stdev_profit = float(profit_per_scenario.std())

        # Risk metrics
        var_alpha = float(np.quantile(profit_per_scenario, 1.0 - alpha_CVaR))
        cvar_alpha = float(profit_per_scenario[profit_per_scenario <= var_alpha].mean())
        prob_loss = float((profit_per_scenario < 0).mean())
        profit_to_risk = expected_profit / stdev_profit if stdev_profit > 0 else np.nan

        # Benchmark vs. perfect-foresight
        expected_regret = float(self.expected_perfect_profit - expected_profit)
        profit_efficiency = (
            expected_profit / self.expected_perfect_profit
            if self.expected_perfect_profit != 0
            else np.nan
        )

        # Revenue decomposition (DA vs. imbalance)
        expected_da_revenue = float(evaluator.da_profit.sum(axis=1).mean())
        expected_imb_revenue = float(evaluator.imb_profit.sum(axis=1).mean())

        # Bid quality metrics
        bid_bias_mw = float((p_DA - self._mean_p_real).mean())
        expected_exposure = _mape(self._mean_p_real, p_DA)

        return dict(
            profit_per_scenario=profit_per_scenario,
            expected_profit=expected_profit,
            stdev_profit=stdev_profit,
            var_alpha=var_alpha,
            cvar_alpha=cvar_alpha,
            prob_loss=prob_loss,
            profit_to_risk=profit_to_risk,
            expected_regret=expected_regret,
            profit_efficiency=profit_efficiency,
            expected_da_revenue=expected_da_revenue,
            expected_imb_revenue=expected_imb_revenue,
            bid_bias_mw=bid_bias_mw,
            expected_exposure=expected_exposure,
        )

    def _compute_real_metrics(self, p_DA: np.ndarray) -> dict:
        """Evaluate the bid against realised observations (when available).

        Returns
        -------
        dict
            ``profit_real`` and ``exposure_to_imbalance``, both ``None`` when
            no real data was supplied at construction time.
        """
        if self.p_real is None:
            return dict(profit_real=None, exposure_to_imbalance=None)

        evaluator = self._evaluator_cls(
            p_da=p_DA,
            p_real=self.p_real,
            spot_price=self.spot_price,
            imbalance_price=self.imbalance_price,
            imbalance_direction=self.imbalance_direction,
        )
        return dict(
            profit_real=float(evaluator.total_profit.sum()),
            exposure_to_imbalance=_mape(self.p_real, p_DA),
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate_strategy(
        self,
        p_DA: np.ndarray,
        strategy_id: int,
        strategy_name: str,
        others: Optional[dict] = None,
        alpha_CVaR: float = 0.95,
    ) -> "StrategyResults":
        """Evaluate a day-ahead bidding strategy.

        Parameters
        ----------
        p_DA:
            Day-ahead bid vector, shape ``(96,)`` [MW].
        strategy_id:
            Numeric identifier for bookkeeping / plotting.
        strategy_name:
            Human-readable label.
        others:
            Arbitrary extra metadata to attach to the result (e.g. optimiser
            parameters).
        alpha_CVaR:
            Tail probability for VaR/CVaR risk metrics (default ``0.95``).

        Returns
        -------
        StrategyResults
            Fully populated results dataclass.
        """
        scenario_metrics = self._compute_scenario_metrics(p_DA, alpha_CVaR)
        real_metrics = self._compute_real_metrics(p_DA)

        return StrategyResults(
            strategy_id=strategy_id,
            strategy_name=strategy_name,
            details=others,
            p_DA=p_DA,
            **scenario_metrics,
            **real_metrics,
        )


# ---------------------------------------------------------------------------
# Results container
# ---------------------------------------------------------------------------

@dataclass
class StrategyResults:
    """Container for all performance metrics of a single bidding strategy.

    Attributes
    ----------
    strategy_id:
        Numeric identifier.
    strategy_name:
        Human-readable label.
    details:
        Optional metadata dict (e.g. optimiser parameters).
    p_DA:
        Day-ahead bid vector used [MW], shape ``(96,)``.
    profit_per_scenario:
        Total daily profit for each scenario [EUR], shape ``(n_scenarios,)``.
    expected_profit:
        Mean profit across scenarios [EUR].
    stdev_profit:
        Standard deviation of scenario profits [EUR].
    var_alpha:
        Value-at-Risk at the ``(1 - alpha_CVaR)`` quantile [EUR].
    cvar_alpha:
        Conditional VaR / Expected Shortfall at the same tail [EUR].
    prob_loss:
        Fraction of scenarios with negative profit [0, 1].
    profit_to_risk:
        Expected profit divided by standard deviation (Sharpe-like ratio).
    expected_regret:
        Perfect-foresight expected profit minus strategy expected profit [EUR].
    profit_efficiency:
        ``expected_profit / expected_perfect_profit`` [0, 1].
    expected_da_revenue:
        Mean day-ahead revenue component across scenarios [EUR].
    expected_imb_revenue:
        Mean imbalance revenue component across scenarios [EUR].
    bid_bias_mw:
        Signed mean deviation of bid from expected realisation [MW].
        Positive → overbid; negative → underbid.
    expected_exposure:
        MAPE between bid and mean scenario realisation.
    profit_real:
        Realised daily profit computed against actual observations [EUR],
        or ``None`` if no real data was supplied.
    exposure_to_imbalance:
        MAPE between bid and actual realisation,
        or ``None`` if no real data was supplied.
    """

    strategy_id: int
    strategy_name: str
    details: Optional[dict]
    p_DA: np.ndarray
    profit_per_scenario: np.ndarray
    expected_profit: float
    stdev_profit: float
    var_alpha: float
    cvar_alpha: float
    prob_loss: float
    profit_to_risk: float
    expected_regret: float
    profit_efficiency: float
    expected_da_revenue: float
    expected_imb_revenue: float
    bid_bias_mw: float
    expected_exposure: float
    profit_real: Optional[float]
    exposure_to_imbalance: Optional[float]