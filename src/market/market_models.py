import linopy
import numpy as np
import pandas as pd
import xarray as xr

class BaseModel(linopy.Model):
    """Base stochastic optimization model for day-ahead wind farm bidding.

    Scenarios array shape: (4, n_scenarios, n_time)
      [0] wind power (MW), [1] spot price (EUR/MWh),
      [2] imbalance direction, [3] imbalance price (EUR/MWh)
    """

    def __init__(self, scenarios, P_max):
        super().__init__()
        self.n_scenarios = scenarios.shape[1]
        self.n_time = scenarios.shape[2]
        self.P_max = P_max
        self.prob = np.ones(self.n_scenarios) / self.n_scenarios

        coords = {"omega": range(self.n_scenarios), "t": range(self.n_time)}
        self.p_real = xr.DataArray(scenarios[0], dims=["omega", "t"], coords=coords)
        self.spot_price = xr.DataArray(scenarios[1], dims=["omega", "t"], coords=coords)
        self.imb_direction = xr.DataArray(scenarios[2], dims=["omega", "t"], coords=coords)
        self.imb_price = xr.DataArray(scenarios[3], dims=["omega", "t"], coords=coords)
        self.pi = xr.DataArray(self.prob, dims=["omega"], coords={"omega": range(self.n_scenarios)})

        self.p_DA = self.add_variables(
            lower=0, upper=self.P_max,
            coords=[pd.Index(range(self.n_time), name="t")], name="p_DA",
        )

    @property
    def p_DA_opt(self):
        if self.p_DA.solution is None:
            raise ValueError("Model not solved. Call fit() first.")
        return self.p_DA.solution.values

    def _build_scenario_profit_expr(self):
        """Return linopy expression for per-scenario profit — variable-dependent terms only (shape: omega)."""
        raise NotImplementedError

    def _scenario_profit_constant(self):
        """Return DataArray (omega,) of per-scenario constant terms (no decision variables).
        Override when the profit has terms independent of decision variables."""
        return 0

    def evaluate_da(self, spot_price):
        """Day-ahead profit per period (numpy). Shape: same as spot_price."""
        return self.p_DA_opt * spot_price / 4

    def evaluate_imbalance(self, p_real, spot_price, imb_price, imb_direction=None):
        """Imbalance profit per period (numpy). Must be overridden."""
        raise NotImplementedError

    def evaluate(self, p_real, spot_price, imb_price, imb_direction=None):
        """Total per-period profit = DA + imbalance."""
        return (self.evaluate_da(spot_price)
                + self.evaluate_imbalance(p_real, spot_price, imb_price, imb_direction))

    def fit(self, solver_name="highs", alpha=0.95, beta=0):
        """Solve the stochastic optimisation.

        Args:
            solver_name: LP solver backend.
            alpha: CVaR confidence level (used when beta > 0).
            beta: Risk-aversion weight in [0, 1].
                  0 = pure expected-profit maximisation,
                  1 = pure CVaR maximisation.
        """
        var_profit = self._build_scenario_profit_expr()           # LinExpr (omega,)
        expected_profit = (self.pi * var_profit).sum("omega")     # scalar LinExpr

        if beta == 0:
            self.add_objective(expected_profit, sense="max")
        else:
            # Full profit needed for CVaR: constant shifts which scenarios are worst-case
            full_profit = var_profit + self._scenario_profit_constant()

            self.VaR = self.add_variables(name="VaR")
            self.nu = self.add_variables(
                lower=0, dims=["omega"],
                coords={"omega": range(self.n_scenarios)}, name="nu",
            )
            self.add_constraints(
                full_profit - self.VaR + self.nu >= 0,
                name="CVaR_constraints",
            )
            cvar = self.VaR - (self.nu * self.pi).sum("omega") / (1 - alpha)
            obj = (1 - beta) * expected_profit + beta * cvar
            self.add_objective(obj, sense="max")

        self.solve(solver_name=solver_name)
        self._compute_expected_profits()

    def _compute_expected_profits(self):
        da = self.evaluate_da(self.spot_price.values)
        imb = self.evaluate_imbalance(
            self.p_real.values, self.spot_price.values,
            self.imb_price.values, self.imb_direction.values,
        )
        profit = da + imb

        self.expected_profit_t = profit.mean(axis=0)
        self.expected_profit_omega = profit.sum(axis=1)
        self.expected_profit_sum = float(self.expected_profit_t.sum())

        self.expected_day_ahead_profit_t = da.mean(axis=0)
        self.expected_day_ahead_profit_omega = da.sum(axis=1)
        self.expected_day_ahead_profit_sum = float(self.expected_day_ahead_profit_t.sum())

        self.expected_imbalance_cost_t = imb.mean(axis=0)
        self.expected_imbalance_cost_omega = imb.sum(axis=1)
        self.expected_imbalance_cost_sum = float(self.expected_imbalance_cost_t.sum())



class OnePriceModel(BaseModel):

    def _build_scenario_profit_expr(self):
        # profit_ω = Σ_t [(λ_DA − λ_I)·p_DA] / 4   (variable part only)
        coeff = (self.spot_price - self.imb_price) / 4
        return (coeff * self.p_DA).sum("t")

    def _scenario_profit_constant(self):
        # Σ_t [λ_I · p_real] / 4  — no decision variables
        return ((self.imb_price * self.p_real) / 4).sum("t")

    def evaluate_imbalance(self, p_real, spot_price, imb_price, imb_direction=None):
        p_DA = self.p_DA_opt
        return imb_price * (p_real - p_DA) / 4


class TwoPriceModel(BaseModel):

    def _build_scenario_profit_expr(self):
        self.lambda_plus = self.spot_price.where(self.imb_direction >= 0, self.imb_price)
        self.lambda_minus = self.spot_price.where(self.imb_direction < 0, self.imb_price)

        coords = [
            pd.Index(range(self.n_scenarios), name="omega"),
            pd.Index(range(self.n_time), name="t"),
        ]
        self.delta_plus = self.add_variables(
            lower=0, dims=["omega", "t"], coords=coords, name="delta_plus",
        )
        self.delta_minus = self.add_variables(
            lower=0, dims=["omega", "t"], coords=coords, name="delta_minus",
        )
        self.add_constraints(
            self.delta_plus - self.delta_minus + self.p_DA == self.p_real,
            name="delta_balance",
        )

        # All terms involve decision variables — no constant needed
        return (
            self.spot_price / 4 * self.p_DA
            + self.lambda_plus / 4 * self.delta_plus
            - self.lambda_minus / 4 * self.delta_minus
        ).sum("t")

    def evaluate_imbalance(self, p_real, spot_price, imb_price, imb_direction=None):
        p_DA = self.p_DA_opt
        lambda_plus = np.where(imb_direction >= 0, spot_price, imb_price)
        lambda_minus = np.where(imb_direction < 0, spot_price, imb_price)
        delta_plus = np.maximum(p_real - p_DA, 0)
        delta_minus = np.maximum(p_DA - p_real, 0)
        return (lambda_plus * delta_plus - lambda_minus * delta_minus) / 4