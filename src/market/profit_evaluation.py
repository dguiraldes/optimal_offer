import numpy as np


class OnePriceEvaluator:
    """Evaluator for One-Price Imbalance Settlement.

    Imbalance (over/under production) is settled at the single imbalance price,
    regardless of grid direction.

    Parameters
    ----------
    p_da               : Day-ahead quantity offered (MW).
    p_real             : Realised production (MW).
    spot_price         : Day-ahead spot price (EUR/MWh).
    imbalance_price    : Single imbalance settlement price (EUR/MWh).
    imbalance_direction: Not used; accepted for API symmetry with TwoPriceEvaluator.

    Attributes
    ----------
    da_profit    : Day-ahead revenue  = spot_price * p_da / 4
    imb_profit   : Imbalance revenue  = imbalance_price * (p_real - p_da) / 4
    total_profit : da_profit + imb_profit
    """

    def __init__(self, p_da, p_real, spot_price, imbalance_price, imbalance_direction=None):
        self.da_profit    = spot_price * p_da / 4
        self.imb_profit   = imbalance_price * (p_real - p_da) / 4
        self.total_profit = self.da_profit + self.imb_profit


class TwoPriceEvaluator:
    """Evaluator for Two-Price Imbalance Settlement.

    - Grid short (imbalance_direction >= 0):
        surplus rewarded at spot_price, deficit penalised at imbalance_price.
    - Grid long  (imbalance_direction <  0):
        surplus penalised at imbalance_price, deficit penalised at spot_price.

    Parameters
    ----------
    p_da               : Day-ahead quantity offered (MW).
    p_real             : Realised production (MW).
    spot_price         : Day-ahead spot price (EUR/MWh).
    imbalance_price    : Imbalance settlement price (EUR/MWh).
    imbalance_direction: Grid imbalance direction signal (>= 0 → short, < 0 → long).

    Attributes
    ----------
    da_profit    : Day-ahead revenue  = spot_price * p_da / 4
    imb_profit   : Imbalance revenue  = (lambda_plus * delta_plus - lambda_minus * delta_minus) / 4
    total_profit : da_profit + imb_profit
    """

    def __init__(self, p_da, p_real, spot_price, imbalance_price, imbalance_direction):
        imbalance_direction = np.asarray(imbalance_direction)

        lambda_plus  = np.where(imbalance_direction >= 0, spot_price, imbalance_price)
        lambda_minus = np.where(imbalance_direction <  0, spot_price, imbalance_price)

        delta_plus  = np.maximum(p_real - p_da, 0)   # overproduction
        delta_minus = np.maximum(p_da - p_real, 0)   # underproduction

        self.da_profit    = spot_price * p_da / 4
        self.imb_profit   = (lambda_plus * delta_plus - lambda_minus * delta_minus) / 4
        self.total_profit = self.da_profit + self.imb_profit
