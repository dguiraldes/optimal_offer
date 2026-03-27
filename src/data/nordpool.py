from dataclasses import dataclass

import pandas as pd

from src.data.base import BaseAPI


# ------------------------------------------------------------------
# Param models
# ------------------------------------------------------------------

@dataclass
class DayAheadPriceParams:
    date: str
    delivery_area: str
    market: str = "DayAhead"
    currency: str = "EUR"

    def to_query_params(self) -> dict:
        return {
            "date": self.date,
            "market": self.market,
            "deliveryArea": self.delivery_area,
            "currency": self.currency,
        }


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------

_BASE_URL = "https://dataportal-api.nordpoolgroup.com/api"


class NordpoolAPI(BaseAPI):
    """Client for the Nordpool Data Portal API."""

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def day_ahead_price(
        self,
        params: dict | None = None,
        *,
        date: str | None = None,
        delivery_area: str | None = None,
        currency: str = "EUR",
    ) -> pd.DataFrame:
        """
        Fetch day-ahead spot prices.

        Parameters
        ----------
        params : dict, optional
            All arguments as a dict. Overrides any explicit kwargs.
        date : str
            Delivery date in YYYY-MM-DD format.
        delivery_area : str
            Bidding zone code, e.g. 'DK1', 'DK2', 'SE3'.
        currency : str
            Currency for prices. Default is 'EUR'.

        Returns
        -------
        pd.DataFrame
            Columns: [deliveryStart, deliveryEnd, area, price_EUR].

        Examples
        --------
        >>> api.day_ahead_price({"date": "2026-03-15", "delivery_area": "DK2"})
        >>> api.day_ahead_price(params={"date": "2026-03-15", "delivery_area": "DK2"})
        >>> api.day_ahead_price(date="2026-03-15", delivery_area="DK2")
        """
        query = self._resolve_params(
            params,
            DayAheadPriceParams,
            date=date,
            delivery_area=delivery_area,
            currency=currency,
        )
        data = self._get(f"{_BASE_URL}/DayAheadPrices", params=query)
        return self._parse_day_ahead_price(data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_day_ahead_price(data: dict) -> pd.DataFrame:
        rows = [
            {
                "deliveryStart": entry["deliveryStart"],
                "deliveryEnd": entry["deliveryEnd"],
                "area": area,
                "price_EUR": price,
            }
            for entry in data["multiAreaEntries"]
            for area, price in entry["entryPerArea"].items()
        ]
        df = pd.DataFrame(rows)
        df["deliveryStart"] = pd.to_datetime(df["deliveryStart"])
        df["deliveryEnd"] = pd.to_datetime(df["deliveryEnd"])
        return df
