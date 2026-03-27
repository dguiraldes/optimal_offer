import json
from dataclasses import dataclass

import pandas as pd

from src.data.base import BaseAPI


# ------------------------------------------------------------------
# Param models
# ------------------------------------------------------------------

@dataclass
class ImbalancePriceParams:
    start: str
    end: str
    PriceArea: str

    def to_query_params(self) -> dict:
        return {
            "offset": 0,
            "start": f"{self.start}T00:00",
            "end": f"{self.end}T00:00",
            "filter": json.dumps({"PriceArea": [self.PriceArea]}),
            "sort": "TimeUTC ASC",
        }


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------

_BASE_URL = "https://api.energidataservice.dk/dataset"


class EnerginetAPI(BaseAPI):
    """Client for the Energinet Data Service API."""

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def imbalance_dataset(
        self,
        params: dict | None = None,
        *,
        start: str | None = None,
        end: str | None = None,
        price_area: str | None = None,
        currency: str = "EUR",
    ) -> pd.DataFrame:
        """
        Fetch imbalance dataset.

        Parameters
        ----------
        params : dict, optional
            All arguments as a dict. Overrides any explicit kwargs.
        start : str
            Start date in YYYY-MM-DD format.
        end : str
            End date in YYYY-MM-DD format.
        price_area : str
            Price area code, e.g. 'DK1', 'DK2', 'SE3'.

        Returns
        -------
        pd.DataFrame
            Columns: [
            TimeUTC, TimeDK, PriceArea, SatisfiedDemand, ImbalancePriceEUR, 
            ImbalancePriceDKK, SpotPriceEUR, DominatingDirection, 
            aFRRUpMW, aFRRVWAUpEUR, aFRRVWAUpDKK, aFRRDownMW, 
            aFRRVWADownEUR, aFRRVWADownDKK, mFRRMarginalPriceUpEUR, 
            mFRRMarginalPriceUpDKK, mFRRMarginalPriceDownEUR, 
            mFRRMarginalPriceDownDKK].

        Examples
        --------
        >>> api.imbalance_dataset({"start": "2026-03-01", "end": "2026-03-31", "price_area": "DK2"})
        >>> api.imbalance_dataset(params={"start": "2026-03-01", "end": "2026-03-31", "price_area": "DK2"})
        >>> api.imbalance_dataset(start="2026-03-01", end="2026-03-31", price_area="DK2")
        """
        query = self._resolve_params(
            params,
            ImbalancePriceParams,
            start=start,
            end=end,
            PriceArea=price_area,
        )
        data = self._get(f"{_BASE_URL}/ImbalancePrice", params=query)
        return self._parse_imbalance_dataset(data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_imbalance_dataset(data: dict) -> pd.DataFrame:
        df = pd.DataFrame(data["records"])
        df["TimeUTC"] = pd.to_datetime(df["TimeUTC"])
        df["TimeDK"] = pd.to_datetime(df["TimeDK"])
        return df
