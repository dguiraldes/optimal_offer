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


@dataclass
class Forecasts5MinParams:
    start: str
    end: str
    PriceArea: str
    ForecastType: str | None = None

    def to_query_params(self) -> dict:
        filt: dict[str, list[str]] = {"PriceArea": [self.PriceArea]}
        if self.ForecastType is not None:
            filt["ForecastType"] = [self.ForecastType]
        return {
            "offset": 0,
            "start": f"{self.start}T00:00",
            "end": f"{self.end}T00:00",
            "filter": json.dumps(filt),
            "sort": "Minutes5UTC DESC",
        }


@dataclass
class ProductionExchange5MinParams:
    start: str
    end: str
    PriceArea: str

    def to_query_params(self) -> dict:
        return {
            "offset": 0,
            "start": f"{self.start}T00:00",
            "end": f"{self.end}T00:00",
            "filter": json.dumps({"PriceArea": [self.PriceArea]}),
            "sort": "Minutes5UTC DESC",
        }


@dataclass
class CapacityPerMunicipalityParams:
    start: str
    end: str
    MunicipalityNo: str | None = None

    def to_query_params(self) -> dict:
        query: dict = {
            "offset": 0,
            "start": f"{self.start}T00:00",
            "end": f"{self.end}T00:00",
            "sort": "Month DESC",
        }
        if self.MunicipalityNo is not None:
            query["filter"] = json.dumps({"MunicipalityNo": [self.MunicipalityNo]})
        return query


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

    def forecasts_5min(
        self,
        params: dict | None = None,
        *,
        start: str | None = None,
        end: str | None = None,
        price_area: str | None = None,
        forecast_type: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch Forecast Wind and Solar Power at 5-minute resolution.

        Parameters
        ----------
        params : dict, optional
            All arguments as a dict. Overrides any explicit kwargs.
        start : str
            Start date in YYYY-MM-DD format.
        end : str
            End date in YYYY-MM-DD format.
        price_area : str
            Price area code, e.g. 'DK1', 'DK2'.
        forecast_type : str, optional
            One of 'Solar', 'Offshore Wind', 'Onshore Wind'.
            If omitted, all types are returned.

        Returns
        -------
        pd.DataFrame
            Columns: [
            Minutes5UTC, Minutes5DK, PriceArea, ForecastType,
            ForecastDayAhead, Forecast5Hour, Forecast1Hour,
            ForecastCurrent, TimestampUTC, TimestampDK].

        Examples
        --------
        >>> api.forecasts_5min(start="2026-01-01", end="2026-03-31", price_area="DK2")
        >>> api.forecasts_5min(start="2026-01-01", end="2026-03-31", price_area="DK2", forecast_type="Offshore Wind")
        """
        query = self._resolve_params(
            params,
            Forecasts5MinParams,
            start=start,
            end=end,
            PriceArea=price_area,
            ForecastType=forecast_type,
        )
        data = self._get(f"{_BASE_URL}/Forecasts_5Min", params=query)
        return self._parse_forecasts_5min(data)

    def production_exchange_5min(
        self,
        params: dict | None = None,
        *,
        start: str | None = None,
        end: str | None = None,
        price_area: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch Electricity Production and Exchange at 5-minute resolution.

        Real-time SCADA-based production and cross-border exchange data,
        updated every 5 minutes.

        Parameters
        ----------
        params : dict, optional
            All arguments as a dict. Overrides any explicit kwargs.
        start : str
            Start date in YYYY-MM-DD format.
        end : str
            End date in YYYY-MM-DD format.
        price_area : str
            Price area code, e.g. 'DK1', 'DK2'.

        Returns
        -------
        pd.DataFrame
            Columns: [
            Minutes5UTC, Minutes5DK, PriceArea,
            ProductionLt100MW, ProductionGe100MW,
            OffshoreWindPower, OnshoreWindPower, SolarPower,
            ExchangeGreatBelt, ExchangeGermany, ExchangeNetherlands,
            ExchangeGreatBritain, ExchangeNorway, ExchangeSweden,
            BornholmSE4].

        Examples
        --------
        >>> api.production_exchange_5min(start="2026-01-01", end="2026-03-31", price_area="DK1")
        """
        query = self._resolve_params(
            params,
            ProductionExchange5MinParams,
            start=start,
            end=end,
            PriceArea=price_area,
        )
        data = self._get(f"{_BASE_URL}/ElectricityProdex5MinRealtime", params=query)
        return self._parse_production_exchange_5min(data)

    def capacity_per_municipality(
        self,
        params: dict | None = None,
        *,
        start: str | None = None,
        end: str | None = None,
        municipality_no: str | None = None,
    ) -> pd.DataFrame:
        """
        Fetch installed capacity and unit counts per municipality.

        Updated during the first week of the following month.
        Capacities are calculated at midnight on the first day of the month.

        Parameters
        ----------
        params : dict, optional
            All arguments as a dict. Overrides any explicit kwargs.
        start : str
            Start date in YYYY-MM-DD format.
        end : str
            End date in YYYY-MM-DD format.
        municipality_no : str, optional
            Danish municipality number (e.g. '101' for Copenhagen).
            If omitted, all municipalities are returned.

        Returns
        -------
        pd.DataFrame
            Columns: [
            Month, MunicipalityNo,
            CapacityGe100MW, CapacityLt100MW,
            OffshoreWindCapacity, OnshoreWindCapacity, SolarPowerCapacity,
            NumberGenerationUnitsGe100MW, NumberGenerationUnitsLt100MW,
            NumberOffshoreWindGenerators, NumberOnshoreWindGenerators,
            NumberSolarPanels].

        Examples
        --------
        >>> api.capacity_per_municipality(start="2025-01-01", end="2026-01-01")
        >>> api.capacity_per_municipality(start="2025-01-01", end="2026-01-01", municipality_no="101")
        """
        query = self._resolve_params(
            params,
            CapacityPerMunicipalityParams,
            start=start,
            end=end,
            MunicipalityNo=municipality_no,
        )
        data = self._get(f"{_BASE_URL}/CapacityPerMunicipality", params=query)
        return self._parse_capacity_per_municipality(data)

    # ------------------------------------------------------------------
    # Private helpers to handle outputs
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_imbalance_dataset(data: dict) -> pd.DataFrame:
        df = pd.DataFrame(data["records"])
        df["TimeUTC"] = pd.to_datetime(df["TimeUTC"])
        df["TimeDK"] = pd.to_datetime(df["TimeDK"])
        return df

    @staticmethod
    def _parse_forecasts_5min(data: dict) -> pd.DataFrame:
        df = pd.DataFrame(data["records"])
        df["Minutes5UTC"] = pd.to_datetime(df["Minutes5UTC"])
        df["Minutes5DK"] = pd.to_datetime(df["Minutes5DK"])
        if "TimestampUTC" in df.columns:
            df["TimestampUTC"] = pd.to_datetime(df["TimestampUTC"])
        if "TimestampDK" in df.columns:
            df["TimestampDK"] = pd.to_datetime(df["TimestampDK"])
        return df

    @staticmethod
    def _parse_production_exchange_5min(data: dict) -> pd.DataFrame:
        df = pd.DataFrame(data["records"])
        df["Minutes5UTC"] = pd.to_datetime(df["Minutes5UTC"])
        df["Minutes5DK"] = pd.to_datetime(df["Minutes5DK"])
        return df

    @staticmethod
    def _parse_capacity_per_municipality(data: dict) -> pd.DataFrame:
        df = pd.DataFrame(data["records"])
        df["Month"] = pd.to_datetime(df["Month"])
        return df
