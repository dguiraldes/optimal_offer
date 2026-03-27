import json
import os
from dataclasses import dataclass
from io import StringIO

import pandas as pd
import requests
from dotenv import load_dotenv

from src.data.base import BaseAPI


# ------------------------------------------------------------------
# Param models
# ------------------------------------------------------------------

@dataclass
class WindDataParams:
    lat: float
    lon: float
    date_from: str
    date_to: str
    capacity: float = 1.0
    height: int = 100
    turbine: str = "Vestas V80 2000"

    def to_query_params(self) -> dict:
        return {
            "lat": self.lat,
            "lon": self.lon,
            "date_from": self.date_from,
            "date_to": self.date_to,
            "capacity": self.capacity,
            "height": self.height,
            "turbine": self.turbine,
            "format": "json",
        }


# ------------------------------------------------------------------
# Client
# ------------------------------------------------------------------

_BASE_URL = "https://www.renewables.ninja/api"


class RenewablesNinjaAPI(BaseAPI):
    """Client for the Renewables.ninja API."""

    def __init__(self, token: str = None):
        if token is None:
            load_dotenv()  # No-op if no .env file exists
            token = os.environ.get("renewables_ninja_api_token")

        if token is None:
            raise ValueError(
                "No API token provided. Pass it as an argument or set "
                "'renewables_ninja_api_token' in your environment or .env file."
            )

        self._token = token

    # ------------------------------------------------------------------
    # Public endpoints
    # ------------------------------------------------------------------

    def wind_data(
        self,
        params: dict | None = None,
        *,
        lat: float | None = None,
        lon: float | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
        capacity: float = 1.0,
        height: int = 100,
        turbine: str = "Vestas V80 2000",
    ) -> pd.DataFrame:
        """
        Fetch wind power output data.

        Parameters
        ----------
        params : dict, optional
            All arguments as a dict. Overrides any explicit kwargs.
        lat : float
            Latitude of the location.
        lon : float
            Longitude of the location.
        date_from : str
            Start date in YYYY-MM-DD format.
        date_to : str
            End date in YYYY-MM-DD format.
        capacity : float
            Installed capacity in kW. Default is 1.0.
        height : int
            Hub height in metres. Default is 100.
        turbine : str
            Turbine model name. Default is 'Vestas V80 2000'.

        Returns
        -------
        pd.DataFrame
            Time-indexed DataFrame with wind power output.

        Examples
        --------
        >>> api.wind_data(lat=34.125, lon=39.814, date_from="2015-01-01", date_to="2015-12-31")
        >>> api.wind_data(params={"lat": 34.125, "lon": 39.814, "date_from": "2015-01-01", "date_to": "2015-12-31"})
        """
        query = self._resolve_params(
            params,
            WindDataParams,
            lat=lat,
            lon=lon,
            date_from=date_from,
            date_to=date_to,
            capacity=capacity,
            height=height,
            turbine=turbine,
        )
        data = self._get_with_token(f"{_BASE_URL}/data/wind", params=query)
        return self._parse_ninja_response(data)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_with_token(self, url: str, params: dict, timeout: int = 30):
        """GET request with Authorization header required by Renewables.ninja."""
        headers = {"Authorization": f"Token {self._token}"}
        response = requests.get(url, params=params, headers=headers, timeout=timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _parse_ninja_response(data: dict) -> pd.DataFrame:
        df = pd.read_json(StringIO(json.dumps(data["data"])), orient="index")
        df.index = pd.to_datetime(df.index)
        df.index.name = "time"
        return df
