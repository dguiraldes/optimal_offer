# Optimal wind farm offering simulation

This repository contains the code for simulating optimal wind farm offering considering the day ahead and balancing markets. It's based on the Assignment 2 of the course "Renewables in Energy Markets" at DTU by Professor Jalal Kazempour in 2025.

## Data clients

All API clients live under `src/data/` as flat modules:

```
src/data/
    __init__.py          # Re-exports all API classes
    base.py              # BaseAPI with shared _get() and _resolve_params()
    energinet.py         # EnerginetAPI
    nordpool.py          # NordpoolAPI
    renewablesninja.py   # RenewablesNinjaAPI
```

Every client inherits from `BaseAPI`, which provides:
- `_get()` — HTTP GET with error handling
- `_resolve_params()` — resolves a raw `dict` or typed kwargs into query parameters

### Adding a new endpoint to an existing API

1. Open the corresponding module (e.g. `src/data/nordpool.py`).
2. Define a params dataclass for the new endpoint:

```python
@dataclass
class CapacityPriceParams:
    date: str
    delivery_area: str

    def to_query_params(self) -> dict:
        return {
            "date": self.date,
            "deliveryArea": self.delivery_area,
        }
```

3. Add a public method to the API class:

```python
def capacity_prices(
    self,
    params: dict | None = None,
    *,
    date: str | None = None,
    delivery_area: str | None = None,
) -> pd.DataFrame:
    query = self._resolve_params(params, CapacityPriceParams, date=date, delivery_area=delivery_area)
    data = self._get(f"{_BASE_URL}/CapacityPrices", params=query)
    return self._parse_capacity_prices(data)
```

4. Add a static `_parse_*` helper to convert the JSON response into a DataFrame.

### Adding a new API source

1. Create a new module `src/data/newapi.py`:

```python
from dataclasses import dataclass

import pandas as pd

from src.data.base import BaseAPI

# -- Param models --

@dataclass
class SomeEndpointParams:
    start: str
    end: str

    def to_query_params(self) -> dict:
        return {"start": self.start, "end": self.end}

# -- Client --

_BASE_URL = "https://api.example.com"

class NewAPI(BaseAPI):
    """Client for the New API."""

    def some_endpoint(
        self,
        params: dict | None = None,
        *,
        start: str | None = None,
        end: str | None = None,
    ) -> pd.DataFrame:
        query = self._resolve_params(params, SomeEndpointParams, start=start, end=end)
        data = self._get(f"{_BASE_URL}/some-endpoint", params=query)
        return pd.DataFrame(data["results"])
```

2. Register it in `src/data/__init__.py`:

```python
from src.data.newapi import NewAPI

__all__ = [
    ...,
    "NewAPI",
]
```