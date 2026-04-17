# Optimal wind farm offering simulation

This repository contains the code for simulating optimal wind farm offering considering the day ahead and balancing markets. It's based on the Assignment 2 of the course "Renewables in Energy Markets" at DTU by Professor Jalal Kazempour in 2025.

## Getting Started

### Prerequisites
- Python 3.11+
- Git

### Installation

1. **Clone the repository**
```bash
   git clone git@github.com:dguiraldes/optimal_offer.git
   cd optimal_offer
```

2. **Create and activate a virtual environment**

   macOS/Linux:
```bash
   python -m venv .venv
   source .venv/bin/activate
```

   Windows:
```bash
   python -m venv .venv
   .venv\Scripts\activate
```

3. **Install the project and its dependencies**
```bash
   pip install -e .
```

That's it. The `src/` package is now importable from anywhere in the project,
including notebooks.


### Environment variables
Some data sources require API keys. Copy the example file and fill in your credentials:
```bash
cp .env.example .env
```
Never commit the `.env` file — it is listed in `.gitignore`.

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

## Scenario generators

Scenario generators live under `src/scenarios/`, with one subfolder per resource:

```
src/scenarios/
    __init__.py              # Re-exports all generators
    base.py                  # ScenarioModel (ABC) + ScenarioGenerator (front-end)
    wind_production/
        __init__.py          # WindProductionGenerator
        conditional_nvp.py   # Conditional Real-NVP normalizing flow
        gaussian_noise.py    # Gaussian noise baseline (placeholder)
    spot_price/
        __init__.py          # SpotPriceGenerator
        arimax.py            # ARIMAX model (placeholder)
    imbalance_direction/
        __init__.py          # ImbalanceDirectionGenerator
        markov_chain.py      # Markov chain model (placeholder)
    imbalance_price/
        __init__.py          # ImbalancePriceGenerator
        conditional_sampling.py  # Conditional sampling (placeholder)
```

Pre-trained model weights are stored in `trained_models/` and training notebooks
live in `model_training/`.

### Using a generator

```python
from src.scenarios.wind_production import WindProductionGenerator

# Load with the default pre-trained checkpoint
gen = WindProductionGenerator(model="conditional_nvp")

# Or point to a specific checkpoint
gen = WindProductionGenerator(model="conditional_nvp", model_path="path/to/model.pt")

# Check required inputs
print(gen.required_inputs)  # ['wind_forecast']

# Generate 200 scenarios conditioned on a day-ahead forecast (numpy array, shape (96,))
scenarios = gen.generate(n_scenarios=200, wind_forecast=forecast)
# scenarios.shape == (200, 96)
```

All generators follow the same interface — swap `WindProductionGenerator` for
`SpotPriceGenerator`, `ImbalanceDirectionGenerator`, or `ImbalancePriceGenerator`.

### Adding a new model to an existing generator

1. Create a new file in the generator's folder, e.g. `src/scenarios/wind_production/my_model.py`:

```python
import numpy as np
from src.scenarios.base import ScenarioModel, N_STEPS

class MyModel(ScenarioModel):
    name = "my_model"

    def __init__(self, model_path=None):
        # Load your pre-trained weights here
        ...

    @property
    def required_inputs(self) -> list[str]:
        return ["wind_forecast"]

    def generate(self, n_scenarios: int, seed: int | None = None, **inputs) -> np.ndarray:
        # Return array of shape (n_scenarios, N_STEPS)
        ...
```

2. Register it in the generator's `__init__.py`:

```python
from src.scenarios.wind_production.my_model import MyModel

class WindProductionGenerator(ScenarioGenerator):
    _models = {
        ...,
        MyModel.name: MyModel,
    }
    _default_model_paths = {
        ...,
        "my_model": "trained_models/wind_generation/my_model.pt",  # optional default
    }
```

3. Train the model in a notebook under `model_training/` and save weights to
   `trained_models/`.

### Adding a new generator (new resource type)

1. Create a new folder `src/scenarios/my_resource/` with an `__init__.py` and
   one `.py` file per model (following the pattern above).

2. Register the generator in `src/scenarios/__init__.py`:

```python
from src.scenarios.my_resource import MyResourceGenerator
```