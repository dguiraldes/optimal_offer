import requests


class BaseAPI:
    """
    Base class for all external data source clients.

    Provides shared infrastructure (session management, retry logic,
    logging) without enforcing a generic endpoint contract.
    Each subclass exposes its own named methods per endpoint.
    """

    def _get(self, url: str, params: dict, timeout: int = 10):
        """
        Internal HTTP GET with error handling.
        All clients should use this instead of calling requests directly.
        """
        response = requests.get(url, params=params, timeout=timeout)
        response.raise_for_status()
        return response.json()

    @staticmethod
    def _resolve_params(params: dict | None, model_class: type, **kwargs) -> dict:
        """
        Resolve final query params from a dict or explicit kwargs.
        When a dict is provided it is passed through directly.
        Otherwise kwargs are filtered for None values and forwarded to the
        internal params dataclass to produce the final query dict.
        """
        if isinstance(params, dict):
            return params
        filtered = {k: v for k, v in kwargs.items() if v is not None}
        return model_class(**filtered).to_query_params()