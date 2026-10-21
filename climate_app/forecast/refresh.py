"""Explicit refresh of one saved place using the existing free-source budget."""
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

from climate_app.forecast.domain import ProviderError, instant, stamp, utcnow
from climate_app.forecast.providers import OPEN_MODELS, fetch_metno, fetch_open_meteo
from climate_app.forecast.worker import rebuild_consensus


@dataclass
class RefreshResult:
    updated: list[str] = field(default_factory=list)
    cached: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)
    retry_at: datetime | None = None


class RefreshCache:
    def __init__(self, store, location_id, now):
        self.store, self.location_id, self.now = store, location_id, now
        self.cached = set()

    def state(self, key, default=None):
        value = self.store.state(key, default)
        provider = key.split(":")[1] if key.startswith("http:") else None
        if not value or key != f"http:{provider}:{self.location_id}":
            return value
        expires = self.now - timedelta(seconds=1)
        if provider == "metno":
            # Honor the upstream Expires header; old cache entries stay valid until
            # their existing expiry because their upstream deadline is unknown.
            try:
                expires = (parsedate_to_datetime(value["provider_expires"]) if value.get("provider_expires")
                           else instant(value["expires"]))
            except (ValueError, TypeError, KeyError):
                expires = instant(value["expires"])
            if expires > self.now:
                self.cached.add(provider)
        return {**value, "expires": stamp(expires)}

    def set_state(self, *args):
        return self.store.set_state(*args)

    def reserve(self, *args):
        return self.store.reserve(*args)


def refresh_saved_forecast(store, location_id, now=None):
    now = now or utcnow()
    retry_at = store.claim_manual_refresh(location_id, now)
    if retry_at:
        return RefreshResult(retry_at=retry_at)
    location = next((loc for loc in store.locations() if loc.id == location_id), None)
    if location is None:
        raise ValueError("El lugar ya no está activo")
    result = RefreshResult()
    cache = RefreshCache(store, location_id, now)
    def fetch(provider):
        try:
            forecast = (fetch_metno(location, cache) if provider == "metno"
                        else fetch_open_meteo(provider, location, cache))
            return provider, forecast, None
        except Exception as exc:
            message = str(exc) if isinstance(exc, ProviderError) else "La fuente no ha respondido"
            return provider, None, message

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, (*OPEN_MODELS, "metno")))
    for provider, forecast, error in results:
        if error:
            result.errors[provider] = error
            continue
        try:
            if provider in cache.cached:
                result.cached.append(provider)
                continue
            store.save_forecast(location_id, forecast)
            result.updated.append(provider)
            store.set_state(f"task:{location_id}:{provider}", {
                "status": "ok", "updated_at": stamp(forecast.fetched_at),
                "next_due": stamp(now + timedelta(hours=3)), "count": len(forecast.points)})
        except Exception as exc:
            result.errors[provider] = str(exc) if isinstance(exc, ProviderError) else "La fuente no ha respondido"
    if result.updated:
        rebuild_consensus(store, location, include_private=False)
    return result
