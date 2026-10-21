"""On-demand forecasts for a browser's approximate location, without archiving GPS."""
from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from threading import RLock
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import pandas as pd

from climate_app.forecast.analysis import combine
from climate_app.forecast.domain import Location, ProviderError, finite, instant, utcnow
from climate_app.forecast.providers import OPEN_MODELS, fetch_metno, fetch_open_meteo
from climate_app.forecast.store import POINT_COLUMNS

AUTOMATIC = "__device_location__"


def device_location(payload, now=None):
    """Validate untrusted browser values and reduce precision before provider calls."""
    now = now or utcnow()
    if not isinstance(payload, dict):
        raise ValueError("No se ha podido obtener la ubicación")
    lat, lon = finite(payload.get("latitude")), finite(payload.get("longitude"))
    accuracy, timestamp = finite(payload.get("accuracy")), finite(payload.get("timestamp"))
    if lat is None or lon is None or not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError("El dispositivo no ha devuelto una ubicación válida")
    if accuracy is None or not 0 <= accuracy <= 10_000:
        raise ValueError("La ubicación es demasiado imprecisa. Activa la ubicación precisa o elige un lugar guardado")
    if timestamp is None or not -60 <= now.timestamp() - timestamp / 1000 <= 600:
        raise ValueError("La ubicación ha caducado. Vuelve a localizar el dispositivo")
    timezone = payload.get("timezone", "UTC")
    try:
        if not isinstance(timezone, str) or len(timezone) > 80:
            raise ValueError
        ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        timezone = "UTC"
    lat, lon = round(lat, 2), round(lon, 2)
    identity = hashlib.sha256(f"{lat:.2f},{lon:.2f},{timezone}".encode()).hexdigest()[:16]
    return Location("auto:" + identity, "Ubicación actual", lat, lon, timezone)


class SessionCache:
    """HTTP data stays in this web session; only the global quota is persistent."""
    def __init__(self, shared_store, cache):
        self.shared_store, self.cache = shared_store, cache
        self.lock = RLock()

    def state(self, key, default=None):
        with self.lock:
            return self.cache.get(key, default)

    def set_state(self, key, value):
        with self.lock:
            self.cache.pop(key, None)
            self.cache[key] = value
            while len(self.cache) > 16:
                self.cache.pop(next(iter(self.cache)))

    def reserve(self, *args):
        return self.shared_store.reserve(*args)


def forecast_frame(location, forecasts):
    rows = []
    for forecast in forecasts:
        for point in forecast.points:
            rows.append({**{key: None for key in POINT_COLUMNS}, **point,
                         "location_id": location.id, "provider": forecast.provider,
                         "fetched_at": forecast.fetched_at, "init_time": forecast.init_time,
                         "metadata": json.dumps(forecast.metadata)})
    frame = pd.DataFrame(rows)
    if not frame.empty:
        for key in ("valid_time", "fetched_at", "init_time"):
            frame[key] = pd.to_datetime(frame[key], utc=True)
        for key in POINT_COLUMNS:
            frame[key] = pd.to_numeric(frame[key], errors="coerce")
    return frame


def automatic_forecasts(location, shared_store, session, now=None):
    """Use the existing free sources and quota, never the shared place/archive tables."""
    now = now or utcnow()
    previous = session.get("automatic_weather")
    if previous and previous["location_id"] == location.id and instant(previous["retry_at"]) > now:
        return previous["frame"], previous["errors"]
    cache = SessionCache(shared_store, session.setdefault("automatic_http_cache", {}))

    def fetch(provider):
        try:
            result = fetch_metno(location, cache) if provider == "metno" else fetch_open_meteo(provider, location, cache)
            return result, None
        except Exception as exc:
            message = str(exc) if isinstance(exc, ProviderError) else "La fuente no ha respondido"
            return None, message

    providers = [*OPEN_MODELS, "metno"]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(fetch, providers))
    frame = forecast_frame(location, [result for result, _ in results if result is not None])
    errors = {provider: error for provider, (_, error) in zip(providers, results) if error}
    if not frame.empty:
        consensus = combine(frame, as_of=now)
        if not consensus.empty:
            consensus["metadata"] = json.dumps({"trained": False, "temporary_location": True})
            frame = pd.concat([frame, consensus], ignore_index=True)
        frame = frame.sort_values("valid_time").reset_index(drop=True)
    retry = timedelta(minutes=15 if errors else 180) if not frame.empty else timedelta(minutes=2)
    session["automatic_weather"] = {"location_id": location.id, "frame": frame, "errors": errors,
                                    "retry_at": (now + retry).isoformat()}
    return frame, errors
