from __future__ import annotations

from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime

import requests

from climate_app.forecast.domain import Forecast, Location, ProviderError, finite, instant, stamp, utcnow
from climate_app.forecast.store import Store

USER_AGENT = "ClimateApp/2.0 (https://github.com/blondy007/climate_app)"
OPEN_MODELS = {
    "ecmwf": ("ecmwf", "ecmwf_ifs", 15),
    "icon": ("dwd-icon", "icon_seamless", 7),
    "gfs": ("gfs", "gfs_seamless", 16),
}
OPEN_VARIABLES = {
    "temperature_2m": "temperature", "relative_humidity_2m": "humidity",
    "precipitation": "precipitation", "wind_speed_10m": "wind_speed",
    "wind_direction_10m": "wind_direction", "wind_gusts_10m": "wind_gust",
    "cloud_cover": "cloud_cover",
    "precipitation_probability": "precip_probability",
}


def get_json(url, params, store: Store, cache_key: str, min_ttl=3600):
    now = utcnow()
    cached = store.state("http:" + cache_key)
    if cached and instant(cached["expires"]) > now:
        return cached["data"]
    if "open-meteo.com" in url and not store.reserve("open-meteo", now.date().isoformat(), 32, 8000):
        raise ProviderError("Cuota preventiva diaria gratuita agotada; se conserva la última previsión")
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if cached and cached.get("last_modified"):
        headers["If-Modified-Since"] = cached["last_modified"]
    try:
        response = requests.get(url, params=params, headers=headers, timeout=(5, 35))
        if response.status_code == 304 and cached:
            data = cached["data"]
        else:
            if response.status_code != 200:
                raise ProviderError(f"Proveedor temporalmente no disponible (HTTP {response.status_code})")
            data = response.json()
        expires = now + timedelta(seconds=min_ttl)
        if response.headers.get("Expires"):
            try:
                expires = max(expires, parsedate_to_datetime(response.headers["Expires"]))
            except (ValueError, TypeError):
                pass
        store.set_state("http:" + cache_key, {"data": data, "expires": stamp(expires),
                        "provider_expires": response.headers.get("Expires"),
                        "last_modified": response.headers.get("Last-Modified")})
        return data
    except requests.RequestException:
        raise ProviderError("No se pudo conectar con el proveedor; se reintentará automáticamente") from None
    except (ValueError, KeyError):
        raise ProviderError("El proveedor devolvió una respuesta inválida") from None


def parse_open_meteo(provider: str, data: dict, fetched_at=None) -> Forecast:
    hourly = data.get("hourly", {})
    points = []
    for index, time in enumerate(hourly.get("time", [])):
        point = {"valid_time": datetime.fromtimestamp(time, tz=utcnow().tzinfo)}
        for source, target in OPEN_VARIABLES.items():
            values = hourly.get(source, [])
            point[target] = finite(values[index]) if index < len(values) else None
        if point["temperature"] is not None:
            points.append(point)
    if not points:
        raise ProviderError("Sin datos de este modelo para las coordenadas solicitadas")
    return Forecast(provider, fetched_at or utcnow(), points, metadata={
        "source": "Open-Meteo", "model": OPEN_MODELS[provider][1],
        "grid_latitude": data.get("latitude"), "grid_longitude": data.get("longitude"),
        "timing": "snapshot", "precipitation_interval": "preceding_hour",
        "license": "CC BY 4.0", "interpolation": "La API puede interpolar los pasos nativos a una hora",
    })


def fetch_open_meteo(provider: str, location: Location, store: Store) -> Forecast:
    endpoint, model, days = OPEN_MODELS[provider]
    data = get_json(f"https://api.open-meteo.com/v1/{endpoint}", {
        "latitude": location.latitude, "longitude": location.longitude,
        "hourly": ",".join(key for key in OPEN_VARIABLES if key != "precipitation_probability" or provider == "gfs"), "models": model,
        "forecast_days": days, "timeformat": "unixtime", "timezone": "UTC",
        "wind_speed_unit": "kmh", "temperature_unit": "celsius", "precipitation_unit": "mm",
    }, store, f"{provider}:{location.id}", min_ttl=3 * 3600)
    return parse_open_meteo(provider, data)


def parse_metno(data: dict, fetched_at=None) -> Forecast:
    by_time = {}
    series = data.get("properties", {}).get("timeseries", [])
    for item in series:
        when = instant(item["time"])
        detail = item["data"].get("instant", {}).get("details", {})
        point = by_time.setdefault(stamp(when), {"valid_time": when})
        for source, target in {"air_temperature": "temperature", "relative_humidity": "humidity",
                               "wind_from_direction": "wind_direction", "cloud_area_fraction": "cloud_cover"}.items():
            point[target] = finite(detail.get(source))
        speed, gust = finite(detail.get("wind_speed")), finite(detail.get("wind_speed_of_gust"))
        point["wind_speed"] = speed * 3.6 if speed is not None else None
        point["wind_gust"] = gust * 3.6 if gust is not None else None
        # MET precipitation is for the NEXT hour; our common convention is the preceding hour.
        following = item["data"].get("next_1_hours", {}).get("details", {})
        if following:
            end = when + timedelta(hours=1)
            rain = by_time.setdefault(stamp(end), {"valid_time": end})
            rain["precipitation"] = finite(following.get("precipitation_amount"))
            rain["precip_probability"] = finite(following.get("probability_of_precipitation"))
    points = sorted(by_time.values(), key=lambda p: p["valid_time"])
    if not points:
        raise ProviderError("MET Norway no devolvió datos para este lugar")
    updated = data.get("properties", {}).get("meta", {}).get("updated_at")
    return Forecast("metno", fetched_at or utcnow(), points,
                    metadata={"source": "MET Norway", "source_updated_at": updated,
                              "timing": "snapshot", "precipitation_interval": "preceding_hour",
                              "license": "CC BY 4.0", "sparse_after_48h": True})


def fetch_metno(location: Location, store: Store) -> Forecast:
    data = get_json("https://api.met.no/weatherapi/locationforecast/2.0/compact", {
        "lat": f"{location.latitude:.4f}", "lon": f"{location.longitude:.4f}",
    }, store, f"metno:{location.id}", min_ttl=3 * 3600)
    return parse_metno(data)


def geocode(query: str, store: Store) -> list[dict]:
    query = query.strip()
    if not 2 <= len(query) <= 120:
        raise ValueError("Introduce entre 2 y 120 caracteres")
    data = get_json("https://geocoding-api.open-meteo.com/v1/search", {
        "name": query, "count": 8, "language": "es", "format": "json",
    }, store, "geocode:" + query.casefold(), min_ttl=86400)
    return data.get("results", [])
