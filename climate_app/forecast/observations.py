from __future__ import annotations

from datetime import timedelta
from io import BytesIO
import os
import re
import time

import pandas as pd
import requests
from climate_app.data.meteostat_config import Hourly, Stations
from climate_app.data.paths import DATA_DIR

from climate_app.forecast.domain import ProviderError, distance_km, utcnow


def resolve_station(location, store):
    if location.station_id and location.station_distance_km is not None:
        return location
    catalog = Stations()
    if location.station_id:
        metadata = catalog.fetch()
        candidates = metadata.loc[metadata.index.astype(str) == location.station_id]
    else:
        candidates = catalog.nearby(location.latitude, location.longitude, radius=50000).inventory("hourly").fetch(10)
        if "hourly_end" in candidates:
            recent = candidates[pd.to_datetime(candidates.hourly_end) >= pd.Timestamp(utcnow().date() - timedelta(days=30))]
            if not recent.empty:
                candidates = recent
    if candidates.empty:
        raise ProviderError("No hay estación Meteostat cercana; la previsión sigue disponible sin evaluación observacional")
    station = candidates.iloc[0]
    station_id = str(candidates.index[0])
    distance = distance_km(location.latitude, location.longitude, float(station.latitude), float(station.longitude))
    store.set_station(location.id, station_id, str(station["name"]), round(distance, 1))
    return next(item for item in store.locations() if item.id == location.id)


OBSERVED_SOURCES = {"metar", "synop", "isd_lite", "isd", "dwd", "dwd_cdc", "dwd_poi", "ghcn"}
OBSERVATION_COLUMNS = {"temp": "temperature", "rhum": "humidity", "prcp": "precipitation",
                       "wspd": "wind_speed", "wdir": "wind_direction"}


def parse_observation_csv(data, start, end):
    """Mask model substitutes per variable using the modern dump's source columns."""
    data = data.copy()
    data["valid_time"] = pd.to_datetime(data[["year", "month", "day", "hour"]], utc=True)
    data = data[(data.valid_time >= pd.Timestamp(start)) & (data.valid_time <= pd.Timestamp(end))]
    result = data[["valid_time"]].copy()
    for column, target in OBSERVATION_COLUMNS.items():
        if column in data and column + "_source" in data:
            result[target] = pd.to_numeric(data[column], errors="coerce").where(data[column + "_source"].isin(OBSERVED_SOURCES))
        else:
            result[target] = float("nan")
    return result.dropna(subset=list(OBSERVATION_COLUMNS.values()), how="all").to_dict("records")


def fetch_observations(location, start, end):
    if not location.station_id or not re.fullmatch(r"[A-Za-z0-9]{3,8}", location.station_id):
        raise ProviderError("Identificador de estación no válido")
    cache = DATA_DIR / "meteostat-observations"
    cache.mkdir(parents=True, exist_ok=True)
    points = []
    for year in range(start.year, end.year + 1):
        path = cache / f"{location.station_id}-{year}.csv.gz"
        ttl = 6 * 3600 if year == utcnow().year else 30 * 86400
        if path.exists() and time.time() - path.stat().st_mtime < ttl:
            content = path.read_bytes()
        else:
            try:
                response = requests.get(f"https://data.meteostat.net/hourly/{year}/{location.station_id}.csv.gz",
                    headers={"User-Agent": "ClimateApp/2.0 (https://github.com/blondy007/climate_app)"}, timeout=(5, 35))
            except requests.RequestException:
                raise ProviderError("Meteostat no responde; se conservarán las observaciones anteriores") from None
            if response.status_code == 404:
                continue
            if response.status_code != 200:
                raise ProviderError(f"Meteostat temporalmente no disponible (HTTP {response.status_code})")
            content = response.content
            temporary = path.with_suffix(".gz.tmp")
            temporary.write_bytes(content)
            temporary.replace(path)
        data = pd.read_csv(BytesIO(content), compression="gzip")
        points.extend(parse_observation_csv(data, start, end))
    if not points:
        raise ProviderError("La estación todavía no ha publicado observaciones para este periodo")
    return points
