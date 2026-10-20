from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

UTC = timezone.utc
VARIABLES = ("temperature", "humidity", "precipitation", "wind_speed", "wind_direction",
             "wind_gust", "cloud_cover", "precip_probability")
LABELS = {
    "climateapp_private": "ClimateApp · consenso con WeatherNext",
    "climateapp": "ClimateApp · consenso",
    "ecmwf": "ECMWF IFS · Open-Meteo",
    "icon": "DWD ICON · Open-Meteo",
    "gfs": "NOAA GFS · Open-Meteo",
    "metno": "MET Norway",
    "weathernext3": "Google WeatherNext 3",
}
HORIZONS = ((6, "0–6 h"), (24, "6–24 h"), (48, "24–48 h"),
            (120, "2–5 días"), (240, "5–10 días"), (400, "10–16 días"))


def utcnow() -> datetime:
    return datetime.now(UTC)


def stamp(value: datetime | str) -> str:
    if isinstance(value, str):
        value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None:
        raise ValueError("Se requiere fecha con zona horaria")
    return value.astimezone(UTC).isoformat(timespec="seconds")


def instant(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def horizon(hours: float) -> str:
    return next((label for upper, label in HORIZONS if hours <= upper), HORIZONS[-1][1])


def finite(value) -> float | None:
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (ValueError, TypeError):
        return None


def distance_km(lat1, lon1, lat2, lon2) -> float:
    a, b = math.radians(lat1), math.radians(lat2)
    delta_lat, delta_lon = b - a, math.radians(lon2 - lon1)
    h = math.sin(delta_lat / 2) ** 2 + math.cos(a) * math.cos(b) * math.sin(delta_lon / 2) ** 2
    return 6371 * 2 * math.asin(min(1, math.sqrt(h)))


@dataclass(frozen=True)
class Location:
    id: str
    name: str
    latitude: float
    longitude: float
    timezone: str = "Europe/Madrid"
    station_id: str | None = None
    station_name: str | None = None
    station_distance_km: float | None = None


@dataclass
class Forecast:
    provider: str
    fetched_at: datetime
    points: list[dict]
    init_time: datetime | None = None
    metadata: dict = field(default_factory=dict)


class ProviderError(RuntimeError):
    """A safe user-facing provider failure (never include credentials)."""
