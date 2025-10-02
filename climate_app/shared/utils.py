from typing import Optional

import pandas as pd

from .constants import STATION_ID_OVERRIDES


def canonical_station_id(station_id: Optional[str]) -> str:
    """Normaliza IDs de estación para alinearlas con Meteostat."""
    value = "" if station_id is None else str(station_id).strip()
    if not value:
        return value
    key = value.lower()
    if key in STATION_ID_OVERRIDES:
        return STATION_ID_OVERRIDES[key]
    if "_" in value:
        suffix = value.split("_")[-1].strip()
        if suffix:
            return suffix.upper()
    return value.upper()


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1f}"
