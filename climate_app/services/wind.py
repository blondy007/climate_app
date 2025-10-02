from typing import Dict, List

import pandas as pd

from climate_app.shared.constants import DEFAULT_THRESHOLDS, SCENARIO_CATEGORIES


def build_thresholds(cities: List[str], stored: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for city in cities:
        base = stored.get(city) or DEFAULT_THRESHOLDS.get(city) or {"calma": 10.0, "ventoso": 25.0}
        result[city] = {"calma": float(base["calma"]), "ventoso": float(base["ventoso"])}
    return result


def classify_row_wind(row: pd.Series, thresholds_map: Dict[str, Dict[str, float]]) -> str:
    wind = row.get("wspd_kmh")
    if pd.isna(wind):
        return "Sin dato"
    city = row.get("Lugar")
    thresholds = thresholds_map.get(city)
    if thresholds is None:
        return "Sin dato"
    if wind <= thresholds["calma"]:
        return "Calma"
    if wind <= thresholds["ventoso"]:
        return "Ventoso"
    return "Viento fuerte"


def apply_wind_class(df: pd.DataFrame, thresholds_map: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    classified = df.copy()
    if not classified.empty:
        classified["escenario"] = classified.apply(classify_row_wind, axis=1, args=(thresholds_map,))
        classified["escenario"] = pd.Categorical(
            classified["escenario"], categories=SCENARIO_CATEGORIES, ordered=False
        )
    return classified
