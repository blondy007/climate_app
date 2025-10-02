from typing import Dict

import numpy as np
import pandas as pd

from climate_app.shared.constants import MONTH_NAMES, STATION_NAME_MAP
from climate_app.shared.utils import canonical_station_id


def compute_wind_chill(temp_c: float, wind_kmh: float) -> float:
    if pd.isna(temp_c):
        return np.nan
    if pd.isna(wind_kmh):
        return temp_c
    if temp_c <= 10 and wind_kmh > 4.8:
        wind_factor = wind_kmh ** 0.16
        return 13.12 + 0.6215 * temp_c - 11.37 * wind_factor + 0.3965 * temp_c * wind_factor
    return temp_c


def compute_heat_index(temp_c: float, humidity: float) -> float:
    if pd.isna(temp_c) or pd.isna(humidity):
        return np.nan
    humidity = float(np.clip(humidity, 0.0, 100.0))
    temp_f = (temp_c * 9 / 5) + 32
    hi_f = (
        -42.379
        + 2.04901523 * temp_f
        + 10.14333127 * humidity
        - 0.22475541 * temp_f * humidity
        - 6.83783e-3 * temp_f**2
        - 5.481717e-2 * humidity**2
        + 1.22874e-3 * temp_f**2 * humidity
        + 8.5282e-4 * temp_f * humidity**2
        - 1.99e-6 * temp_f**2 * humidity**2
    )
    if humidity < 13 and 80 <= temp_f <= 112:
        hi_f -= ((13 - humidity) / 4) * np.sqrt((17 - abs(temp_f - 95)) / 17)
    elif humidity > 85 and 80 <= temp_f <= 87:
        hi_f += ((humidity - 85) / 10) * ((87 - temp_f) / 5)
    return (hi_f - 32) * 5 / 9


def compute_feels_like(temp_c: float, wind_kmh: float, humidity: float) -> float:
    if pd.isna(temp_c):
        return np.nan
    if not pd.isna(wind_kmh) and temp_c <= 10 and wind_kmh > 4.8:
        return compute_wind_chill(temp_c, wind_kmh)
    if not pd.isna(humidity) and temp_c >= 27 and humidity >= 40:
        return compute_heat_index(temp_c, humidity)
    return temp_c


def _infer_month_categories(month_series: pd.Series) -> pd.Categorical:
    month_values = [
        MONTH_NAMES.get(int(month), str(int(month)))
        for month in sorted(month_series.dropna().unique())
    ]
    return pd.Categorical(month_series.map(lambda month: MONTH_NAMES.get(int(month), str(int(month)))), categories=month_values, ordered=True)


def normalize_meteostat_df(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        columns = [
            "datetime",
            "Fecha",
            "Hora",
            "MesNum",
            "Mes",
            "station",
            "Lugar",
            "temp",
            "humedad_%",
            "wspd",
            "wspd_kmh",
            "feels_like",
        ]
        return pd.DataFrame(columns=columns)

    df_norm = df.copy()
    df_norm.columns = [str(col).strip().lower().replace(" ", "_") for col in df_norm.columns]

    legacy_cols = ["lugar", "mesnum", "mes", "hora", "fecha"]
    drop_cols = [col for col in legacy_cols if col in df_norm.columns]
    if drop_cols:
        df_norm = df_norm.drop(columns=drop_cols)

    time_col = None
    for candidate in ("datetime", "time", "date_time", "fecha", "fecha_hora"):
        if candidate in df_norm.columns:
            time_col = candidate
            break
    if time_col is None:
        raise KeyError("No se encontró columna temporal compatible")

    if time_col != "datetime":
        df_norm = df_norm.rename(columns={time_col: "datetime"})

    df_norm["datetime"] = pd.to_datetime(df_norm["datetime"].astype(str).str.strip(), errors="coerce", utc=True)
    df_norm = df_norm.dropna(subset=["datetime"]).copy()
    df_norm["datetime"] = df_norm["datetime"].dt.tz_convert("Europe/Madrid")

    if "station" not in df_norm.columns:
        raise KeyError("Falta columna station en el CSV")
    df_norm["station"] = df_norm["station"].astype(str).str.strip().apply(canonical_station_id)

    if "temp" not in df_norm.columns:
        raise KeyError("Falta columna temp en el CSV")

    if "wspd" not in df_norm.columns:
        df_norm["wspd"] = np.nan

    if "humedad_%" in df_norm.columns:
        df_norm["humedad_%"] = pd.to_numeric(df_norm["humedad_%"], errors="coerce")
    else:
        hum_col = next((c for c in ("rhum", "rh") if c in df_norm.columns), None)
        if hum_col is not None:
            df_norm["humedad_%"] = pd.to_numeric(df_norm[hum_col], errors="coerce")
        else:
            df_norm["humedad_%"] = np.nan

    temp_numeric = pd.to_numeric(df_norm["temp"], errors="coerce")
    wspd_numeric = pd.to_numeric(df_norm["wspd"], errors="coerce")
    humidity_numeric = pd.to_numeric(df_norm["humedad_%"], errors="coerce")

    df_norm["wspd_kmh"] = wspd_numeric * 3.6
    df_norm["feels_like"] = [
        compute_feels_like(temp, wind, humidity)
        for temp, wind, humidity in zip(temp_numeric, df_norm["wspd_kmh"], humidity_numeric)
    ]

    df_norm["Lugar"] = df_norm["station"].map(STATION_NAME_MAP).fillna(df_norm.get("name", df_norm["station"]))

    df_norm["MesNum"] = df_norm["datetime"].dt.month
    df_norm["Mes"] = df_norm["MesNum"].map(lambda month: MONTH_NAMES.get(int(month), str(int(month))))
    month_values = [MONTH_NAMES.get(int(month), str(int(month))) for month in sorted(df_norm["MesNum"].dropna().unique())]
    df_norm["Mes"] = pd.Categorical(df_norm["Mes"], categories=month_values, ordered=True)
    df_norm["Hora"] = df_norm["datetime"].dt.strftime("%H:%M")
    df_norm["Fecha"] = df_norm["datetime"].dt.date

    df_norm = df_norm.loc[:, ~df_norm.columns.duplicated()]
    return df_norm
