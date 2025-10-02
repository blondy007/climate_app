from datetime import date
from typing import Dict, List

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype

from climate_app.shared.constants import MONTH_NAME_TO_NUM
from climate_app.shared.utils import format_metric

AGG_FUNCTIONS = {
    "mediana": pd.Series.median,
    "media": pd.Series.mean,
    "p25": lambda series: series.quantile(0.25),
    "p75": lambda series: series.quantile(0.75),
    "min": pd.Series.min,
    "max": pd.Series.max,
}

AGG_LABELS = {
    "mediana": "Mediana",
    "media": "Media",
    "p25": "Percentil 25",
    "p75": "Percentil 75",
    "min": "Mínimo",
    "max": "Máximo",
}


def determine_granularity(start: date, end: date) -> str:
    span_days = (end - start).days + 1
    if span_days <= 2:
        return "hourly"
    if span_days <= 31:
        return "daily"
    return "monthly"


def format_axis_values(series: pd.Series, granularity: str) -> List[str]:
    if series.empty:
        return []
    if granularity == "monthly":
        return series.astype(str).tolist()
    converted = pd.to_datetime(series)
    if granularity == "daily":
        return converted.dt.strftime("%Y-%m-%d").tolist()
    if converted.dt.tz is None:
        converted = converted.dt.tz_localize("UTC")
    return converted.dt.tz_convert("Europe/Madrid").dt.strftime("%Y-%m-%d %H:%M").tolist()


def summarize_temperatures(series: pd.Series, agg_func) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    return float(agg_func(values)) if not values.empty else float("nan")


def summarize_pair(group: pd.DataFrame, agg_func) -> pd.Series:
    temp_value = summarize_temperatures(group["temp"], agg_func)
    feel_value = summarize_temperatures(group["feels_like"], agg_func)

    temps = pd.to_numeric(group["temp"], errors="coerce")
    feels = pd.to_numeric(group["feels_like"], errors="coerce")
    diff_mask = temps.notna() & feels.notna() & (np.abs(temps - feels) > 0.1)
    if diff_mask.any():
        subset = feels[diff_mask].dropna()
        if not subset.empty:
            try:
                adjusted = agg_func(subset)
                if not pd.isna(adjusted):
                    feel_value = adjusted
            except Exception:  # noqa: BLE001
                pass

    humidity_value = np.nan
    if "humedad_%" in group.columns:
        humidity_value = summarize_temperatures(group["humedad_%"], agg_func)

    wind_value = np.nan
    if "wspd_kmh" in group.columns:
        wind_value = summarize_temperatures(group["wspd_kmh"], agg_func)

    return pd.Series(
        {
            "Temp (°C)": temp_value,
            "Se siente (°C)": feel_value,
            "Humedad_%": humidity_value,
            "Viento (km/h)": wind_value,
        }
    )


def _empty_frame(time_label: str, has_humidity: bool, has_wind: bool) -> pd.DataFrame:
    columns = [time_label, "Lugar", "Temp (°C)", "Se siente (°C)"]
    if has_humidity:
        columns.append("Humedad_%")
    if has_wind:
        columns.append("Viento (km/h)")
    return pd.DataFrame(columns=columns)


def aggregate_series(df: pd.DataFrame, agg: str, granularity: str) -> pd.DataFrame:
    agg_func = AGG_FUNCTIONS.get(agg, pd.Series.median)

    has_humidity = "humedad_%" in df.columns
    has_wind = "wspd_kmh" in df.columns

    if df.empty:
        if granularity == "monthly":
            return _empty_frame("Mes", has_humidity, has_wind)
        if granularity == "daily":
            return _empty_frame("Fecha", has_humidity, has_wind)
        return _empty_frame("FechaHora", has_humidity, has_wind)

    if "Lugar" in df.columns and isinstance(df["Lugar"].dtype, pd.CategoricalDtype):
        cities_categories = list(df["Lugar"].cat.categories)
    else:
        cities_categories = list(dict.fromkeys(df.get("Lugar", pd.Series(dtype=str)).dropna()))

    def apply_city_categories(result: pd.DataFrame) -> pd.DataFrame:
        if cities_categories:
            result["Lugar"] = pd.Categorical(result["Lugar"], categories=cities_categories, ordered=False)
        return result

    def reorder_columns(result: pd.DataFrame, time_label: str) -> pd.DataFrame:
        columns = [time_label, "Lugar", "Temp (°C)", "Se siente (°C)"]
        if has_humidity and "Humedad_%" in result.columns:
            columns.append("Humedad_%")
        if has_wind and "Viento (km/h)" in result.columns:
            columns.append("Viento (km/h)")
        for column in columns:
            if column not in result.columns:
                result[column] = np.nan
        return result.loc[:, columns]

    if granularity == "monthly":
        if "Mes" in df.columns and isinstance(df["Mes"].dtype, CategoricalDtype):
            months_categories = list(df["Mes"].cat.categories)
        else:
            months_categories = sorted(
                df.get("Mes", pd.Series(dtype=str)).dropna().unique(),
                key=lambda name: MONTH_NAME_TO_NUM.get(name, 13),
            )
        grouped = (
            df.groupby(["Mes", "Lugar"], observed=True, group_keys=False)
            .apply(summarize_pair, agg_func)
            .reset_index()
        )
        if months_categories:
            grouped["Mes"] = pd.Categorical(grouped["Mes"], categories=months_categories, ordered=True)
        grouped = apply_city_categories(grouped)
        grouped = grouped.sort_values(["Mes", "Lugar"]).reset_index(drop=True)
        return reorder_columns(grouped, "Mes")

    if granularity == "daily":
        grouped = (
            df.groupby(["Fecha", "Lugar"], observed=True, group_keys=False)
            .apply(summarize_pair, agg_func)
            .reset_index()
        )
        grouped["Fecha"] = pd.to_datetime(grouped["Fecha"])
        grouped = apply_city_categories(grouped)
        grouped = grouped.sort_values(["Fecha", "Lugar"]).reset_index(drop=True)
        return reorder_columns(grouped, "Fecha")

    df_hourly = df.copy()
    df_hourly["FechaHora"] = df_hourly["datetime"].dt.tz_convert("Europe/Madrid")
    grouped = (
        df_hourly.groupby(["FechaHora", "Lugar"], observed=True, group_keys=False)
        .apply(summarize_pair, agg_func)
        .reset_index()
    )
    grouped["FechaHora"] = pd.to_datetime(grouped["FechaHora"])
    grouped = apply_city_categories(grouped)
    grouped = grouped.sort_values(["FechaHora", "Lugar"]).reset_index(drop=True)
    return reorder_columns(grouped, "FechaHora")


def kpis(df: pd.DataFrame, agg: str) -> pd.DataFrame:
    agg_func = AGG_FUNCTIONS.get(agg, pd.Series.median)

    cities: List[str] = []
    if "Lugar" in df.columns:
        if isinstance(df["Lugar"].dtype, pd.CategoricalDtype):
            cities = list(df["Lugar"].cat.categories)
        else:
            cities = list(dict.fromkeys(df["Lugar"].dropna()))

    has_humidity = "humedad_%" in df.columns
    has_wind = "wspd_kmh" in df.columns

    if df.empty:
        base = {
            "Lugar": cities,
            "Temp (°C)": [np.nan] * len(cities),
            "Se siente (°C)": [np.nan] * len(cities),
        }
        base["Humedad_%"] = [np.nan] * len(cities)
        base["Viento (km/h)"] = [np.nan] * len(cities)
        return pd.DataFrame(base)

    agg_kwargs: Dict[str, tuple] = {
        "temp": ("temp", agg_func),
        "feels_like": ("feels_like", agg_func),
    }
    if has_humidity:
        agg_kwargs["humedad"] = ("humedad_%", agg_func)
    if has_wind:
        agg_kwargs["viento"] = ("wspd_kmh", agg_func)

    result = (
        df.groupby("Lugar", observed=True)
        .agg(**agg_kwargs)
        .reset_index()
    )

    rename_map = {
        "temp": "Temp (°C)",
        "feels_like": "Se siente (°C)",
    }
    if has_humidity and "humedad" in result.columns:
        rename_map["humedad"] = "Humedad_%"
    if has_wind and "viento" in result.columns:
        rename_map["viento"] = "Viento (km/h)"

    result = result.rename(columns=rename_map)

    for column in ["Humedad_%", "Viento (km/h)"]:
        if column not in result.columns:
            result[column] = np.nan

    if cities:
        result["Lugar"] = pd.Categorical(result["Lugar"], categories=cities, ordered=False)
        result = result.sort_values("Lugar").reset_index(drop=True)

    return result


def build_summary_markdown(
    period_label: str,
    agg_label: str,
    filters: Dict[str, str],
    kpi_data: pd.DataFrame,
    missing_logs: List[str],
    total_rows: int,
) -> str:
    lines: List[str] = ["# Resumen Meteostat", ""]
    lines.append(f"- Periodo analizado: {period_label}")
    lines.append(f"- Agregación aplicada: {agg_label}")
    lines.append(f"- Registros visibles tras filtros: {total_rows}")
    lines.append("")
    lines.append("## Filtros activos")
    for label, value in filters.items():
        lines.append(f"- **{label}**: {value}")

    lines.append("")
    lines.append("## Indicadores por ciudad")
    if kpi_data.empty:
        lines.append("Sin datos para calcular indicadores con los filtros actuales.")
    else:
        for _, row in kpi_data.iterrows():
            lines.append(
                f"- **{row['Lugar']}**: Temp {format_metric(row['Temp (°C)'])} °C · "
                f"Se siente {format_metric(row['Se siente (°C)'])} °C · "
                f"Humedad {format_metric(row['Humedad_%'])} % · "
                f"Viento {format_metric(row['Viento (km/h)'])} km/h"
            )

    if missing_logs:
        lines.append("")
        lines.append("## Incidencias de datos")
        for message in sorted(set(missing_logs)):
            lines.append(f"- {message}")

    return "\n".join(lines)
