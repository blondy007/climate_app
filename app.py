import hashlib
import io
import json
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas.api.types import CategoricalDtype
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

st.set_page_config(page_title="Clima Meteostat · Comparativa", layout="wide")

DEFAULT_FILE = Path(__file__).resolve().parent / "meteostat_raw_oct-abr_torrejon08227_sanjavier08433.csv"
DATA_DIR = Path(__file__).resolve().parent / "data"
MASTER_FILE = DATA_DIR / "meteostat_master.csv"
THRESHOLDS_FILE = DATA_DIR / "wind_thresholds.json"

STATION_NAME_MAP = {
    "torrejon_08227": "Torrejón de Ardoz",
    "08227": "Torrejón de Ardoz",
    "san_javier_08433": "Los Alcázares (San Javier)",
    "08433": "Los Alcázares (San Javier)",
}

STATION_ID_OVERRIDES = {
    "torrejon_08227": "08227",
    "08227": "08227",
    "san_javier_08433": "08433",
    "08433": "08433",
}

DEFAULT_THRESHOLDS = {
    "Torrejón de Ardoz": {"calma": 10.0, "ventoso": 25.0},
    "Los Alcázares (San Javier)": {"calma": 10.0, "ventoso": 35.0},
}

MONTH_NAMES = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}

MONTH_NAME_TO_NUM = {name: number for number, name in MONTH_NAMES.items()}
SCENARIO_CATEGORIES = ["Calma", "Ventoso", "Viento fuerte", "Sin dato"]
SCENARIO_OPTIONS = SCENARIO_CATEGORIES + ["Todos"]


def agg_median(series: pd.Series) -> float:
    return series.median()


def agg_mean(series: pd.Series) -> float:
    return series.mean()


def agg_p25(series: pd.Series) -> float:
    return series.quantile(0.25)


def agg_p75(series: pd.Series) -> float:
    return series.quantile(0.75)


def agg_min(series: pd.Series) -> float:
    return series.min()


def agg_max(series: pd.Series) -> float:
    return series.max()


AGG_FUNCTIONS = {
    "mediana": agg_median,
    "media": agg_mean,
    "p25": agg_p25,
    "p75": agg_p75,
    "min": agg_min,
    "max": agg_max,
}

AGG_LABELS = {
    "mediana": "Mediana",
    "media": "Media",
    "p25": "Percentil 25",
    "p75": "Percentil 75",
    "min": "Mínimo",
    "max": "Máximo",
}

COUNTRY_OPTIONS = ["Todos", "ES", "US", "FR", "PT", "IT", "DE", "GB", "CA"]
UPLOAD_CACHE: Dict[str, bytes] = {}


def canonical_station_id(station_id: Optional[str]) -> str:
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


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1f}"


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
    hi_f = (-42.379 + 2.04901523 * temp_f + 10.14333127 * humidity
            - 0.22475541 * temp_f * humidity - 6.83783e-3 * temp_f ** 2
            - 5.481717e-2 * humidity ** 2 + 1.22874e-3 * temp_f ** 2 * humidity
            + 8.5282e-4 * temp_f * humidity ** 2 - 1.99e-6 * temp_f ** 2 * humidity ** 2)
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


# Remaining functions omitted for brevity
@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    if path.startswith("_uploaded::"):
        data_bytes = UPLOAD_CACHE.get(path)
        if data_bytes is None:
            raise FileNotFoundError("Archivo subido no disponible")
        buffer = io.BytesIO(data_bytes)
        df = pd.read_csv(buffer, low_memory=False)
    else:
        df = pd.read_csv(path, low_memory=False)
    return normalize_meteostat_df(df)


@st.cache_data(show_spinner=False)
def load_master_csv() -> pd.DataFrame:
    ensure_data_dir()
    if not MASTER_FILE.exists():
        return pd.DataFrame(columns=["station", "datetime"])
    df = pd.read_csv(MASTER_FILE, low_memory=False)
    if df.empty:
        return normalize_meteostat_df(df)
    return normalize_meteostat_df(df)


def save_master_csv(df: pd.DataFrame) -> None:
    ensure_data_dir()
    df_to_save = df.copy()
    if not df_to_save.empty:
        df_to_save = df_to_save.sort_values("datetime")
        if {"station", "datetime"}.issubset(df_to_save.columns):
            df_to_save = df_to_save.drop_duplicates(subset=["station", "datetime"], keep="last")
    df_to_save.to_csv(MASTER_FILE, index=False)
    load_master_csv.clear()


def load_wind_thresholds() -> Dict[str, Dict[str, float]]:
    ensure_data_dir()
    if not THRESHOLDS_FILE.exists():
        save_wind_thresholds(DEFAULT_THRESHOLDS)
        return {city: values.copy() for city, values in DEFAULT_THRESHOLDS.items()}
    with THRESHOLDS_FILE.open("r", encoding="utf-8-sig") as fp:
        data = json.load(fp)
    parsed: Dict[str, Dict[str, float]] = {}
    for city, values in data.items():
        calma = float(values.get("calma", 10.0))
        ventoso = float(values.get("ventoso", max(calma, 25.0)))
        parsed[city] = {"calma": calma, "ventoso": ventoso}
    return parsed


def save_wind_thresholds(data: Dict[str, Dict[str, float]]) -> None:
    ensure_data_dir()
    with THRESHOLDS_FILE.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)


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
        classified["escenario"] = pd.Categorical(classified["escenario"], categories=SCENARIO_CATEGORIES, ordered=False)
    else:
        classified["escenario"] = pd.Categorical([], categories=SCENARIO_CATEGORIES, ordered=False)
    return classified


def build_thresholds(cities: List[str], stored: Dict[str, Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    result: Dict[str, Dict[str, float]] = {}
    for city in cities:
        base = stored.get(city) or DEFAULT_THRESHOLDS.get(city) or {"calma": 10.0, "ventoso": 25.0}
        result[city] = {"calma": float(base["calma"]), "ventoso": float(base["ventoso"])}
    return result


@st.cache_data(show_spinner=False)
def search_stations(query: str, country: Optional[str]) -> pd.DataFrame:
    stations = Stations()
    if country and country != "Todos":
        stations = stations.region(country=country)
    stations = stations.inventory("hourly")
    df = stations.fetch()
    if df.empty:
        return pd.DataFrame(columns=["id", "name", "country", "latitude", "longitude", "elevation", "timezone"])
    df = df.reset_index().rename(columns={"index": "id"})
    if query:
        query_norm = query.strip().lower()
        masks = []
        for column in ("id", "name", "city", "region", "country"):
            if column in df.columns:
                masks.append(df[column].astype(str).str.lower().str.contains(query_norm, na=False))
        if masks:
            mask = masks[0]
            for extra in masks[1:]:
                mask |= extra
            df = df[mask]
    columns = ["id", "name", "country", "latitude", "longitude", "elevation", "timezone"]
    available_columns = [col for col in columns if col in df.columns]
    return df[available_columns].reset_index(drop=True).head(200)


@st.cache_data(show_spinner=False)
def get_station_metadata(station_id: str) -> Optional[Dict[str, object]]:
    canonical_id = canonical_station_id(station_id)
    stations = Stations().inventory("hourly")
    meta_df = stations.fetch()
    if meta_df.empty:
        return None
    if canonical_id in meta_df.index:
        meta_row = meta_df.loc[canonical_id]
    elif canonical_id.lower() in meta_df.index:
        meta_row = meta_df.loc[canonical_id.lower()]
    else:
        return None
    if isinstance(meta_row, pd.DataFrame):
        meta_row = meta_row.iloc[0]
    return meta_row.to_dict()


@st.cache_data(show_spinner=False)
def fetch_station_hourly(station_id: str, start: datetime, end: datetime) -> pd.DataFrame:
    canonical_id = canonical_station_id(station_id)
    data = Hourly(canonical_id, start, end, timezone="Europe/Madrid", model=False).fetch()
    if data.empty:
        return pd.DataFrame(columns=["station", "datetime"])
    df = data.reset_index().rename(columns={"time": "datetime"})
    df["station"] = canonical_id
    meta = get_station_metadata(canonical_id)
    if meta and "name" in meta:
        df["name"] = meta["name"]
    return normalize_meteostat_df(df)


def append_station_to_master(
    station_id: str,
    start: datetime,
    end: datetime,
    station_name: Optional[str] = None,
) -> int:
    df_new = fetch_station_hourly(station_id, start, end)
    if df_new.empty:
        return 0
    if "temp" not in df_new.columns or "station" not in df_new.columns:
        raise KeyError("Datos incompletos desde Meteostat")

    if station_name:
        df_new["name"] = station_name
        df_new["Lugar"] = df_new["Lugar"].where(df_new["Lugar"].notna(), station_name)

    df_new = df_new.drop_duplicates(subset=["station", "datetime"], keep="last")

    existing = load_master_csv()
    frames = [existing] if not existing.empty else []
    frames.append(df_new)
    combined = pd.concat(frames, ignore_index=True)
    combined = combined.drop_duplicates(subset=["station", "datetime"], keep="last")
    combined = normalize_meteostat_df(combined)

    existing_index = (
        pd.MultiIndex.from_frame(existing[["station", "datetime"]])
        if not existing.empty
        else pd.MultiIndex(levels=[[], []], codes=[[], []], names=["station", "datetime"])
    )
    new_index = pd.MultiIndex.from_frame(df_new[["station", "datetime"]])
    new_rows = int((~new_index.isin(existing_index)).sum())

    save_master_csv(combined)
    return new_rows
def aggregate_monthly(df: pd.DataFrame, agg: str) -> pd.DataFrame:
    agg_func = AGG_FUNCTIONS.get(agg, agg_median)

    months_categories: List[str] = []
    if "Mes" in df.columns:
        if isinstance(df["Mes"].dtype, pd.CategoricalDtype):
            months_categories = list(df["Mes"].cat.categories)
        else:
            raw_months = df["Mes"].dropna().unique().tolist()
            months_categories = sorted(raw_months, key=lambda name: MONTH_NAME_TO_NUM.get(name, 13))

    cities_categories: List[str] = []
    if "Lugar" in df.columns:
        if isinstance(df["Lugar"].dtype, pd.CategoricalDtype):
            cities_categories = list(df["Lugar"].cat.categories)
        else:
            cities_categories = list(dict.fromkeys(df["Lugar"].dropna()))

    if df.empty:
        empty = pd.DataFrame(columns=["Mes", "Lugar", "Temp (°C)", "Se siente (°C)"])
        if months_categories:
            empty["Mes"] = pd.Categorical([], categories=months_categories, ordered=True)
        if cities_categories:
            empty["Lugar"] = pd.Categorical([], categories=cities_categories, ordered=False)
        return empty

    def summarize(group: pd.DataFrame) -> pd.Series:
        temps = pd.to_numeric(group["temp"], errors="coerce")
        feels = pd.to_numeric(group["feels_like"], errors="coerce")

        temp_value = agg_func(temps.dropna()) if temps.notna().any() else np.nan
        feel_value = agg_func(feels.dropna()) if feels.notna().any() else np.nan

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

        return pd.Series({"Temp (°C)": temp_value, "Se siente (°C)": feel_value})

    grouped = df.groupby(["Mes", "Lugar"], observed=True, group_keys=False).apply(summarize, include_groups=False)
    grouped = grouped.reset_index()

    if months_categories:
        grouped["Mes"] = pd.Categorical(grouped["Mes"], categories=months_categories, ordered=True)
    if cities_categories:
        grouped["Lugar"] = pd.Categorical(grouped["Lugar"], categories=cities_categories, ordered=False)

    sort_columns = ["Mes", "Lugar"] if months_categories else ["Lugar"]
    grouped = grouped.sort_values(sort_columns).reset_index(drop=True)
    return grouped


def kpis(df: pd.DataFrame, agg: str) -> pd.DataFrame:
    agg_func = AGG_FUNCTIONS.get(agg, agg_median)

    cities: List[str] = []
    if "Lugar" in df.columns:
        if isinstance(df["Lugar"].dtype, pd.CategoricalDtype):
            cities = list(df["Lugar"].cat.categories)
        else:
            cities = list(dict.fromkeys(df["Lugar"].dropna()))

    if df.empty:
        return pd.DataFrame(
            {
                "Lugar": cities,
                "Temp (°C)": [np.nan] * len(cities),
                "Se siente (°C)": [np.nan] * len(cities),
                "Humedad_%": [np.nan] * len(cities),
                "Viento (km/h)": [np.nan] * len(cities),
            }
        )

    result = (
        df.groupby("Lugar", observed=True)
        .agg(
            temp=("temp", agg_func),
            feels_like=("feels_like", agg_func),
            humedad=("humedad_%", agg_func),
            viento=("wspd_kmh", agg_func),
        )
        .reset_index()
        .rename(
            columns={
                "temp": "Temp (°C)",
                "feels_like": "Se siente (°C)",
                "humedad": "Humedad_%",
                "viento": "Viento (km/h)",
            }
        )
    )

    if cities:
        result["Lugar"] = pd.Categorical(result["Lugar"], categories=cities, ordered=False)
        result = result.sort_values("Lugar").reset_index(drop=True)

    return result


def handle_station_feedback() -> None:
    feedback = st.session_state.pop("add_station_feedback", None)
    if not feedback:
        return
    level, message = feedback
    if level == "success":
        st.success(message)
    elif level == "info":
        st.info(message)
    elif level == "error":
        st.error(message)


def parse_date_range(value) -> Optional[tuple]:
    if isinstance(value, tuple) and len(value) == 2:
        return value
    if isinstance(value, list) and len(value) == 2:
        return value[0], value[1]
    if value:
        return (value, value)
    return None


def station_select_label(row: pd.Series) -> str:
    station_id = row.get("id")
    name = row.get("name")
    country = row.get("country")
    base = str(station_id)
    if pd.notna(name) and str(name).strip():
        base = f"{station_id} · {name}"
    if pd.notna(country) and str(country).strip():
        return f"{base} ({country})"
    return base


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
def main() -> None:
    ensure_data_dir()

    if "station_search_results" not in st.session_state:
        st.session_state["station_search_results"] = pd.DataFrame()

    stored_thresholds = load_wind_thresholds()

    master_df = load_master_csv()
    default_df = pd.DataFrame()
    if DEFAULT_FILE.exists():
        try:
            default_df = load_csv(str(DEFAULT_FILE))
        except Exception as exc:  # noqa: BLE001
            st.warning(f"No se pudo cargar el archivo por defecto: {exc}")

    combined_sources = []
    if not master_df.empty:
        combined_sources.append(master_df)
    if not default_df.empty:
        combined_sources.append(default_df)

    if combined_sources:
        combined = pd.concat(combined_sources, ignore_index=True)
        combined = combined.drop_duplicates(subset=["station", "datetime"], keep="last")
        combined = normalize_meteostat_df(combined)
    else:
        combined = master_df

    master_df = combined
    df_base = master_df.copy()

    uploaded_file = st.sidebar.file_uploader("Archivo CSV (Meteostat Hourly)", type=["csv"])
    source_label = MASTER_FILE.name if uploaded_file is None else uploaded_file.name
    data_key = str(DEFAULT_FILE)

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        data_key = f"_uploaded::{uploaded_file.name}::{file_hash}"
        UPLOAD_CACHE[data_key] = file_bytes
        try:
            df_base = load_csv(data_key)
        except Exception as exc:  # noqa: BLE001
            st.error(f"No se pudo cargar el archivo subido: {exc}")
            return
    elif df_base.empty and DEFAULT_FILE.exists():
        df_base = load_csv(str(DEFAULT_FILE))
        source_label = DEFAULT_FILE.name

    st.title("Clima Meteostat · Comparativa")
    st.caption(f"Archivo activo: {source_label}")

    handle_station_feedback()

    if df_base.empty:
        st.info("No hay datos disponibles.")

    min_date = df_base["Fecha"].min() if "Fecha" in df_base.columns and not df_base.empty else None
    max_date = df_base["Fecha"].max() if "Fecha" in df_base.columns and not df_base.empty else None

    today = date.today()
    one_year_ago = today - timedelta(days=365)
    master_min_date = master_df["Fecha"].min() if "Fecha" in master_df.columns and not master_df.empty else None
    default_min_date = default_df["Fecha"].min() if "Fecha" in default_df.columns and not default_df.empty else None

    range_start_candidates = [d for d in (master_min_date, default_min_date, one_year_ago) if d is not None]
    range_start_default = min(range_start_candidates) if range_start_candidates else one_year_ago

    base_min_date = date(1970, 1, 1)
    min_downloadable_candidates = [base_min_date]
    if master_min_date:
        min_downloadable_candidates.append(master_min_date)
    if default_min_date:
        min_downloadable_candidates.append(default_min_date)
    min_downloadable_date = min(min_downloadable_candidates)

    default_range_value = (range_start_default, today)

    months_all = (
        list(df_base["Mes"].cat.categories)
        if "Mes" in df_base.columns and isinstance(df_base["Mes"].dtype, pd.CategoricalDtype)
        else sorted(df_base.get("Mes", pd.Series(dtype=str)).dropna().unique(), key=lambda name: MONTH_NAME_TO_NUM.get(name, 13))
    )
    hours_all = sorted(df_base.get("Hora", pd.Series(dtype=str)).dropna().unique())
    cities_all = list(dict.fromkeys(df_base.get("Lugar", pd.Series(dtype=str)).dropna()))

    sidebar_threshold_defaults = build_thresholds(cities_all, stored_thresholds)

    st.sidebar.markdown("### Añadir estación Meteostat")
    station_query = st.sidebar.text_input("Nombre o ciudad", key="station_query")
    country_index = COUNTRY_OPTIONS.index("ES") if "ES" in COUNTRY_OPTIONS else 0
    station_country = st.sidebar.selectbox("País", options=COUNTRY_OPTIONS, index=country_index, key="country_select")
    date_range_download = st.sidebar.date_input(
        "Rango de fechas (descarga)",
        value=default_range_value,
        min_value=min_downloadable_date,
        max_value=today,
        format="YYYY/MM/DD",
        key="download_range",
    )

    def enqueue_station_download(station_id: Optional[str], station_name: Optional[str] = None) -> None:
        station_id_clean = canonical_station_id(station_id)
        if not station_id_clean:
            st.session_state["add_station_feedback"] = ("error", "Debes indicar un ID de estación")
            st.rerun()

        range_parsed = parse_date_range(date_range_download)
        if not range_parsed:
            st.session_state["add_station_feedback"] = ("error", "Selecciona un rango de fechas válido")
            st.rerun()
        start_date, end_date = range_parsed
        if start_date > end_date:
            st.session_state["add_station_feedback"] = ("error", "La fecha inicial no puede ser posterior a la final")
            st.rerun()

        start_dt = datetime.combine(start_date, datetime.min.time())
        end_dt = datetime.combine(end_date, datetime.min.time()) + timedelta(days=1)

        station_name_clean = (station_name or "").strip() or None
        if station_name_clean is None:
            meta = get_station_metadata(station_id_clean)
            if meta and "name" in meta and pd.notna(meta["name"]):
                station_name_clean = str(meta["name"])

        try:
            new_rows = append_station_to_master(station_id_clean, start_dt, end_dt, station_name_clean)
        except Exception as exc:  # noqa: BLE001
            st.session_state["add_station_feedback"] = ("error", f"Error al añadir estación: {exc}")
            st.rerun()

        if new_rows > 0:
            st.session_state["add_station_feedback"] = ("success", f"{new_rows} filas nuevas añadidas a data/meteostat_master.csv")
        else:
            st.session_state["add_station_feedback"] = ("info", "La estación ya estaba en el maestro (ninguna fila nueva)")

        st.session_state.pop("station_search_results", None)
        load_master_csv.clear()
        st.rerun()

    if st.sidebar.button("Buscar estaciones", key="search_button"):
        try:
            results = search_stations(station_query, station_country)
            st.session_state["station_search_results"] = results
            if results.empty:
                st.sidebar.info("Sin resultados para la búsqueda realizada.")
        except Exception as exc:  # noqa: BLE001
            st.session_state["station_search_results"] = pd.DataFrame()
            st.sidebar.error(f"Error al buscar estaciones: {exc}")

    results_df = st.session_state.get("station_search_results", pd.DataFrame())
    if not results_df.empty:
        st.sidebar.dataframe(results_df, use_container_width=True)
        station_labels = [station_select_label(row) for _, row in results_df.iterrows()]
        selection_idx = st.sidebar.selectbox(
            "Estación encontrada",
            options=range(len(results_df)),
            format_func=lambda idx: station_labels[idx],
            key="station_search_select",
        )
        if st.sidebar.button("Añadir estación al maestro", key="add_searched_station"):
            station_id = str(results_df.iloc[selection_idx]["id"]) if "id" in results_df.columns else None
            station_name = results_df.iloc[selection_idx].get("name") if "name" in results_df.columns else None
            if not station_id:
                st.session_state["add_station_feedback"] = ("error", "No se pudo identificar la estación seleccionada")
                st.rerun()
            enqueue_station_download(station_id, station_name)

    st.sidebar.markdown("#### Añadir estación por ID")
    station_id_manual = st.sidebar.text_input("ID Meteostat (manual)", key="manual_station_id")
    station_name_manual = st.sidebar.text_input("Nombre manual (opcional)", key="manual_station_name")
    if st.sidebar.button("Añadir por ID", key="btn_add_manual_station"):
        enqueue_station_download(station_id_manual, station_name_manual or None)

    existing_options_map: Dict[str, tuple] = {}
    if "station" in master_df.columns and not master_df.empty:
        existing_df = master_df[["station", "Lugar"]].dropna(subset=["station"]).drop_duplicates()
        for _, row in existing_df.iterrows():
            code = canonical_station_id(row["station"])
            label_name = str(row.get("Lugar", "")).strip()
            label = f"{code} · {label_name}" if label_name else code
            existing_options_map[label] = (code, label_name or None)

    if existing_options_map:
        st.sidebar.markdown("#### Actualizar estación ya descargada")
        existing_label = st.sidebar.selectbox(
            "Estaciones en maestro",
            options=sorted(existing_options_map.keys()),
            key="existing_station_select",
        )
        if st.sidebar.button("Actualizar estación existente", key="btn_update_existing_station"):
            station_id_existing, station_name_existing = existing_options_map[existing_label]
            enqueue_station_download(station_id_existing, station_name_existing)

    st.sidebar.markdown("---")

    date_filter_value = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_date or range_start_default, max_date or today),
        min_value=min_downloadable_date,
        max_value=today,
        format="YYYY/MM/DD",
        key="date_range_filter",
    )

    if "months_filter" not in st.session_state:
        st.session_state["months_filter"] = months_all.copy()
    st.session_state["months_filter"] = [m for m in st.session_state["months_filter"] if m in months_all] or months_all.copy()

    summer_months = {"Junio", "Julio", "Agosto", "Septiembre"}
    winter_months = {"Diciembre", "Enero", "Febrero", "Marzo"}
    month_buttons = st.sidebar.columns(3)
    if month_buttons[0].button("Ver verano", use_container_width=True):
        st.session_state["months_filter"] = [m for m in months_all if m in summer_months]
        st.rerun()
    if month_buttons[1].button("Ver invierno", use_container_width=True):
        st.session_state["months_filter"] = [m for m in months_all if m in winter_months]
        st.rerun()
    if month_buttons[2].button("Todos los meses", use_container_width=True):
        st.session_state["months_filter"] = months_all.copy()
        st.rerun()

    months_selected = st.sidebar.multiselect(
        "Mes(es)",
        options=months_all,
        key="months_filter",
    )
    if not months_selected:
        months_selected = months_all.copy()

    if "hours_filter" not in st.session_state:
        st.session_state["hours_filter"] = hours_all.copy()
    st.session_state["hours_filter"] = [h for h in st.session_state["hours_filter"] if h in hours_all] or hours_all.copy()

    hour_buttons = st.sidebar.columns(3)
    if hour_buttons[0].button("Horas día (06-18)", use_container_width=True):
        st.session_state["hours_filter"] = [h for h in hours_all if 6 <= int(h.split(":")[0]) <= 18]
        st.rerun()
    if hour_buttons[1].button("Horas noche", use_container_width=True):
        st.session_state["hours_filter"] = [
            h for h in hours_all if int(h.split(":")[0]) <= 5 or int(h.split(":")[0]) >= 19
        ]
        st.rerun()
    if hour_buttons[2].button("Todas las horas", use_container_width=True):
        st.session_state["hours_filter"] = hours_all.copy()
        st.rerun()

    hours_selected = st.sidebar.multiselect(
        "Hora(s)",
        options=hours_all,
        key="hours_filter",
    )
    if not hours_selected:
        hours_selected = hours_all.copy()

    if "cities_filter" not in st.session_state:
        st.session_state["cities_filter"] = cities_all.copy()
    st.session_state["cities_filter"] = [c for c in st.session_state["cities_filter"] if c in cities_all] or cities_all.copy()
    cities_selected = st.sidebar.multiselect(
        "Ciudad(es)",
        options=cities_all,
        key="cities_filter",
    )
    if not cities_selected:
        cities_selected = cities_all.copy()

    if "scenario_filter" not in st.session_state:
        st.session_state["scenario_filter"] = ["Todos"]
    escenario_selected = st.sidebar.multiselect(
        "Escenario(s)",
        options=SCENARIO_OPTIONS,
        key="scenario_filter",
    )
    if not escenario_selected:
        escenario_selected = ["Todos"]

    agg_option = st.sidebar.selectbox(
        "Agregación",
        options=list(AGG_FUNCTIONS.keys()),
        index=list(AGG_FUNCTIONS.keys()).index("mediana"),
        format_func=lambda key: AGG_LABELS.get(key, key),
        key="agg_option",
    )

    show_distribution = st.sidebar.checkbox("Mostrar gráfico de distribución", value=False, key="show_distribution")

    st.sidebar.markdown("**Umbrales de viento (km/h)**")
    thresholds_map: Dict[str, Dict[str, float]] = {}
    for city in cities_all:
        base = sidebar_threshold_defaults.get(city) or {"calma": 10.0, "ventoso": 25.0}
        city_hash = hashlib.md5(city.encode("utf-8")).hexdigest()
        calma_value = st.sidebar.number_input(
            f"{city} · Calma",
            min_value=0.0,
            max_value=500.0,
            value=base["calma"],
            step=1.0,
            format="%.1f",
            key=f"calma_{city_hash}",
        )
        ventoso_value = st.sidebar.number_input(
            f"{city} · Ventoso",
            min_value=calma_value,
            max_value=500.0,
            value=max(base["ventoso"], calma_value),
            step=1.0,
            format="%.1f",
            key=f"ventoso_{city_hash}",
        )
        thresholds_map[city] = {"calma": calma_value, "ventoso": ventoso_value}

    updated_thresholds = stored_thresholds.copy()
    updated_thresholds.update(thresholds_map)
    save_wind_thresholds(updated_thresholds)
    df_classified = apply_wind_class(df_base, thresholds_map)

    date_range = parse_date_range(date_filter_value)
    filtered = df_classified.copy()
    if date_range:
        start_date, end_date = date_range
        filtered = filtered[(filtered["Fecha"] >= start_date) & (filtered["Fecha"] <= end_date)]

    if months_selected:
        filtered = filtered[filtered["Mes"].isin(months_selected)]
    else:
        filtered = filtered.iloc[0:0]

    if hours_selected:
        filtered = filtered[filtered["Hora"].isin(hours_selected)]
    else:
        filtered = filtered.iloc[0:0]

    if cities_selected:
        filtered = filtered[filtered["Lugar"].isin(cities_selected)]
    else:
        filtered = filtered.iloc[0:0]

    if "Todos" not in escenario_selected and escenario_selected:
        filtered = filtered[filtered["escenario"].isin(escenario_selected)]

    filtered = filtered.copy()
    filtered["Mes"] = pd.Categorical(filtered["Mes"], categories=months_selected, ordered=True)
    filtered["Lugar"] = pd.Categorical(filtered["Lugar"], categories=cities_selected, ordered=False)

    agg_label = AGG_LABELS.get(agg_option, agg_option)
    if filtered.empty:
        period_label = "sin datos con los filtros actuales"
    else:
        period_start = filtered["Fecha"].min()
        period_end = filtered["Fecha"].max()
        period_label = f"{period_start:%Y/%m/%d} – {period_end:%Y/%m/%d}"
    st.markdown(f"*Indicadores agregados por ciudad ({agg_label}) sobre el periodo {period_label}.*")

    monthly = aggregate_monthly(filtered, agg_option)
    kpi_data = kpis(filtered, agg_option)

    missing_logs: List[str] = []

    kpi_cols = st.columns(len(kpi_data)) if not kpi_data.empty else []
    for col, (_, row) in zip(kpi_cols, kpi_data.iterrows()):
        col.subheader(row["Lugar"])
        col.caption(f"{agg_label} con los filtros aplicados")
        col.metric("Temp (°C)", format_metric(row["Temp (°C)"]))
        col.metric("Se siente (°C)", format_metric(row["Se siente (°C)"]))
        col.metric("Humedad_%", format_metric(row["Humedad_%"]))
        col.metric("Viento (km/h)", format_metric(row["Viento (km/h)"]))

    if filtered.empty:
        st.info("No hay datos con los filtros actuales.")

    value_columns = ["Temp (°C)", "Se siente (°C)"]
    has_monthly_data = not monthly[value_columns].dropna(how="all").empty if not monthly.empty else False

    if has_monthly_data:
        missing_rows = monthly[monthly[value_columns].isna().all(axis=1)]
        if not missing_rows.empty:
            for _, missing in missing_rows.iterrows():
                missing_logs.append(f"Sin datos para {missing['Lugar']} en {missing['Mes']} con los filtros actuales")

        fig = go.Figure()
        for city in cities_selected:
            city_data = monthly[monthly["Lugar"] == city]
            if city_data.empty:
                continue
            x_values = city_data["Mes"].astype(str)
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=city_data["Temp (°C)"],
                    mode="lines+markers",
                    name=f"{city} · Temp",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=city_data["Se siente (°C)"],
                    mode="lines+markers",
                    line=dict(dash="dash"),
                    name=f"{city} · Feels like",
                )
            )

        fig.update_layout(
            title=f"Evolución mensual · {agg_label}",
            xaxis_title="Mes",
            yaxis_title="°C",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=60, b=110, l=60, r=20),
            height=450,
        )
        fig.update_xaxes(categoryorder="array", categoryarray=months_selected)
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        delta_df = monthly.copy()
        delta_df["Delta (°C)"] = delta_df["Se siente (°C)"] - delta_df["Temp (°C)"]
        delta_available = delta_df.dropna(subset=["Delta (°C)"])
        if not delta_available.empty:
            fig_delta = go.Figure()
            for city in cities_selected:
                city_data = delta_available[delta_available["Lugar"] == city]
                if city_data.empty:
                    continue
                fig_delta.add_trace(
                    go.Bar(
                        x=city_data["Mes"].astype(str),
                        y=city_data["Delta (°C)"],
                        name=city,
                    )
                )
            fig_delta.update_layout(
                barmode="group",
                title="Diferencia Feels like vs Temp",
                xaxis_title="Mes",
                yaxis_title="Δ °C",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(t=60, b=110, l=60, r=20),
                height=420,
            )
            fig_delta.update_xaxes(categoryorder="array", categoryarray=months_selected)
            st.plotly_chart(fig_delta, use_container_width=True, theme="streamlit")
    else:
        st.info("Sin datos agregados para mostrar en el gráfico mensual.")

    st.download_button(
        "Descargar agregado mensual (CSV)",
        monthly.to_csv(index=False).encode("utf-8"),
        file_name="agregado_mensual.csv",
        mime="text/csv",
    )

    filters_description = {
        "Meses": ", ".join(months_selected) if months_selected else "Todos",
        "Horas": ", ".join(hours_selected) if hours_selected else "Todas",
        "Ciudades": ", ".join(cities_selected) if cities_selected else "Todas",
        "Escenarios": "Todos" if "Todos" in escenario_selected else ", ".join(escenario_selected),
    }

    summary_md = build_summary_markdown(
        period_label,
        agg_label,
        filters_description,
        kpi_data,
        missing_logs,
        len(filtered),
    )
    st.download_button(
        "Descargar resumen (Markdown)",
        summary_md.encode("utf-8"),
        file_name="resumen_meteostat.md",
        mime="text/markdown",
    )

    if show_distribution:
        st.subheader("Distribución de temperatura y sensación térmica")
        if filtered.empty:
            st.info("No hay datos para el gráfico de distribución.")
        else:
            temp_traces = []
            feel_traces = []
            for city in cities_selected:
                temp_values = pd.to_numeric(filtered.loc[filtered["Lugar"] == city, "temp"], errors="coerce").dropna()
                feel_values = pd.to_numeric(filtered.loc[filtered["Lugar"] == city, "feels_like"], errors="coerce").dropna()
                if not temp_values.empty:
                    temp_traces.append((city, temp_values))
                if not feel_values.empty:
                    feel_traces.append((city, feel_values))

            if temp_traces or feel_traces:
                fig_dist = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=["Temp (°C)", "Feels like (°C)"])
                for city, values in temp_traces:
                    fig_dist.add_trace(go.Box(y=values, name=city, boxpoints="outliers"), row=1, col=1)
                for city, values in feel_traces:
                    fig_dist.add_trace(go.Box(y=values, name=city, boxpoints="outliers"), row=1, col=2)
                fig_dist.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    margin=dict(t=80, b=100, l=60, r=20),
                )
                st.plotly_chart(fig_dist, use_container_width=True, theme="streamlit")
            else:
                st.info("No hay suficientes datos para el gráfico de distribución.")

            distribution_export = filtered[["datetime", "Lugar", "temp", "feels_like", "escenario"]].copy()
            st.download_button(
                "Descargar datos de distribución (CSV)",
                distribution_export.to_csv(index=False).encode("utf-8"),
                file_name="distribucion.csv",
                mime="text/csv",
            )

    if missing_logs:
        with st.expander("Incidencias de datos", expanded=False):
            for message in sorted(set(missing_logs)):
                st.markdown(f"- {message}")

    table_columns = ["datetime", "Lugar", "temp", "humedad_%", "wspd_kmh", "feels_like", "escenario"]
    available_columns = [col for col in table_columns if col in filtered.columns]
    table_df = filtered[available_columns].sort_values("datetime")

    st.subheader("Datos crudos filtrados")
    st.dataframe(table_df, use_container_width=True)
    st.download_button(
        "Descargar tabla filtrada (CSV)",
        table_df.to_csv(index=False).encode("utf-8"),
        file_name="datos_filtrados.csv",
        mime="text/csv",
    )


if __name__ == "__main__":
    main()


