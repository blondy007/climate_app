import hashlib
import io
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_categorical_dtype

st.set_page_config(page_title="Clima Meteostat · Comparativa", layout="wide")

DEFAULT_FILE = Path(__file__).resolve().parent / "meteostat_raw_oct-abr_torrejon08227_sanjavier08433.csv"

STATION_NAME_MAP = {
    "torrejon_08227": "Torrejón de Ardoz",
    "san_javier_08433": "Los Alcázares (San Javier)",
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
    "min": "Minimo",
    "max": "Maximo",
}

AGG_KEYS = list(AGG_FUNCTIONS.keys())
UPLOAD_CACHE: Dict[str, bytes] = {}


def format_metric(value: float) -> str:
    if pd.isna(value):
        return "-"
    return f"{value:.1f}"


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

    df.columns = [str(col).strip().lower().replace(" ", "_") for col in df.columns]

    time_col = None
    for candidate in ("datetime", "time", "date_time", "fecha", "fecha_hora"):
        if candidate in df.columns:
            time_col = candidate
            break
    if time_col is None:
        raise KeyError("No se encontró columna temporal compatible")

    df = df.rename(columns={time_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"].astype(str).str.strip(), errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).copy()
    df["datetime"] = df["datetime"].dt.tz_convert("Europe/Madrid")

    if "station" not in df.columns:
        raise KeyError("Falta columna station en el CSV")
    if "temp" not in df.columns:
        raise KeyError("Falta columna temp en el CSV")
    if "wspd" not in df.columns:
        df["wspd"] = np.nan

    hum_col = None
    for candidate in ("rhum", "rh"):
        if candidate in df.columns:
            hum_col = candidate
            break
    if hum_col is None:
        df["rh"] = np.nan
        hum_col = "rh"

    df = df.copy()
    df.rename(columns={hum_col: "humedad_%"}, inplace=True)

    station_lower = df["station"].astype(str).str.strip().str.lower()
    df["Lugar"] = station_lower.map(STATION_NAME_MAP).fillna(df["station"])

    df["MesNum"] = df["datetime"].dt.month
    month_values = sorted(df["MesNum"].dropna().unique().tolist())
    month_labels = [MONTH_NAMES.get(month, str(month)) for month in month_values]
    df["Mes"] = df["MesNum"].map(lambda month: MONTH_NAMES.get(month, str(month)))
    df["Mes"] = pd.Categorical(df["Mes"], categories=month_labels, ordered=True)
    df["Hora"] = df["datetime"].dt.strftime("%H:%M")
    df["Fecha"] = df["datetime"].dt.date

    temp_numeric = pd.to_numeric(df["temp"], errors="coerce")
    wspd_numeric = pd.to_numeric(df["wspd"], errors="coerce")
    df["wspd_kmh"] = wspd_numeric * 3.6
    df["feels_like"] = [
        compute_wind_chill(temp_value, wind_value)
        for temp_value, wind_value in zip(temp_numeric, df["wspd_kmh"])
    ]

    return df


def compute_wind_chill(temp_c: float, wind_kmh: float) -> float:
    if pd.isna(temp_c):
        return np.nan
    if pd.isna(wind_kmh):
        return temp_c
    if temp_c <= 10 and wind_kmh > 4.8:
        wind_factor = wind_kmh ** 0.16
        return 13.12 + 0.6215 * temp_c - 11.37 * wind_factor + 0.3965 * temp_c * wind_factor
    return temp_c


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
    classified["escenario"] = classified.apply(classify_row_wind, axis=1, args=(thresholds_map,))
    classified["escenario"] = pd.Categorical(classified["escenario"], categories=SCENARIO_CATEGORIES, ordered=False)
    return classified


@st.cache_data(show_spinner=False)
def aggregate_monthly(df: pd.DataFrame, agg: str) -> pd.DataFrame:
    agg_func = AGG_FUNCTIONS.get(agg, agg_median)

    months_categories: List[str] = []
    if "Mes" in df.columns:
        if is_categorical_dtype(df["Mes"]):
            months_categories = list(df["Mes"].cat.categories)
        else:
            raw_months = df["Mes"].dropna().unique().tolist()
            months_categories = sorted(raw_months, key=lambda name: MONTH_NAME_TO_NUM.get(name, 13))

    cities_categories: List[str] = []
    if "Lugar" in df.columns:
        if is_categorical_dtype(df["Lugar"]):
            cities_categories = list(df["Lugar"].cat.categories)
        else:
            cities_categories = list(dict.fromkeys(df["Lugar"].dropna()))

    if df.empty:
        if months_categories and cities_categories:
            idx = pd.MultiIndex.from_product([months_categories, cities_categories], names=["Mes", "Lugar"])
            empty = pd.DataFrame(index=idx, columns=["Temp (°C)", "Se siente (°C)"], dtype=float).reset_index()
        else:
            empty = pd.DataFrame(columns=["Mes", "Lugar", "Temp (°C)", "Se siente (°C)"])
        if months_categories:
            empty["Mes"] = pd.Categorical(empty.get("Mes", []), categories=months_categories, ordered=True)
        if cities_categories:
            empty["Lugar"] = pd.Categorical(empty.get("Lugar", []), categories=cities_categories, ordered=False)
        return empty

    grouped = (
        df.groupby(["Mes", "Lugar"], observed=True)[["temp", "feels_like"]]
        .agg(agg_func)
        .rename(columns={"temp": "Temp (°C)", "feels_like": "Se siente (°C)"})
    )

    if months_categories and cities_categories:
        idx = pd.MultiIndex.from_product([months_categories, cities_categories], names=["Mes", "Lugar"])
        grouped = grouped.reindex(idx)

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
        if is_categorical_dtype(df["Lugar"]):
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


def build_thresholds(cities: List[str]) -> Dict[str, Dict[str, float]]:
    thresholds: Dict[str, Dict[str, float]] = {}
    for city in cities:
        base = DEFAULT_THRESHOLDS.get(city, {"calma": 10.0, "ventoso": 25.0})
        thresholds[city] = {"calma": float(base["calma"]), "ventoso": float(base["ventoso"]) }
    return thresholds


def main() -> None:
    uploaded_file = st.sidebar.file_uploader("Archivo CSV (Meteostat Hourly)", type=["csv"])

    data_key = str(DEFAULT_FILE)
    source_label = DEFAULT_FILE.name

    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_bytes).hexdigest()
        data_key = f"_uploaded::{uploaded_file.name}::{file_hash}"
        UPLOAD_CACHE[data_key] = file_bytes
        source_label = uploaded_file.name

    try:
        df_raw = load_csv(data_key)
    except Exception as exc:  # noqa: BLE001
        st.error(f"No se pudo cargar el archivo seleccionado: {exc}")
        return

    st.title("Clima Meteostat · Comparativa")
    st.caption(f"Archivo activo: {source_label}")

    min_date = df_raw["Fecha"].min()
    max_date = df_raw["Fecha"].max()

    months_all = list(df_raw["Mes"].cat.categories) if is_categorical_dtype(df_raw["Mes"]) else sorted(df_raw["Mes"].dropna().unique(), key=lambda name: MONTH_NAME_TO_NUM.get(name, 13))
    hours_all = sorted(df_raw["Hora"].dropna().unique())
    cities_all = list(dict.fromkeys(df_raw["Lugar"].dropna()))

    sidebar_thresholds = build_thresholds(cities_all)

    date_value = st.sidebar.date_input(
        "Rango de fechas",
        value=(min_date, max_date) if min_date and max_date else (),
        min_value=min_date,
        max_value=max_date,
    )

    months_selected = st.sidebar.multiselect("Mes(es)", options=months_all, default=months_all)
    hours_selected = st.sidebar.multiselect("Hora(s)", options=hours_all, default=hours_all)
    cities_selected = st.sidebar.multiselect("Ciudad(es)", options=cities_all, default=cities_all)
    escenario_selected = st.sidebar.multiselect("Escenario(s)", options=SCENARIO_OPTIONS, default=["Todos"])

    agg_index = AGG_KEYS.index("mediana") if "mediana" in AGG_KEYS else 0
    agg_option = st.sidebar.selectbox(
        "Agregación",
        options=AGG_KEYS,
        index=agg_index,
        format_func=lambda key: AGG_LABELS.get(key, key),
    )

    show_distribution = st.sidebar.checkbox("Mostrar gráfico de distribución", value=False)

    st.sidebar.markdown("**Umbrales de viento (km/h)**")
    thresholds_map: Dict[str, Dict[str, float]] = {}
    for city in cities_all:
        base = sidebar_thresholds[city]
        city_hash = hashlib.md5(city.encode("utf-8")).hexdigest()
        calma_key = f"calma_{city_hash}"
        ventoso_key = f"ventoso_{city_hash}"
        calma_value = st.sidebar.number_input(
            f"{city} · Calma",
            min_value=0.0,
            max_value=500.0,
            value=base["calma"],
            step=1.0,
            format="%.1f",
            key=calma_key,
        )
        ventoso_value = st.sidebar.number_input(
            f"{city} · Ventoso",
            min_value=calma_value,
            max_value=500.0,
            value=max(base["ventoso"], calma_value),
            step=1.0,
            format="%.1f",
            key=ventoso_key,
        )
        thresholds_map[city] = {"calma": calma_value, "ventoso": ventoso_value}

    df_classified = apply_wind_class(df_raw, thresholds_map)

    if isinstance(date_value, tuple) and len(date_value) == 2:
        start_date, end_date = date_value
    elif isinstance(date_value, list) and len(date_value) == 2:
        start_date, end_date = date_value
    elif date_value:
        start_date = end_date = date_value
    else:
        start_date = min_date
        end_date = max_date

    filtered = df_classified.copy()
    if start_date and end_date:
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

    monthly = aggregate_monthly(filtered, agg_option)
    kpi_data = kpis(filtered, agg_option)

    kpi_cols = st.columns(len(kpi_data)) if not kpi_data.empty else []
    for col, (_, row) in zip(kpi_cols, kpi_data.iterrows()):
        col.subheader(row["Lugar"])
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
                st.warning(f"Sin datos para {missing['Lugar']} en {missing['Mes']} con los filtros actuales")

        fig, ax = plt.subplots()
        for city in cities_selected:
            city_data = monthly[monthly["Lugar"] == city]
            if city_data.empty:
                continue
            x_values = city_data["Mes"].astype(str)
            ax.plot(x_values, city_data["Temp (°C)"], marker="o", label=f"{city} · Temp")
            ax.plot(x_values, city_data["Se siente (°C)"], marker="o", linestyle="--", label=f"{city} · Feels like")
        ax.set_xlabel("Mes")
        ax.set_ylabel("°C")
        ax.set_title(f"Evolución mensual · {AGG_LABELS.get(agg_option, agg_option)}")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        st.pyplot(fig)
    else:
        st.info("Sin datos agregados para mostrar en el gráfico mensual.")

    st.download_button(
        "Descargar agregado mensual (CSV)",
        monthly.to_csv(index=False).encode("utf-8"),
        file_name="agregado_mensual.csv",
        mime="text/csv",
    )

    if show_distribution:
        st.subheader("Distribución de temperatura y sensación térmica")
        if filtered.empty:
            st.info("No hay datos para el gráfico de distribución.")
        else:
            temp_data = []
            temp_labels = []
            for city in cities_selected:
                values = filtered.loc[filtered["Lugar"] == city, "temp"].dropna()
                if values.empty:
                    continue
                temp_data.append(values)
                temp_labels.append(city)

            feels_data = []
            feels_labels = []
            for city in cities_selected:
                values = filtered.loc[filtered["Lugar"] == city, "feels_like"].dropna()
                if values.empty:
                    continue
                feels_data.append(values)
                feels_labels.append(city)

            if temp_data or feels_data:
                fig_dist, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
                if temp_data:
                    axes[0].boxplot(temp_data, labels=temp_labels)
                    axes[0].set_title("Temp (°C)")
                    axes[0].set_xlabel("Ciudad")
                    axes[0].set_ylabel("°C")
                else:
                    axes[0].set_visible(False)
                if feels_data:
                    axes[1].boxplot(feels_data, labels=feels_labels)
                    axes[1].set_title("Feels like (°C)")
                    axes[1].set_xlabel("Ciudad")
                else:
                    axes[1].set_visible(False)
                plt.tight_layout()
                st.pyplot(fig_dist)
            else:
                st.info("No hay suficientes datos para el gráfico de distribución.")

            distribution_export = filtered[["datetime", "Lugar", "temp", "feels_like", "escenario"]].copy()
            st.download_button(
                "Descargar datos de distribución (CSV)",
                distribution_export.to_csv(index=False).encode("utf-8"),
                file_name="distribucion.csv",
                mime="text/csv",
            )

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



