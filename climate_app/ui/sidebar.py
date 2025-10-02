import hashlib
from datetime import date, datetime, timedelta
from typing import Dict, Tuple

import pandas as pd
import streamlit as st

from climate_app.data.master_repo import load_master_csv
from climate_app.data.meteostat_client import get_station_metadata, search_stations
from climate_app.data.thresholds_repo import save_wind_thresholds
from climate_app.services.analytics import AGG_FUNCTIONS, AGG_LABELS
from climate_app.services.filters import parse_date_range
from climate_app.services.stations import append_station_to_master
from climate_app.services.wind import build_thresholds
from climate_app.shared.constants import COUNTRY_OPTIONS, MONTH_NAME_TO_NUM, SCENARIO_OPTIONS
from climate_app.shared.models import FilterSelections
from climate_app.shared.utils import canonical_station_id


def _sync_multiselect(
    *,
    key_state: str,
    key_widget: str,
    options,
    default_options,
    label: str,
) -> list:
    or_default = default_options if default_options else []
    st.session_state.setdefault(key_state, or_default.copy())
    st.session_state.setdefault(key_widget, st.session_state[key_state].copy())

    if not st.session_state[key_state] and default_options:
        st.session_state[key_state] = default_options.copy()
        st.session_state[key_widget] = default_options.copy()

    selected = st.sidebar.multiselect(
        label,
        options=options,
        default=st.session_state.get(key_state, default_options),
        key=key_widget,
    )
    if selected:
        st.session_state[key_state] = selected
    else:
        st.session_state[key_state] = default_options.copy()
        st.session_state[key_widget] = st.session_state[key_state].copy()
    return st.session_state[key_state]


def render_sidebar(
    df_base: pd.DataFrame,
    master_df: pd.DataFrame,
    stored_thresholds: Dict[str, Dict[str, float]],
    default_range_value: Tuple[date, date],
    min_downloadable_date: date,
    today: date,
    initial_range: Tuple[date, date],
) -> Tuple[FilterSelections, Dict[str, Dict[str, float]]]:
    months_all = (
        list(df_base["Mes"].cat.categories)
        if "Mes" in df_base.columns and isinstance(df_base["Mes"].dtype, pd.CategoricalDtype)
        else sorted(
            df_base.get("Mes", pd.Series(dtype=str)).dropna().unique(),
            key=lambda name: MONTH_NAME_TO_NUM.get(name, 13),
        )
    )
    hours_all = sorted(df_base.get("Hora", pd.Series(dtype=str)).dropna().unique())
    cities_all = list(dict.fromkeys(df_base.get("Lugar", pd.Series(dtype=str)).dropna()))

    sidebar_threshold_defaults = build_thresholds(cities_all, stored_thresholds)

    st.sidebar.header("Filtros")
    st.sidebar.subheader("Ciudades y agregación")
    st.session_state.setdefault("cities_filter", cities_all.copy())
    st.session_state.setdefault("cities_filter_widget", st.session_state["cities_filter"].copy())

    if cities_all:
        quick_container = st.sidebar.container()
        quick_container.markdown("_Accesos rápidos_")
        cols = quick_container.columns(min(3, max(1, len(cities_all))))
        for idx, city in enumerate(cities_all):
            col = cols[idx % len(cols)]
            if col.button(city, key=f"quick_city_{idx}"):
                st.session_state["cities_filter"] = [city]
                st.session_state["cities_filter_widget"] = [city]
                st.experimental_rerun()
        if quick_container.button("Todas las ciudades", key="quick_city_all", use_container_width=True):
            st.session_state["cities_filter"] = cities_all.copy()
            st.session_state["cities_filter_widget"] = cities_all.copy()
            st.experimental_rerun()

    cities_selected = _sync_multiselect(
        key_state="cities_filter",
        key_widget="cities_filter_widget",
        options=cities_all,
        default_options=cities_all.copy(),
        label="Ciudad(es)",
    )

    agg_keys = list(AGG_FUNCTIONS.keys())
    default_agg_index = agg_keys.index("mediana") if "mediana" in agg_keys else 0
    agg_option = st.sidebar.selectbox(
        "Agregación",
        options=agg_keys,
        index=default_agg_index,
        format_func=lambda key: AGG_LABELS.get(key, key),
        key="agg_option",
    )

    st.sidebar.subheader("Fechas y meses")
    if "date_range_filter" not in st.session_state:
        st.session_state["date_range_filter"] = initial_range
    date_filter_value = st.sidebar.date_input(
        "Rango de fechas",
        value=st.session_state["date_range_filter"],
        min_value=min_downloadable_date,
        max_value=today,
        format="YYYY/MM/DD",
        key="date_range_filter",
    )

    months_selected = _sync_multiselect(
        key_state="months_filter",
        key_widget="months_filter_widget",
        options=months_all,
        default_options=months_all.copy(),
        label="Mes(es)",
    )

    st.sidebar.subheader("Horas")
    hours_selected = _sync_multiselect(
        key_state="hours_filter",
        key_widget="hours_filter_widget",
        options=hours_all,
        default_options=hours_all.copy(),
        label="Hora(s)",
    )

    st.sidebar.subheader("Escenarios")
    escenario_selected = _sync_multiselect(
        key_state="scenario_filter",
        key_widget="scenario_filter_widget",
        options=SCENARIO_OPTIONS,
        default_options=["Todos"],
        label="Escenario(s)",
    )

    show_distribution = st.sidebar.checkbox(
        "Mostrar gráfico de distribución", value=False, key="show_distribution_toggle"
    )

    st.sidebar.header("Gestión de estaciones")
    station_query = st.sidebar.text_input("Nombre o ciudad", key="station_query")
    country_index = COUNTRY_OPTIONS.index("ES") if "ES" in COUNTRY_OPTIONS else 0
    station_country = st.sidebar.selectbox(
        "País", options=COUNTRY_OPTIONS, index=country_index, key="country_select"
    )
    date_range_download = st.sidebar.date_input(
        "Rango de fechas (descarga)",
        value=default_range_value,
        min_value=min_downloadable_date,
        max_value=today,
        format="YYYY/MM/DD",
        key="download_range",
    )

    def enqueue_station_download(station_id: str, station_name: str | None = None) -> None:
        station_id_clean = canonical_station_id(station_id)
        if not station_id_clean:
            st.session_state["add_station_feedback"] = ("error", "Debes indicar un ID de estación")
            st.experimental_rerun()

        range_parsed = parse_date_range(date_range_download)
        if not range_parsed:
            st.session_state["add_station_feedback"] = ("error", "Selecciona un rango de fechas válido")
            st.experimental_rerun()
        start_date, end_date = range_parsed
        if start_date > end_date:
            st.session_state["add_station_feedback"] = ("error", "La fecha inicial no puede ser posterior a la final")
            st.experimental_rerun()

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
            st.experimental_rerun()

        if new_rows > 0:
            st.session_state["add_station_feedback"] = (
                "success",
                f"{new_rows} filas nuevas añadidas a data/meteostat_master.csv",
            )
        else:
            st.session_state["add_station_feedback"] = (
                "info",
                "La estación ya estaba en el maestro (ninguna fila nueva)",
            )

        st.session_state.pop("station_search_results", None)
        load_master_csv.clear()
        st.experimental_rerun()

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
                st.session_state["add_station_feedback"] = (
                    "error",
                    "No se pudo identificar la estación seleccionada",
                )
                st.experimental_rerun()
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
            label = f"{code} - {label_name}" if label_name else code
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

    st.sidebar.header("Configuración de umbrales")
    thresholds_map: Dict[str, Dict[str, float]] = {}
    for city in cities_all:
        base = sidebar_threshold_defaults.get(city) or {"calma": 10.0, "ventoso": 25.0}
        city_hash = hashlib.md5(city.encode("utf-8")).hexdigest()
        calma_value = st.sidebar.number_input(
            f"{city} - Calma",
            min_value=0.0,
            max_value=500.0,
            value=base["calma"],
            step=1.0,
            format="%.1f",
            key=f"calma_{city_hash}",
        )
        ventoso_value = st.sidebar.number_input(
            f"{city} - Ventoso",
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

    filters = FilterSelections(
        cities=cities_selected,
        agg_option=agg_option,
        raw_date_value=date_filter_value,
        months=months_selected,
        hours=hours_selected,
        scenarios=escenario_selected,
        show_distribution=show_distribution,
    )

    return filters, thresholds_map


def station_select_label(row: pd.Series) -> str:
    station_id = row.get("id")
    name = row.get("name")
    country = row.get("country")
    base = str(station_id)
    if pd.notna(name) and str(name).strip():
        base = f"{station_id} - {name}"
    if pd.notna(country) and str(country).strip():
        return f"{base} ({country})"
    return base
