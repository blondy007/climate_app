import hashlib
from datetime import date, datetime, timedelta

import pandas as pd
import streamlit as st

from climate_app.data.loaders import load_csv
from climate_app.data.master_repo import load_master_csv
from climate_app.data.paths import DEFAULT_FILE, MASTER_FILE, ensure_data_dir
from climate_app.data.thresholds_repo import load_wind_thresholds
from climate_app.data.transformations import normalize_meteostat_df
from climate_app.services.dataset import apply_filters as apply_filters_service
from climate_app.services.filters import resolve_active_date_range
from climate_app.services.wind import apply_wind_class
from climate_app.shared.state import UPLOAD_CACHE
from climate_app.ui.dashboard import render_dashboard
from climate_app.ui.notifications import handle_station_feedback
from climate_app.ui.sidebar import render_sidebar

st.set_page_config(page_title="Clima Meteostat · Comparativa", layout="wide")


def main() -> None:
    ensure_data_dir()

    if "station_search_results" not in st.session_state:
        st.session_state["station_search_results"] = pd.DataFrame()

    stored_thresholds = load_wind_thresholds()
    uploaded_file = st.session_state.get("upload_csv")

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

    data_key = str(DEFAULT_FILE)
    source_label = MASTER_FILE.name if uploaded_file is None else uploaded_file.name

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
    initial_range = (min_date or range_start_default, max_date or today)

    filters, thresholds_map = render_sidebar(
        df_base,
        master_df,
        stored_thresholds,
        default_range_value,
        min_downloadable_date,
        today,
        initial_range,
    )

    df_classified = apply_wind_class(df_base, thresholds_map)

    date_range = resolve_active_date_range(filters.raw_date_value, st.session_state)
    filtered = apply_filters_service(df_classified, filters, date_range)

    render_dashboard(filtered, filters, date_range, filters.agg_option, source_label)

    st.sidebar.header("Carga de datos")
    st.sidebar.file_uploader("Archivo CSV (Meteostat Hourly)", type=["csv"], key="upload_csv")


if __name__ == "__main__":
    main()
