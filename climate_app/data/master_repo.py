import pandas as pd
import streamlit as st

from climate_app.data.paths import MASTER_FILE, ensure_data_dir
from climate_app.data.transformations import normalize_meteostat_df


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
    temp_path = MASTER_FILE.with_name(f"{MASTER_FILE.name}.tmp")
    df_to_save.to_csv(temp_path, index=False, encoding="utf-8")
    temp_path.replace(MASTER_FILE)
    load_master_csv.clear()
