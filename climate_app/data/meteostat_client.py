from datetime import datetime
from typing import Dict, Optional

import pandas as pd
import streamlit as st
from meteostat import Hourly, Stations

from climate_app.data.transformations import normalize_meteostat_df
from climate_app.shared.utils import canonical_station_id


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
    return df


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
