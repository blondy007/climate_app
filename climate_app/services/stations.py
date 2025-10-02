from datetime import datetime
from typing import Optional

import pandas as pd

from climate_app.data.meteostat_client import fetch_station_hourly
from climate_app.data.master_repo import load_master_csv, save_master_csv
from climate_app.data.transformations import normalize_meteostat_df


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
