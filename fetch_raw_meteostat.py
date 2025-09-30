# fetch_raw_meteostat.py
from datetime import datetime
import pandas as pd
from meteostat import Hourly

STATIONS = {
    "torrejon_08227": "08227",      # Madrid / Torrejón
    "san_javier_08433": "08433"     # Murcia / San Javier
}

start = datetime(2024, 10, 1)
end   = datetime(2025, 4, 30, 23, 59)

def fetch_station_raw(station_id: str) -> pd.DataFrame:
    df = Hourly(station_id, start, end, timezone="Europe/Madrid", model=False).fetch()
    df.reset_index(inplace=True)                # index = datetime
    df.rename(columns={"time": "datetime"}, inplace=True)
    return df

dfs = []
for name, sid in STATIONS.items():
    d = fetch_station_raw(sid)
    d.insert(0, "station", name)
    dfs.append(d)

raw = pd.concat(dfs, ignore_index=True)
raw.to_csv("meteostat_raw_oct-abr_torrejon08227_sanjavier08433.csv", index=False)
print("OK -> meteostat_raw_oct-abr_torrejon08227_sanjavier08433.csv")
