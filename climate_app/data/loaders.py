import io

import pandas as pd
import streamlit as st

from climate_app.data.transformations import normalize_meteostat_df
from climate_app.shared.state import UPLOAD_CACHE


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
