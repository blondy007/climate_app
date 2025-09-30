# app.py  —  versión sin 1er gráfico y con umbrales de viento por ciudad
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FILENAME = "meteostat_raw_oct-abr_torrejon08227_sanjavier08433.csv"

STATION_NAME = {
    "torrejon_08227": "Torrejón de Ardoz",
    "san_javier_08433": "Los Alcázares (San Javier)"
}

# Umbrales por ciudad (km/h)
WIND_THRESHOLDS = {
    "Torrejón de Ardoz": {"calma": 10, "ventoso": 25},          # >25 => Viento fuerte
    "Los Alcázares (San Javier)": {"calma": 10, "ventoso": 35}, # >35 => Viento fuerte
}

MONTH_ORDER = [10, 11, 12, 1, 2, 3, 4]
MONTH_LABEL = {10:"Octubre", 11:"Noviembre", 12:"Diciembre", 1:"Enero", 2:"Febrero", 3:"Marzo", 4:"Abril"}
HOURS_WANTED = ["14:00", "00:00", "07:00"]

@st.cache_data
def load_raw(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # localizar columna temporal
    time_col = None
    for cand in ("datetime", "time", "date_time", "fecha", "fecha_hora"):
        if cand in df.columns:
            time_col = cand
            break
    if time_col is None:
        raise KeyError("No hay columna de fecha/hora en el CSV")

    df.rename(columns={time_col: "datetime"}, inplace=True)
    # normaliza a UTC y convierte a Europe/Madrid (evita .dt sobre strings / TZ mixtas)
    df["datetime"] = pd.to_datetime(df["datetime"].astype(str).str.strip(), errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).copy()
    df["datetime"] = df["datetime"].dt.tz_convert("Europe/Madrid")

    df["month"] = df["datetime"].dt.month
    df = df[df["month"].isin(MONTH_ORDER)].copy()
    df["Hora"] = df["datetime"].dt.strftime("%H:%M")
    df["Mes"] = pd.Categorical(df["month"].map(MONTH_LABEL), [MONTH_LABEL[m] for m in MONTH_ORDER], ordered=True)

    if "station" not in df.columns:
        raise KeyError("Falta columna 'station' en el CSV (Meteostat)")
    df["Lugar"] = df["station"].map(STATION_NAME).fillna(df["station"])

    # Viento en km/h
    df["wspd_kmh"] = (df["wspd"] * 3.6) if "wspd" in df.columns else np.nan

    if "temp" not in df.columns:
        raise KeyError("Falta columna 'temp' en el CSV")
    hum_col = "rhum" if "rhum" in df.columns else ("rh" if "rh" in df.columns else None)
    if hum_col is None:
        df["rh"] = np.nan
        hum_col = "rh"
    df.rename(columns={hum_col: "Humedad_%"}, inplace=True)

    # Wind-chill canadiense
    def wind_chill(temp_c, wind_kmh):
        if pd.notna(temp_c) and pd.notna(wind_kmh) and (temp_c <= 10) and (wind_kmh > 4.8):
            return 13.12 + 0.6215*temp_c - 11.37*(wind_kmh**0.16) + 0.3965*temp_c*(wind_kmh**0.16)
        return temp_c

    df["Se_siente"] = np.vectorize(wind_chill)(df["temp"], df["wspd_kmh"])
    return df

st.title("Comparativa real (Meteostat) · Torrejón vs Los Alcázares · Oct–Abr")

df_raw = load_raw(FILENAME)

# Clasificación por ciudad con umbrales por defecto (puedes permitir override si quieres)
def clasificar_viento_row(row):
    v = row["wspd_kmh"]
    lugar = row["Lugar"]
    if pd.isna(v) or lugar not in WIND_THRESHOLDS:
        return "Sin dato"
    thr = WIND_THRESHOLDS[lugar]
    if v <= thr["calma"]:
        return "Calma"
    elif v <= thr["ventoso"]:
        return "Ventoso"
    else:
        return "Viento fuerte"

df_raw["Escenario"] = df_raw.apply(clasificar_viento_row, axis=1)

# Filtros
col1, col2 = st.columns(2)
hora_e = col1.selectbox("Hora (evolución)", HOURS_WANTED, index=0, key="hora_ev")
esc_e  = col2.selectbox("Escenario (evolución)", ["Calma","Ventoso","Viento fuerte","Sin dato"], index=0, key="esc_ev")

# Vista cruda opcional
with st.expander("Filas crudas (sin tocar valores)"):
    st.dataframe(
        df_raw[(df_raw["Hora"]==hora_e)&(df_raw["Escenario"]==esc_e)]
        [["datetime","Lugar","temp","Humedad_%","wspd_kmh","Se_siente","Escenario"]]
        .sort_values("datetime")
    )

# --- SEGUNDO GRÁFICO (único) ---
st.subheader("Evolución mensual (oct–abr) · mediana por mes/lugar")

df_ev = df_raw[(df_raw["Hora"]==hora_e) & (df_raw["Escenario"]==esc_e)].copy()
df_ev["Mes"] = df_ev["Mes"].astype(str)
ORDER = ["Octubre","Noviembre","Diciembre","Enero","Febrero","Marzo","Abril"]
df_ev["Mes"] = pd.Categorical(df_ev["Mes"], ORDER, ordered=True)

evol = (
    df_ev.groupby(["Mes","Lugar"], observed=True, as_index=False)
         .agg({"temp":"median","Se_siente":"median"})
         .rename(columns={"temp":"Temp (°C)","Se_siente":"Se siente (°C)"})
         .sort_values(["Mes","Lugar"])
)

fig2, ax2 = plt.subplots()
if not evol.empty:
    for lugar in evol["Lugar"].unique():
        d = evol[evol["Lugar"]==lugar].sort_values("Mes")
        ax2.plot(d["Mes"].astype(str), d["Temp (°C)"], marker="o", label=f"{lugar} · Temp")
        ax2.plot(d["Mes"].astype(str), d["Se siente (°C)"], marker="x", linestyle="--", label=f"{lugar} · Se siente")
    ax2.set_ylabel("°C")
    ax2.set_title(f"{hora_e} · {esc_e}")
    ax2.legend()
    plt.xticks(rotation=45)
else:
    st.info("No hay datos para la combinación seleccionada.")
st.pyplot(fig2)

# Descarga del agregado
st.download_button(
    "Descargar agregado (CSV)",
    evol.to_csv(index=False).encode("utf-8"),
    file_name=f"evolucion_{hora_e}_{esc_e}.csv",
    mime="text/csv"
)
