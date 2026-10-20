from __future__ import annotations

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from climate_app.forecast.analysis import climatology, daily_summary, scores
from climate_app.forecast.access import private_view, visible_forecasts
from climate_app.forecast.domain import HORIZONS, LABELS, instant, utcnow
from climate_app.forecast.providers import geocode
from climate_app.forecast.store import Store
from climate_app.auth import owner_login

METRICS = {"temperature": ("Temperatura", "°C"), "humidity": ("Humedad", "%"),
           "precipitation": ("Precipitación", "mm"), "wind_speed": ("Viento", "km/h"),
           "wind_direction": ("Dirección del viento", "°")}
COLORS = {"climateapp": "#0d9488", "ecmwf": "#6366f1", "icon": "#f59e0b",
          "gfs": "#ec4899", "metno": "#3b82f6", "weathernext3": "#8b5cf6", "climateapp_private": "#0f766e"}


def number(value, unit="", digits=1):
    return "Sin dato" if pd.isna(value) else f"{value:.{digits}f} {unit}".strip()


def direction(value):
    if pd.isna(value):
        return "Sin dirección"
    return f"{['N','NE','E','SE','S','SO','O','NO'][int((value + 22.5) // 45) % 8]} · {value:.0f}°"


def local_times(series, timezone):
    # Naive local datetimes display as the chosen city's clock in every browser.
    return series.dt.tz_convert(timezone).dt.tz_localize(None)


def configure_figure(fig, title, unit, height=330):
    fig.update_layout(title=title, height=height, margin=dict(t=65, b=30, l=15, r=15),
                      hovermode="x unified", legend=dict(orientation="h", y=1.03),
                      yaxis_title=unit, xaxis_title=None)
    return fig


def location_controls(store):
    with st.expander("Añadir o gestionar lugares"):
        st.caption("Hasta 10 lugares activos. Las ubicaciones y sus previsiones se guardan entre visitas.")
        with st.form("search_forecast_place"):
            query = st.text_input("Buscar ciudad o localidad", max_chars=120)
            submitted = st.form_submit_button("Buscar")
        if submitted:
            try:
                st.session_state["forecast_place_results"] = geocode(query, store)
            except Exception as exc:
                st.warning(str(exc))
        results = st.session_state.get("forecast_place_results", [])
        if results:
            selected = st.selectbox("Resultados", range(len(results)),
                format_func=lambda i: " · ".join(str(results[i].get(k, "")) for k in ("name", "admin1", "country")))
            if st.button("Guardar lugar", key="save_geocoded_place"):
                item = results[selected]
                try:
                    location_id = store.add_location(item["name"], item["latitude"], item["longitude"],
                                                     item.get("timezone", "UTC"))
                    st.session_state["forecast_location"] = location_id
                    st.session_state.pop("forecast_place_results", None)
                    st.rerun()
                except ValueError as exc:
                    st.warning(str(exc))
        elif submitted:
            st.info("No se han encontrado lugares. Puedes introducir coordenadas abajo.")
        with st.form("manual_forecast_place"):
            st.markdown("**Añadir por coordenadas**")
            name = st.text_input("Nombre", max_chars=120)
            left, right = st.columns(2)
            lat = left.number_input("Latitud", min_value=-90.0, max_value=90.0, value=40.4594, format="%.4f")
            lon = right.number_input("Longitud", min_value=-180.0, max_value=180.0, value=-3.4697, format="%.4f")
            timezone = st.text_input("Zona horaria", value="Europe/Madrid", help="Por ejemplo: Europe/Madrid, Atlantic/Canary o UTC")
            if st.form_submit_button("Guardar coordenadas"):
                try:
                    location_id = store.add_location(name, lat, lon, timezone)
                    st.session_state["forecast_location"] = location_id
                    st.rerun()
                except (ValueError, KeyError):
                    st.warning("Revisa el nombre, la zona horaria y el límite de lugares activos.")
        locations = store.locations()
        if locations:
            inactive = st.selectbox("Dejar de recopilar un lugar", [loc.id for loc in locations],
                                    format_func=lambda value: next(loc.name for loc in locations if loc.id == value))
            if st.button("Desactivar lugar", help="Conserva todas sus previsiones y observaciones archivadas"):
                store.deactivate(inactive)
                st.session_state.pop("forecast_location", None)
                st.rerun()


def provider_status(store, location):
    with st.expander("Fuentes y actualización automática"):
        rows = []
        for provider, label in LABELS.items():
            if provider in ("climateapp", "climateapp_private"):
                continue
            state = store.state(f"task:{location.id}:{provider}", {})
            updated = state.get("updated_at")
            rows.append({"Fuente": label, "Estado": {"ok": "Disponible", "pending": "Pendiente",
                        "error": "Reintentando"}.get(state.get("status"), "Esperando primera descarga"),
                         "Última consulta": pd.Timestamp(updated).tz_convert(location.timezone).strftime("%d/%m %H:%M") if updated else "—",
                         "Detalle": state.get("message", "")})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        st.caption("La recopilación funciona aunque no abras la web. Los modelos se consultan cada 3 horas; WeatherNext, cada hora cuando esté habilitado. Un fallo conserva los datos anteriores.")
        st.markdown("WeatherNext 3 necesita [acceso aprobado por Google](https://developers.google.com/weathernext/guides/access-forecast). La versión completa permanece pendiente hasta verificar esa conexión.")
        collector = store.state("collector", {})
        if collector.get("heartbeat"):
            age = (utcnow() - instant(collector["heartbeat"])).total_seconds() / 60
            if age > 15:
                st.warning(f"El recolector no comunica actividad desde hace {age:.0f} minutos.")
        else:
            st.info("El recolector todavía no ha iniciado su primera descarga.")


def forecast_view(store, location, frame):
    controls = st.columns([2, 2, 1])
    hours = controls[0].radio("Horizonte", [48, 120, 240, 360], horizontal=True,
                             format_func=lambda h: {48: "48 h", 120: "5 días", 240: "10 días", 360: "15 días"}[h])
    providers = [p for p in LABELS if p in frame.provider.unique()]
    provider = controls[1].selectbox("Previsión principal", providers, format_func=LABELS.get)
    compare = controls[2].checkbox("Comparar modelos", value=True)
    now = pd.Timestamp(utcnow())
    frame = frame[(frame.valid_time > now) & (frame.valid_time <= now + pd.Timedelta(hours=hours))].copy()
    main = frame[frame.provider == provider].sort_values("valid_time")
    if main.empty:
        st.info("Esta fuente no cubre el horizonte elegido.")
        return
    updated = main.fetched_at.max().tz_convert(location.timezone)
    age = now - main.fetched_at.max()
    end = main.valid_time.max().tz_convert(location.timezone)
    st.caption(f"Descargado el {updated:%d/%m/%Y a las %H:%M} · Hora local: {location.timezone} · Cobertura hasta {end:%d/%m %H:%M}")
    if age > pd.Timedelta(hours=9):
        st.warning("Previsión antigua: revisa el estado de las fuentes. La fecha de descarga se conserva.")
    first = main.iloc[0]
    st.markdown(f"### Próxima hora disponible · {first.valid_time.tz_convert(location.timezone):%d/%m %H:%M}")
    cards = st.columns(4)
    cards[0].metric("Temperatura prevista", number(first.temperature, "°C"))
    if pd.notna(first.temperature_p10) and pd.notna(first.temperature_p90):
        cards[0].caption(f"P10–P90: {first.temperature_p10:.1f}–{first.temperature_p90:.1f} °C")
    cards[1].metric("Humedad", number(first.humidity, "%", 0))
    cards[2].metric("Viento", number(first.wind_speed, "km/h"))
    cards[2].caption(direction(first.wind_direction))
    cards[3].metric("Precipitación", number(first.precipitation, "mm"))
    cards[3].caption("Probabilidad: " + number(first.precip_probability, "%", 0))
    if provider in ("climateapp", "climateapp_private"):
        metadata = json.loads(first.metadata)
        st.info("Consenso ponderado por aciertos anteriores." if metadata.get("trained") else
                "Consenso inicial con pesos iguales. Se necesitan al menos 30 observaciones en 3 días por modelo, variable y plazo para aprender los pesos.")
    fig = go.Figure()
    if main.temperature_p10.notna().any() and main.temperature_p90.notna().any():
        fig.add_trace(go.Scatter(x=local_times(main.valid_time, location.timezone), y=main.temperature_p90,
                                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=local_times(main.valid_time, location.timezone), y=main.temperature_p10,
                                mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(139,92,246,.18)",
                                name="P10–P90 nativo"))
    for source in (frame if compare else main).provider.unique():
        data = frame[frame.provider == source]
        fig.add_trace(go.Scatter(x=local_times(data.valid_time, location.timezone), y=data.temperature,
                                name=LABELS.get(source, source), line=dict(color=COLORS[source], width=3 if source == provider else 1),
                                connectgaps=False))
    st.plotly_chart(configure_figure(fig, "Temperatura y comparación de modelos", "°C", 430), use_container_width=True)
    st.caption("La banda P10–P90 solo aparece cuando la fuente la proporciona. La separación entre líneas muestra desacuerdo entre modelos y no es un intervalo probabilístico.")
    left, right = st.columns(2)
    rain = make_subplots(specs=[[{"secondary_y": True}]])
    x = local_times(main.valid_time, location.timezone)
    rain.add_trace(go.Bar(x=x, y=main.precipitation, name="Precipitación (mm)", marker_color="#3b82f6"))
    if main.precip_probability.notna().any():
        rain.add_trace(go.Scatter(x=x, y=main.precip_probability, name="Probabilidad (%)", line=dict(color="#8b5cf6")), secondary_y=True)
        rain.update_yaxes(range=[0, 100], title="%", secondary_y=True)
    left.plotly_chart(configure_figure(rain, "Precipitación por hora", "mm"), use_container_width=True)
    wind = go.Figure()
    wind.add_trace(go.Scatter(x=x, y=main.wind_speed, name="Sostenido", line=dict(color="#0d9488"),
                             customdata=main.wind_direction.map(direction), hovertemplate="%{y:.1f} km/h · %{customdata}"))
    if main.wind_gust.notna().any():
        wind.add_trace(go.Scatter(x=x, y=main.wind_gust, name="Rachas", line=dict(dash="dot", color="#f59e0b")))
    right.plotly_chart(configure_figure(wind, "Viento y rachas", "km/h"), use_container_width=True)
    st.markdown("### Previsión diaria")
    daily = daily_summary(main, location.timezone)
    daily = daily.rename(columns={"date": "Día", "minimum": "Mínima °C", "maximum": "Máxima °C",
        "precipitation": "Lluvia mm", "wind": "Viento máximo km/h", "gust": "Racha máxima km/h",
        "probability": "Prob. horaria máx. %", "rain_hours": "Horas con dato de lluvia", "temperature_hours": "Horas previstas"})
    st.dataframe(daily, hide_index=True, use_container_width=True)
    st.caption("Hoy y el último día pueden estar incompletos. La lluvia suma solo horas disponibles; la probabilidad mostrada es la máxima horaria, no la probabilidad diaria.")
    st.markdown("### Comparación con la climatología local")
    observations = store.observations(location.id, since=utcnow() - timedelta(days=6 * 366))
    target_index = st.select_slider("Hora que quieres comparar", options=list(range(len(main))), value=min(23, len(main) - 1),
                                    format_func=lambda i: main.iloc[i].valid_time.tz_convert(location.timezone).strftime("%d/%m %H:%M"))
    target = main.iloc[target_index]
    baseline = climatology(observations, target.valid_time, target.temperature, location.timezone)
    if baseline:
        columns = st.columns(3)
        columns[0].metric("Previsión", number(target.temperature, "°C"))
        columns[1].metric("Media de años anteriores", number(baseline["mean"], "°C"), delta=f"{baseline['delta']:+.1f} °C previstos")
        columns[2].metric("Percentil histórico", f"{baseline['percentile']:.0f}")
        st.caption(f"{baseline['n']} observaciones · Años {', '.join(map(str, baseline['years']))} · Misma hora local y fechas ±15 días. Es una referencia de la muestra disponible, no una normal climatológica de 30 años.")
    else:
        st.info("Todavía no hay suficientes observaciones de años anteriores para esa fecha y hora. El recolector descarga la referencia histórica automáticamente cuando la estación dispone de ella.")
    station_caption(location)
    with st.expander("Tabla horaria y descarga"):
        export = main[["valid_time", "temperature", "humidity", "precipitation", "wind_speed", "wind_direction", "wind_gust", "precip_probability"]].copy()
        export["valid_time"] = export.valid_time.dt.tz_convert(location.timezone)
        st.dataframe(export.rename(columns={"valid_time": "Hora local", **{key: value[0] for key, value in METRICS.items()}}), use_container_width=True, hide_index=True)
        if provider != "weathernext3":
            st.download_button("Descargar CSV", export.to_csv(index=False).encode("utf-8-sig"), "climateapp-prevision.csv", "text/csv")


def station_caption(location):
    if location.station_id:
        distance = f" · a {location.station_distance_km:.1f} km" if location.station_distance_km is not None else ""
        st.caption(f"Referencia observada: {location.station_name or location.station_id} ({location.station_id}){distance}. La estación y el punto previsto pueden tener condiciones locales distintas.")
    else:
        st.caption("Buscando una estación de referencia a menos de 50 km. Si no existe, no se mostrará una evaluación ficticia.")


def evaluation_view(store, location):
    st.markdown("### Qué modelos aciertan en este lugar")
    options = st.columns(3)
    days = options[0].selectbox("Ventana de evaluación", [30, 60, 90], index=2, format_func=lambda n: f"Últimos {n} días")
    variable = options[1].selectbox("Variable", list(METRICS), format_func=lambda key: METRICS[key][0])
    common = options[2].checkbox("Solo horas comunes", value=True, help="Compara cada plazo usando los mismos instantes observados para los modelos que tienen datos")
    since = utcnow() - timedelta(days=days)
    archive = visible_forecasts(store.forecasts(location.id, since=since, latest=False))
    observations = store.observations(location.id, since=since)
    result = scores(archive, observations, common_only=common)
    station_caption(location)
    if result.empty:
        st.info("La evaluación empieza cuando las previsiones archivadas se cumplen y la estación publica observaciones. No se usan previsiones reconstruidas después del resultado.")
        return
    selected = result[result.variable == variable].copy()
    if selected.empty:
        st.info("La estación todavía no aporta observaciones suficientes de esta variable.")
        return
    unit = METRICS[variable][1]
    selected["model"] = selected.provider.map(LABELS)
    fig = go.Figure()
    for provider, group in selected.groupby("provider"):
        fig.add_trace(go.Bar(x=group.horizon, y=group.mae, name=LABELS[provider], marker_color=COLORS[provider],
                             customdata=group.n, hovertemplate="MAE %{y:.2f}<br>%{customdata} muestras<extra>%{fullData.name}</extra>"))
    fig.update_xaxes(categoryorder="array", categoryarray=[label for _, label in HORIZONS])
    st.plotly_chart(configure_figure(fig, "Error absoluto medio · menor es mejor", unit, 400), use_container_width=True)
    display = selected[["model", "horizon", "mae", "rmse", "bias", "n", "days", "brier", "interval_coverage"]].rename(columns={
        "model": "Modelo", "horizon": "Antelación", "mae": f"MAE {unit}", "rmse": f"RMSE {unit}",
        "bias": f"Sesgo {unit}", "n": "Muestras", "days": "Días", "brier": "Brier lluvia ≥0,1 mm",
        "interval_coverage": "Cobertura P10–P90"})
    st.dataframe(display, use_container_width=True, hide_index=True)
    st.caption("Antelación desde la descarga, porque no todas las APIs publican la hora de ejecución del modelo. Se conserva una muestra por modelo, hora objetivo y plazo. Los pesos se calculan con datos disponibles antes de crear cada consenso; este se archiva para evaluar también sus propios resultados.")
    if not common:
        st.caption("Los tamaños de muestra pueden diferir entre modelos; compara también su cobertura.")
    st.download_button("Descargar evaluación", display.to_csv(index=False).encode("utf-8-sig"), "climateapp-aciertos.csv", "text/csv")


def render_forecast_dashboard(evaluation=False):
    store = Store()
    store.seed()
    st.title("ClimateApp")
    owner_login(store)
    if private_view():
        st.caption("Sesión personal · WeatherNext y consenso completo para uso personal")
    st.caption("Previsión, modelos y clima local · Uso personal · Coste adicional 0 €")
    location_controls(store)
    locations = store.locations()
    if not locations:
        st.info("Añade un lugar para empezar.")
        return
    lookup = {loc.id: loc for loc in locations}
    if st.session_state.get("forecast_location") not in lookup:
        st.session_state["forecast_location"] = locations[0].id
    location_id = st.selectbox("Lugar", list(lookup), format_func=lambda key: lookup[key].name, key="forecast_location")
    location = lookup[location_id]
    if st.button("Actualizar vista", help="Lee las descargas del recolector sin aumentar las consultas a los proveedores"):
        st.rerun()
    provider_status(store, location)
    if evaluation:
        evaluation_view(store, location)
    else:
        frame = visible_forecasts(store.forecasts(location.id, since=utcnow()))
        if frame.empty:
            st.info("Preparando la primera previsión. El recolector consultará automáticamente las fuentes gratuitas.")
        else:
            forecast_view(store, location, frame)
    st.divider()
    st.markdown("Datos: [Open-Meteo](https://open-meteo.com/) · ECMWF · DWD · NOAA · [MET Norway](https://www.met.no/en) · [Meteostat](https://meteostat.net/) · Google DeepMind WeatherNext cuando esté habilitado. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/); conversiones de unidades y agregaciones realizadas por ClimateApp.")
    if private_view():
        st.caption("Google BigQuery · © 2024-6 Google LLC, whose machine learning models were used to create the experimental data made available under the following licence terms https://storage.googleapis.com/weathernext-public/terms-of-use.pdf. This data is intended for experimental modelling only and is not intended, validated, or approved for real world use.")
