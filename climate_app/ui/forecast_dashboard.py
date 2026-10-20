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
from climate_app.ui.design import section, hero_html, weather_details, hourly_strip, daily_cards, empty_state, text
from climate_app.ui.appearance import PALETTES, appearance_controls, palette

METRICS = {"temperature": ("Temperatura", "°C"), "humidity": ("Humedad", "%"),
           "precipitation": ("Precipitación", "mm"), "wind_speed": ("Viento", "km/h"),
           "wind_direction": ("Dirección del viento", "°")}
COLORS = {"climateapp": "#197d8b", "ecmwf": "#647ba8", "icon": "#c99a48",
          "gfs": "#bc7d8d", "metno": "#5c9e8a", "weathernext3": "#8776b1", "climateapp_private": "#245c65"}
CHART_LABELS = {"climateapp": "Consenso", "climateapp_private": "Consenso + WN3",
                "ecmwf": "ECMWF IFS", "icon": "ICON", "gfs": "GFS",
                "metno": "MET Norway", "weathernext3": "WeatherNext 3"}
CHART_CONFIG = {"displayModeBar": False, "responsive": True}


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
    colors = palette()
    fig.update_layout(template="plotly_dark" if colors == PALETTES["dark"] else "plotly_white",
                      title=dict(text=title, font=dict(size=15)), height=height,
                      margin=dict(t=55, b=85, l=45, r=20),
                      paper_bgcolor=colors["surface"], plot_bgcolor=colors["surface"],
                      font=dict(family="Inter, Segoe UI, sans-serif", color=colors["ink"], size=11),
                      hovermode="x unified", legend=dict(orientation="h", y=-0.25,
                                                        yanchor="top", x=0, font=dict(size=11)),
                      yaxis_title=unit, xaxis_title=None)
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=colors["grid"], zeroline=False)
    return fig


def location_controls(store):
    notice = st.session_state.pop("place_notice", None)
    if notice:
        st.success(notice)
    locations = store.locations()
    st.caption(f"{len(locations)} de 10 lugares activos. Al quitar un lugar se detiene su recopilación; su historial se conserva y puedes recuperarlo abajo.")
    if not locations:
        st.info("Todavía no tienes lugares activos. Añade uno o recupera un lugar quitado.")
    for location in locations:
        with st.container(border=True):
            info, action = st.columns([3, 1])
            info.markdown(f'<div class="place-label"><strong>{text(location.name)}</strong><span>{location.latitude:.4f}, {location.longitude:.4f} · {text(location.timezone)}</span></div>', unsafe_allow_html=True)
            if action.button("Quitar", key=f"remove_place_{location.id}", use_container_width=True,
                             help=f"Quitar {location.name} de tus lugares sin borrar su historial"):
                store.deactivate(location.id)
                if st.session_state.get("selected_forecast_location") == location.id:
                    st.session_state.pop("selected_forecast_location", None)
                    st.session_state.pop("forecast_location", None)
                st.session_state["place_notice"] = f"{location.name} se ha quitado. Puedes recuperarlo en «Lugares quitados»."
                st.rerun()
    with st.expander("Añadir lugar"):
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
                    st.session_state["selected_forecast_location"] = location_id
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
                    st.session_state["selected_forecast_location"] = location_id
                    st.rerun()
                except (ValueError, KeyError):
                    st.warning("Revisa el nombre, la zona horaria y el límite de lugares activos.")
    active_ids = {loc.id for loc in locations}
    inactive = [loc for loc in store.locations(include_inactive=True) if loc.id not in active_ids]
    if inactive:
        with st.expander(f"Lugares quitados ({len(inactive)})"):
            st.caption("Recuperar un lugar reactiva su recopilación y permite consultar el historial que ya tenía.")
            for location in inactive:
                with st.container(border=True):
                    info, action = st.columns([3, 1])
                    info.markdown(f"**{location.name}**")
                    if action.button("Recuperar", key=f"restore_place_{location.id}", use_container_width=True,
                                     disabled=len(locations) >= 10, help=f"Volver a añadir {location.name}"):
                        try:
                            store.add_location(location.name, location.latitude, location.longitude, location.timezone)
                            st.session_state["place_notice"] = f"{location.name} vuelve a estar en tus lugares. Su historial sigue disponible."
                            st.rerun()
                        except ValueError as exc:
                            st.warning(str(exc))
            if len(locations) >= 10:
                st.caption("Quita un lugar activo antes de recuperar otro: el máximo es 10.")


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
    hero = st.empty()
    hours = st.radio("Horizonte", [48, 120, 240, 360], horizontal=True,
                     format_func=lambda h: {48: "48 horas", 120: "5 días", 240: "10 días", 360: "15 días"}[h])
    providers = [p for p in LABELS if p in frame.provider.unique()]
    with st.expander("Elegir modelo de previsión"):
        provider = st.selectbox("Previsión principal", providers, format_func=LABELS.get)
        st.caption("El consenso combina los modelos disponibles. También puedes consultar cada fuente por separado.")
    now = pd.Timestamp(utcnow())
    frame = frame[(frame.valid_time > now) & (frame.valid_time <= now + pd.Timedelta(hours=hours))].copy()
    main = frame[frame.provider == provider].sort_values("valid_time")
    if main.empty:
        st.info("Esta fuente no cubre el horizonte elegido. Prueba otro modelo o un plazo más corto.")
        return
    first = main.iloc[0]
    updated = main.fetched_at.max().tz_convert(location.timezone)
    stale = now - main.fetched_at.max() > pd.Timedelta(hours=9)
    hero.markdown(hero_html(location, first, main, CHART_LABELS[provider], updated, stale), unsafe_allow_html=True)
    if stale:
        st.warning("Esta previsión tiene más de 9 horas. Consulta el estado de las fuentes en Ajustes.")
    hourly_strip(main, location.timezone)
    weather_details(first, direction(first.wind_direction))
    daily = daily_summary(main, location.timezone)
    daily_cards(daily, main, location.timezone)

    section("Mira un poco más allá", "Explora la evolución, contrasta modelos y pon el tiempo en contexto.", "chart")
    evolution, comparison, climate = st.tabs(["Evolución", "Comparar modelos", "Climatología"])
    with evolution:
        st.plotly_chart(temperature_figure(main, main, provider, location), use_container_width=True, config=CHART_CONFIG, theme=None)
        left, right = st.columns(2)
        x = local_times(main.valid_time, location.timezone)
        rain = make_subplots(specs=[[{"secondary_y": True}]])
        rain.add_trace(go.Bar(x=x, y=main.precipitation, name="Lluvia · mm", marker_color="#509bb3", marker_line_width=0))
        if main.precip_probability.notna().any():
            rain.add_trace(go.Scatter(x=x, y=main.precip_probability, name="Probabilidad · %", line=dict(color="#c99a48", width=2)), secondary_y=True)
            rain.update_yaxes(range=[0, 100], title="%", secondary_y=True)
        left.plotly_chart(configure_figure(rain, "Lluvia por hora", "mm"), use_container_width=True, config=CHART_CONFIG, theme=None)
        wind = go.Figure()
        wind.add_trace(go.Scatter(x=x, y=main.wind_speed, name="Viento", line=dict(color="#328783", width=2.5),
                                 fill="tozeroy", fillcolor="rgba(50,135,131,.07)",
                                 customdata=main.wind_direction.map(direction), hovertemplate="%{y:.1f} km/h · %{customdata}<extra></extra>"))
        if main.wind_gust.notna().any():
            wind.add_trace(go.Scatter(x=x, y=main.wind_gust, name="Rachas", line=dict(dash="dot", color="#c99a48", width=2)))
        right.plotly_chart(configure_figure(wind, "Viento y rachas", "km/h"), use_container_width=True, config=CHART_CONFIG, theme=None)
        st.caption("Lluvia acumulada durante cada hora. Las probabilidades e intervalos solo aparecen si la fuente los proporciona.")
    with comparison:
        compare = st.checkbox("Comparar modelos", value=True)
        st.plotly_chart(temperature_figure(main, frame if compare else main, provider, location), use_container_width=True, config=CHART_CONFIG, theme=None)
        if provider in ("climateapp", "climateapp_private"):
            metadata = json.loads(first.metadata)
            st.markdown('<div class="insight-card"><h3>' + ("Un consenso que aprende de sus aciertos" if metadata.get("trained") else "El consenso está empezando a aprender") + '</h3><p>' + ("La combinación se pondera según los errores observados anteriormente en este lugar y plazo." if metadata.get("trained") else "Por ahora, los modelos tienen el mismo peso. Los pesos se ajustan cuando cada fuente acumula al menos 30 observaciones en tres días por variable y plazo.") + '</p></div>', unsafe_allow_html=True)
        st.caption("Las diferencias entre líneas muestran desacuerdo entre modelos. La banda P10–P90 es un intervalo nativo de la fuente, cuando existe; el desacuerdo entre modelos no es una probabilidad.")
    with climate:
        section("¿Es normal para esta época?", "Compara cada hora prevista con los mismos momentos de años anteriores.", "layers")
        observations = store.observations(location.id, since=utcnow() - timedelta(days=6 * 366))
        target_index = st.select_slider("Hora que quieres comparar", options=list(range(len(main))), value=min(23, len(main) - 1),
                                        format_func=lambda i: main.iloc[i].valid_time.tz_convert(location.timezone).strftime("%d/%m %H:%M"))
        target = main.iloc[target_index]
        baseline = climatology(observations, target.valid_time, target.temperature, location.timezone)
        if baseline:
            columns = st.columns(3)
            columns[0].metric("Previsión", number(target.temperature, "°C"))
            columns[1].metric("Referencia histórica", number(baseline["mean"], "°C"), delta=f"{baseline['delta']:+.1f} °C previstos")
            columns[2].metric("Percentil histórico", f"{baseline['percentile']:.0f}")
            delta = baseline["delta"]
            label = "Más cálido de lo habitual" if delta > 1 else "Más fresco de lo habitual" if delta < -1 else "En línea con años anteriores"
            st.markdown(f'<div class="insight-card"><h3>{label}</h3><p>La previsión está {abs(delta):.1f} °C {"por encima" if delta >= 0 else "por debajo"} de la media de la muestra comparable.</p></div>', unsafe_allow_html=True)
            st.caption(f"{baseline['n']} observaciones · Años {', '.join(map(str, baseline['years']))} · Misma hora local y fechas ±15 días. Es una referencia de la muestra disponible, no una normal climatológica de 30 años.")
        else:
            st.info("La referencia histórica se está preparando. Aparecerá cuando haya suficientes observaciones de años anteriores para esa fecha y hora.")
        station_caption(location)
    with st.expander("Tablas completas y descargas"):
        st.caption("La lluvia diaria suma las horas disponibles; la probabilidad diaria mostrada es la máxima horaria, no la probabilidad de lluvia durante todo el día.")
        daily_table = daily.rename(columns={"date": "Día", "minimum": "Mínima °C", "maximum": "Máxima °C",
            "precipitation": "Lluvia mm", "wind": "Viento máximo km/h", "gust": "Racha máxima km/h",
            "probability": "Prob. horaria máx. %", "rain_hours": "Horas con dato de lluvia", "temperature_hours": "Horas previstas"}).round(2)
        st.dataframe(daily_table, hide_index=True, use_container_width=True)
        export = main[["valid_time", "temperature", "humidity", "precipitation", "wind_speed", "wind_direction", "wind_gust", "precip_probability"]].copy()
        export["valid_time"] = export.valid_time.dt.tz_convert(location.timezone)
        for source, label in (("source_init_time", "Pasada superficie (UTC)"), ("temperature_init_time", "Pasada temperatura (UTC)")):
            if source in main and main[source].notna().any():
                export[label] = main[source]
        st.dataframe(export.rename(columns={"valid_time": "Hora local", **{key: val[0] for key, val in METRICS.items()}}), use_container_width=True, hide_index=True)
        if provider != "weathernext3":
            st.download_button("Descargar CSV", export.to_csv(index=False).encode("utf-8-sig"), "climateapp-prevision.csv", "text/csv")


def temperature_figure(main, comparison, provider, location):
    fig = go.Figure()
    if main.temperature_p10.notna().any() and main.temperature_p90.notna().any():
        fig.add_trace(go.Scatter(x=local_times(main.valid_time, location.timezone), y=main.temperature_p90,
                                mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
        fig.add_trace(go.Scatter(x=local_times(main.valid_time, location.timezone), y=main.temperature_p10,
                                mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(80,155,179,.15)", name="P10–P90 nativo"))
    for source in comparison.provider.unique():
        data = comparison[comparison.provider == source]
        fig.add_trace(go.Scatter(x=local_times(data.valid_time, location.timezone), y=data.temperature,
                                name=CHART_LABELS.get(source, source), line=dict(color=COLORS[source], width=3 if source == provider else 1.6), connectgaps=False))
    return configure_figure(fig, "Temperatura · hora local", "°C", 370)


def station_caption(location):
    if location.station_id:
        distance = f" · a {location.station_distance_km:.1f} km" if location.station_distance_km is not None else ""
        st.caption(f"Referencia observada: {location.station_name or location.station_id} ({location.station_id}){distance}. La estación y el punto previsto pueden tener condiciones locales distintas.")
    else:
        st.caption("Buscando una estación de referencia a menos de 50 km. Si no existe, no se mostrará una evaluación ficticia.")


def evaluation_view(store, location):
    st.markdown('<div class="page-intro"><h1>La previsión, a prueba.</h1><p>Descubre qué modelos aciertan más en tu lugar, comparando lo que anunciaron con lo que midió la estación.</p></div>', unsafe_allow_html=True)
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
        empty_state("Los aciertos se ganan con el tiempo", "Ya estamos guardando las previsiones. Cuando se cumplan y lleguen las observaciones de la estación, aquí verás qué modelos aciertan más. Sin resultados inventados ni previsiones reconstruidas después.")
        return
    selected = result[result.variable == variable].copy()
    if selected.empty:
        st.info("La estación todavía no aporta observaciones suficientes de esta variable.")
        return
    unit = METRICS[variable][1]
    cards = st.columns(3)
    cards[0].metric("Modelos evaluados", selected.provider.nunique())
    cards[1].metric("Comparaciones", f"{selected.n.sum():,}".replace(",", "."))
    cards[2].metric("Plazos de antelación", selected.horizon.nunique())
    selected["model"] = selected.provider.map(LABELS)
    fig = go.Figure()
    for provider, group in selected.groupby("provider"):
        fig.add_trace(go.Bar(x=group.horizon, y=group.mae, name=CHART_LABELS[provider], marker_color=COLORS[provider],
                             customdata=group.n, hovertemplate="MAE %{y:.2f}<br>%{customdata} muestras<extra>%{fullData.name}</extra>"))
    fig.update_xaxes(categoryorder="array", categoryarray=[label for _, label in HORIZONS])
    st.plotly_chart(configure_figure(fig, "Error absoluto medio", unit, 400), use_container_width=True, config=CHART_CONFIG, theme=None)
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
    locations = store.locations()
    if not locations:
        empty_state("Tu próximo lugar favorito", "Ve a Ajustes y añade un lugar para empezar a explorar su tiempo.")
        return
    lookup = {loc.id: loc for loc in locations}
    if st.session_state.get("forecast_location") not in lookup:
        preferred = st.session_state.get("selected_forecast_location")
        st.session_state["forecast_location"] = preferred if preferred in lookup else locations[0].id
    picker, manage = st.columns([4, 1])
    location_id = picker.selectbox("Lugar", list(lookup), format_func=lambda key: lookup[key].name, key="forecast_location", label_visibility="collapsed")
    def open_settings():
        st.session_state["main_view"] = "⚙️ Ajustes"
    manage.button("Mis lugares", on_click=open_settings, use_container_width=True, help="Añadir, quitar o recuperar lugares en Ajustes")
    location = lookup[location_id]
    st.session_state["selected_forecast_location"] = location_id
    if evaluation:
        evaluation_view(store, location)
    else:
        frame = visible_forecasts(store.forecasts(location.id, since=utcnow()))
        if frame.empty:
            empty_state("Estamos mirando al cielo", "La primera previsión se está preparando. Las fuentes se actualizan automáticamente, también cuando no tienes la app abierta.")
        else:
            forecast_view(store, location, frame)
    if st.button("Actualizar vista", help="Lee las descargas del recolector sin aumentar las consultas a los proveedores"):
        st.rerun()


def render_settings():
    store = Store()
    store.seed()
    st.markdown('<div class="page-intro"><h1>A tu manera.</h1><p>Tus lugares, tu acceso personal y las fuentes que hacen posible tu previsión.</p></div>', unsafe_allow_html=True)
    section("Apariencia", "Un aspecto cómodo a cualquier hora.", "sun")
    appearance_controls()
    section("Tus lugares", "Añade, quita o recupera tus sitios favoritos.", "pin")
    location_controls(store)
    section("Tu cuenta", "Acceso personal a WeatherNext y al consenso completo.", "layers")
    owner_login(store)
    locations = store.locations()
    if locations:
        lookup = {loc.id: loc for loc in locations}
        location_id = st.selectbox("Estado de las fuentes para", list(lookup), format_func=lambda key: lookup[key].name)
        section("Detrás de la previsión", "Fuentes, estaciones y actualización automática.", "clock")
        provider_status(store, lookup[location_id])
    with st.expander("Fuentes, licencias y datos"):
        st.markdown("[Open-Meteo](https://open-meteo.com/) · ECMWF · DWD · NOAA · [MET Norway](https://www.met.no/en) · [Meteostat](https://meteostat.net/). [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). ClimateApp realiza conversiones de unidades y agregaciones.")
        st.caption("Los iconos resumen la nubosidad y precipitación disponibles; no sustituyen avisos oficiales. La temperatura destacada es una previsión de la próxima hora disponible, no una medición actual. Los datos ausentes se muestran con una raya.")
        st.caption("Aplicación de uso personal y coste adicional 0 €. Las previsiones y observaciones se conservan automáticamente sin necesidad de visitar la web.")
    if private_view():
        st.caption("Google BigQuery · © 2024-6 Google LLC, whose machine learning models were used to create the experimental data made available under the following licence terms https://storage.googleapis.com/weathernext-public/terms-of-use.pdf. This data is intended for experimental modelling only and is not intended, validated, or approved for real world use.")
