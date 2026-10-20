"""Shared visual language. Weather illustrations are original, local SVGs."""
from __future__ import annotations

from html import escape
from pathlib import Path
import base64

import pandas as pd
import streamlit as st

from climate_app.ui.appearance import PALETTES, palette, sync_appearance

WEEKDAYS = ("lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo")
MONTHS = ("enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre")


def text(value):
    return escape(str(value), quote=True)


def value(number, suffix="", digits=0):
    return "—" if pd.isna(number) else f"{number:.{digits}f}{suffix}"


def icon(name, size=22):
    paths = {
        "wind": '<path d="M3 8h12a3 3 0 1 0-3-3M3 12h16a3 3 0 1 1-3 3M3 16h5a3 3 0 1 1-3 3"/>',
        "drop": '<path d="M12 3S5 11 5 15a7 7 0 0 0 14 0c0-4-7-12-7-12Z"/><path d="M8 15a4 4 0 0 0 4 4"/>',
        "rain": '<path d="M6 14a4 4 0 0 1 0-8 6 6 0 0 1 11-1 4.5 4.5 0 0 1 1 9M8 17l-1 3m6-3-1 3m6-3-1 3"/>',
        "chart": '<path d="M4 4v16h16M8 15l4-5 4 2 5-7"/>',
        "clock": '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
        "pin": '<path d="M19 10c0 6-7 11-7 11S5 16 5 10a7 7 0 1 1 14 0Z"/><circle cx="12" cy="10" r="2.5"/>',
        "layers": '<path d="m12 3 10 5-10 5L2 8l10-5Zm-9 9 9 5 9-5M3 16l9 5 9-5"/>',
        "arrow": '<path d="M4 12h15m-6-6 6 6-6 6"/>',
        "sun": '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M2 12h2m16 0h2M5 5l1.5 1.5m11 11L19 19M5 19l1.5-1.5m11-11L19 5"/>',
        "settings": '<circle cx="12" cy="12" r="3"/><path d="m9 3-1 3-3 1-2 3 2 2-1 4 3 2 3-1 3 3 3-2v-3l4-2-1-4-3-1-1-4Z"/>',
    }
    return f'<svg xmlns="http://www.w3.org/2000/svg" class="line-icon" width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">{paths.get(name, paths["chart"])}</svg>'


def weather_kind(row, timezone="Europe/Madrid"):
    rain, cloud = row.get("precipitation"), row.get("cloud_cover")
    when = row.get("valid_time")
    hour = pd.Timestamp(when).tz_convert(timezone).hour if when is not None else 12
    night = hour < 7 or hour >= 20
    if pd.notna(rain) and rain >= 0.1:
        return "rain", "Precipitación prevista"
    if pd.notna(cloud) and cloud >= 70:
        return "cloud", "Cielo nuboso"
    if pd.notna(cloud) and cloud >= 25:
        return ("partly_night", "Nubes y claros") if night else ("partly", "Sol y nubes")
    if pd.isna(cloud):
        return "unknown", "Previsión local"
    if night:
        return "night", "Cielo despejado"
    return "sun", "Cielo despejado"


def weather_svg(kind="partly", size=80):
    # No borrowed assets, fonts, external icon requests or image subscriptions.
    sun = '<circle cx="49" cy="43" r="20" fill="#FFC95D"/><g stroke="#FFD57A" stroke-width="3" stroke-linecap="round"><path d="M49 13V7M49 79v-6M19 43h-6M85 43h-6M28 22l-4-4M74 68l-4-4M28 64l-4 4M74 18l-4 4"/></g>'
    cloud = '<path d="M26 76a18 18 0 0 1-1-36 27 27 0 0 1 51-2 19 19 0 0 1 0 38Z" fill="#F2F8FC"/><path d="M26 76a18 18 0 0 1-15-8c15 4 31 1 44-8 12-8 22-8 36-7a19 19 0 0 1-15 23Z" fill="#DCEAF2"/>'
    moon = '<path d="M77 73A34 34 0 0 1 44 16a29 29 0 0 0 33 57Z" fill="#E8F2FF"/><path d="m79 20 2 6 6 2-6 2-2 6-2-6-6-2 6-2Z" fill="#FFC95D"/>'
    if kind == "sun":
        art = '<g transform="translate(11 12)">' + sun + '</g>'
    elif kind == "night":
        art = moon
    elif kind == "partly_night":
        art = moon + cloud
    elif kind == "unknown":
        art = '<circle cx="55" cy="52" r="30" fill="#DCEAF2"/><path d="M37 60h36M37 48h25M37 36h13" stroke="#5B819A" stroke-width="4" stroke-linecap="round"/>'
    else:
        art = ('<g transform="translate(18 -8)">' + sun + '</g>' if kind == "partly" else "") + cloud
        if kind == "rain":
            art += '<path d="m34 85-4 10m27-10-4 10m27-10-4 10" stroke="#7BCCFF" stroke-width="5" stroke-linecap="round"/>'
        elif kind == "snow":
            art += '<g fill="#9BD9FF"><circle cx="31" cy="91" r="3"/><circle cx="54" cy="88" r="3"/><circle cx="76" cy="94" r="3"/></g>'
    return f'<svg class="weather-icon" width="{size}" height="{size}" viewBox="0 0 110 110" role="img" aria-label="{text({"sun":"Despejado","night":"Noche despejada","partly":"Sol y nubes","partly_night":"Noche con nubes y claros","cloud":"Nuboso","rain":"Precipitación","snow":"Nieve","unknown":"Sin dato de nubosidad"}.get(kind,"Tiempo"))}">{art}</svg>'


def apply_design():
    mode = sync_appearance()
    css = Path(__file__).with_name("weather.css").read_text(encoding="utf-8")
    variables = lambda colors: ";".join(f"--{name}:{color}" for name, color in colors.items())
    css += f':root {{{variables(PALETTES["light"])};color-scheme:light;}}'
    dark = f':root {{{variables(PALETTES["dark"])};--accent:#77d1dc;--soft:#243e4b;--hover:#244451;--nav:#192d38f2;--table-filter:invert(.9) hue-rotate(180deg);color-scheme:dark;}}'
    if mode == "dark":
        css += dark
    elif mode == "auto":
        css += f'@media(prefers-color-scheme:dark) {{{dark}}}'
    for index, name in enumerate(("sun", "chart", "clock", "settings"), start=1):
        encoded = base64.b64encode(icon(name).encode()).decode()
        css += f'.element-container:has(.nav-anchor)+.element-container [data-baseweb="radio"]:nth-child({index})::before {{ mask-image:url("data:image/svg+xml;base64,{encoded}"); }}'
    settings_icon = base64.b64encode(icon("settings").encode()).decode()
    css += f'[data-testid="stHorizontalBlock"]:has(.location-toolbar-anchor) button::before {{mask-image:url("data:image/svg+xml;base64,{settings_icon}");}}'
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def plot_chart(fig, **kwargs):
    """Give the historical charts the same typography and palette as forecasts."""
    colors = palette()
    fig.update_layout(template="plotly_dark" if colors == PALETTES["dark"] else "plotly_white",
                      paper_bgcolor=colors["surface"], plot_bgcolor=colors["surface"],
                      font=dict(family="Inter, Segoe UI, sans-serif", color=colors["ink"], size=11),
                      colorway=["#197d8b", "#c99a48", "#647ba8", "#bc7d8d", "#5c9e8a"],
                      legend=dict(orientation="h", y=-.3, yanchor="top", x=0, xanchor="left",
                                  entrywidth=210, font=dict(size=10)),
                      margin=dict(t=60, b=150, l=45, r=20))
    fig.update_xaxes(showgrid=False, zeroline=False)
    fig.update_yaxes(gridcolor=colors["grid"], zeroline=False)
    kwargs["config"] = {"displayModeBar": False, "responsive": True}
    kwargs["theme"] = None
    st.plotly_chart(fig, **kwargs)


def brand():
    st.markdown(f'<div class="app-brand"><div class="brand-name">{weather_svg("partly",40)}<span>climate<span class="brand-light">app</span><b class="brand-dot">.</b></span></div><span class="brand-tagline">Tu clima, con perspectiva.</span></div>', unsafe_allow_html=True)


def section(title, subtitle="", symbol="chart"):
    st.markdown(f'<div class="section-heading"><span class="section-icon">{icon(symbol)}</span><div><h2>{text(title)}</h2>{f"<p>{text(subtitle)}</p>" if subtitle else ""}</div></div>', unsafe_allow_html=True)


def hero_html(location, first, main, provider_label, updated, stale=False):
    kind, condition = weather_kind(first, location.timezone)
    local = first.valid_time.tz_convert(location.timezone)
    near = main[main.valid_time < first.valid_time + pd.Timedelta(hours=24)]
    minimum, maximum = near.temperature.min(), near.temperature.max()
    interval = ""
    if pd.notna(first.temperature_p10) and pd.notna(first.temperature_p90):
        interval = f'<span class="hero-range">P10–P90 {value(first.temperature_p10,"°",1)} – {value(first.temperature_p90,"°",1)}</span>'
    return f'''<section class="weather-hero sky-{kind}" aria-label="Previsión de {text(location.display_name)}">
<div class="hero-top"><span>{icon("pin",16)} {text(location.display_name)}</span><span class="source-chip">{text(provider_label)}</span></div>
<div class="hero-main"><div><p class="hero-eyebrow">PRÓXIMA HORA · {text(WEEKDAYS[local.weekday()])} {local:%H:%M}</p><div class="hero-value">{value(first.temperature,"°")}<small>C</small></div><p class="hero-feels">Sensación <strong>{value(first.get("feels_like"),"°")}</strong><small>Estimada a la sombra</small></p><h1>{text(condition)}</h1>{interval}<p class="hero-extremes">Próximas 24 h <strong>↑ {value(maximum,"°")} &nbsp; ↓ {value(minimum,"°")}</strong><small>Sensación ↑ {value(near.get("feels_like", pd.Series(dtype=float)).max(),"°")} · ↓ {value(near.get("feels_like", pd.Series(dtype=float)).min(),"°")}</small></p></div><div class="hero-art">{weather_svg(kind,220)}</div></div>
<div class="hero-bottom"><span class="freshness {"stale" if stale else ""}"><i></i>{"Datos antiguos" if stale else "Actualizado"} {updated:%H:%M} · {updated:%d/%m}</span><span>Hora local · {text(location.timezone)}</span></div></section>'''


def weather_details(first, wind_direction):
    items = [("wind", "Viento", value(first.wind_speed,"",1), "km/h", wind_direction),
             ("drop", "Humedad", value(first.humidity), "%", "Humedad relativa"),
             ("rain", "Lluvia", value(first.precipitation,"",1), "mm", "Acumulación de esa hora"),
             ("layers", "Probabilidad", value(first.precip_probability), "%", "De lluvia en esa hora")]
    html = '<div class="weather-details">'
    for symbol, label, number, unit, note in items:
        html += f'<article class="detail-card"><div class="detail-label">{icon(symbol)}{text(label)}</div><div class="detail-value">{text(number)} <small>{unit if number != "—" else ""}</small></div><p>{text(note if number != "—" else "Esta fuente no aporta el dato")}</p></article>'
    st.markdown(html + '</div>', unsafe_allow_html=True)


def hourly_strip(main, timezone, title="Próximas horas", selected=False):
    cells = []
    samples = main if selected else main[main.valid_time <= pd.Timestamp.now(tz="UTC") + pd.Timedelta(hours=24)]
    repeated_hours = samples.valid_time.dt.tz_convert(timezone).dt.strftime("%H:%M").duplicated(keep=False) if selected else pd.Series(False, index=samples.index)
    for index, row in samples.iterrows():
        local = row.valid_time.tz_convert(timezone)
        hour = local.strftime("%H:%M %Z" if repeated_hours.loc[index] else "%H:%M")
        kind, _ = weather_kind(row, timezone)
        rain = value(row.precip_probability,"%") if pd.notna(row.precip_probability) else value(row.precipitation," mm",1)
        cells.append(f'<div class="hour-cell"><time>{text(hour)}</time>{weather_svg(kind,44)}<strong>{value(row.temperature,"°")}</strong><span class="hour-feels" title="Sensación térmica estimada a la sombra">Sens. {value(row.get("feels_like"),"°")}</span><span>{text(rain)}</span></div>')
    note = f"{len(samples)} horas disponibles · Desliza" if selected else "Desliza para explorar" if len(samples) else "Sin horas disponibles"
    empty = '<p class="card-footnote">Esta fuente no aporta horas dentro de las próximas 24 horas. Toca un día abajo para explorar las disponibles.</p>' if samples.empty else ""
    st.markdown('<section id="hourly-section" class="hourly-card"><div class="card-heading">' + icon("clock",18) + '<h2>' + text(title) + '</h2><span>' + note + '</span></div><div class="hourly-scroll" tabindex="0" role="region" aria-label="Previsión horaria, desplazamiento horizontal">' + ''.join(cells) + '</div>' + empty + '</section>', unsafe_allow_html=True)


def day_label(day, timezone):
    today = pd.Timestamp.now(tz=timezone).date()
    return "Hoy" if day == today else "Mañana" if (day - today).days == 1 else WEEKDAYS[day.weekday()].capitalize()


def daily_cards(daily, main, timezone, on_select, selected_day=None):
    if daily.empty:
        return
    low, high = daily.minimum.min(), daily.maximum.max()
    span = max(high - low, 1) if pd.notna(low) and pd.notna(high) else 1
    with st.container():
        st.markdown('<div class="daily-anchor card-heading">' + icon("layers",18) + '<h2>Los próximos días</h2><span>Toca un día para ver sus horas</span></div>', unsafe_allow_html=True)
        for _, day in daily.iterrows():
            label = day_label(day.date, timezone)
            samples = main[main.valid_time.dt.tz_convert(timezone).dt.date == day.date]
            representative = samples.iloc[len(samples)//2].copy()
            representative["precipitation"] = day.precipitation
            kind, condition = weather_kind(representative, timezone)
            start = max(0, min(100, 100 * (day.minimum - low) / span)) if pd.notna(day.minimum) and pd.notna(low) else 0
            width = max(3, min(100 - start, 100 * (day.maximum - day.minimum) / span)) if pd.notna(day.maximum) and pd.notna(day.minimum) else 0
            rain = value(day.probability,"%") if pd.notna(day.probability) else value(day.precipitation," mm",1)
            with st.container():
                active = " selected-day" if selected_day == day.date.isoformat() else ""
                st.markdown(f'<div class="day-row{active}"><div><strong>{text(label)}</strong><small>{day.date:%d/%m}</small></div><span title="{text(condition)}">{weather_svg(kind,38)}</span><span class="day-rain">{text(rain)}</span><span class="day-low">{value(day.minimum,"°")}<small class="day-feels" title="Sensación mínima estimada">{value(day.feels_minimum,"°")}</small></span><span class="day-range"><span class="temperature-track"><i style="left:{start:.2f}%;width:{width:.2f}%"></i></span><small>Sensación</small></span><span class="day-high"><strong>{value(day.maximum,"°")}</strong><small class="day-feels" title="Sensación máxima estimada">{value(day.feels_maximum,"°")}</small></span></div>', unsafe_allow_html=True)
                st.button(f"Ver horas de {label}, {day.date:%d/%m}", key=f"forecast_day_{day.date.isoformat()}",
                          on_click=on_select, args=(day.date.isoformat(),), use_container_width=True)
        st.markdown('<p class="card-footnote">Temperatura arriba; sensación estimada a la sombra abajo. Mínima / máxima y lluvia de las horas disponibles. Hoy y el último día pueden estar incompletos. — indica que falta algún dato necesario.</p>', unsafe_allow_html=True)


def empty_state(title, description):
    st.markdown(f'<section class="empty-state">{weather_svg("partly",96)}<h2>{text(title)}</h2><p>{text(description)}</p><div class="empty-steps"><span>1. Guardamos previsiones</span><span>2. Recogemos observaciones</span><span>3. Medimos los aciertos</span></div></section>', unsafe_allow_html=True)
