from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

from climate_app.services.analytics import (
    AGG_LABELS,
    aggregate_series,
    build_summary_markdown,
    determine_granularity,
    format_axis_values,
    kpis,
)
from climate_app.shared.models import FilterSelections


def render_dashboard(
    filtered: pd.DataFrame,
    selections: FilterSelections,
    date_range: Optional[Tuple],
    agg_option: str,
    source_label: str,
) -> None:
    agg_label = AGG_LABELS.get(agg_option, agg_option)

    if not date_range:
        st.info("Selecciona fecha de inicio.")
        return

    start_date, end_date = date_range
    period_start = filtered["Fecha"].min() if not filtered.empty else start_date
    period_end = filtered["Fecha"].max() if not filtered.empty else end_date
    if period_start == period_end:
        period_label = f"{period_start:%Y/%m/%d}"
    else:
        period_label = f"{period_start:%Y/%m/%d} al {period_end:%Y/%m/%d}"
    st.markdown(f"*Indicadores agregados por ciudad ({agg_label}) sobre el periodo {period_label}.*")

    if filtered.empty:
        st.info("No hay datos con los filtros actuales.")
        return

    granularity = determine_granularity(start_date, end_date)
    aggregated = aggregate_series(filtered, agg_option, granularity)
    if granularity == "monthly" and "Mes" in aggregated.columns and aggregated["Mes"].nunique() <= 1:
        granularity = "daily"
        aggregated = aggregate_series(filtered, agg_option, granularity)
    kpi_data = kpis(filtered, agg_option)
    missing_logs: List[str] = []

    humidity_available = "humedad_%" in filtered.columns and filtered["humedad_%"].notna().any()
    wind_available = "wspd_kmh" in filtered.columns and filtered["wspd_kmh"].notna().any()

    missing_metrics = []
    if not humidity_available:
        missing_metrics.append("humedad")
    if not wind_available:
        missing_metrics.append("viento")
    if missing_metrics:
        st.warning("Faltan datos de {}. Se omiten los gráficos relacionados.".format(" y ".join(missing_metrics)))

    if kpi_data.empty:
        st.info("No hay datos con los filtros actuales.")
    else:
        summary_table = (
            kpi_data.rename(
                columns={
                    "Lugar": "Ciudad",
                    "Temp (°C)": "Temp (°C)",
                    "Se siente (°C)": "Se siente (°C)",
                    "Humedad_%": "Humedad (%)",
                    "Viento (km/h)": "Viento (km/h)",
                }
            )
            .set_index("Ciudad")
            .applymap(lambda value: f"{value:.1f}" if pd.notna(value) else "-")
        )
        st.table(summary_table)

    x_field_map = {
        "monthly": ("Mes", "Mes"),
        "daily": ("Fecha", "Fecha"),
        "hourly": ("FechaHora", "Fecha y hora"),
    }
    x_field, x_axis_title = x_field_map[granularity]
    if x_field not in aggregated.columns:
        aggregated[x_field] = pd.Series(dtype=object)
    x_sequence = format_axis_values(aggregated[x_field], granularity)
    category_array = list(dict.fromkeys(x_sequence))

    value_columns = [col for col in ["Temp (°C)", "Se siente (°C)"] if col in aggregated.columns]
    has_temperature_data = bool(value_columns) and not aggregated[value_columns].dropna(how="all").empty

    title_map = {
        "monthly": "Evolución mensual",
        "daily": "Evolución diaria",
        "hourly": "Evolución horaria",
    }

    if has_temperature_data:
        fig = go.Figure()
        for city in selections.cities:
            city_data = aggregated[aggregated["Lugar"] == city]
            if city_data.empty:
                continue
            x_values_city = format_axis_values(city_data[x_field], granularity)
            fig.add_trace(
                go.Scatter(
                    x=x_values_city,
                    y=city_data["Temp (°C)"],
                    mode="lines+markers",
                    name=f"{city} - Temp",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=x_values_city,
                    y=city_data["Se siente (°C)"],
                    mode="lines+markers",
                    line=dict(dash="dash"),
                    name=f"{city} - Se siente",
                )
            )
        fig.update_layout(
            title=f"{title_map[granularity]} - {agg_label}",
            xaxis_title=x_axis_title,
            yaxis_title="°C",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
            margin=dict(t=60, b=110, l=60, r=20),
            height=450,
        )
        fig.update_xaxes(categoryorder="array", categoryarray=category_array)
        st.plotly_chart(fig, use_container_width=True, theme="streamlit")

        delta_df = aggregated.copy()
        delta_df["Delta (°C)"] = delta_df["Se siente (°C)"] - delta_df["Temp (°C)"]
        delta_available = delta_df.dropna(subset=["Delta (°C)"])
        if not delta_available.empty:
            fig_delta = go.Figure()
            for city in selections.cities:
                city_data = delta_available[delta_available["Lugar"] == city]
                if city_data.empty:
                    continue
                x_values_city = format_axis_values(city_data[x_field], granularity)
                if granularity == "monthly":
                    fig_delta.add_trace(go.Bar(x=x_values_city, y=city_data["Delta (°C)"], name=city))
                else:
                    fig_delta.add_trace(
                        go.Scatter(
                            x=x_values_city,
                            y=city_data["Delta (°C)"],
                            mode="lines+markers",
                            name=city,
                        )
                    )
            fig_delta.update_layout(
                title="Diferencia Se siente vs Temp",
                xaxis_title=x_axis_title,
                yaxis_title="°C",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(t=60, b=110, l=60, r=20),
                height=420,
            )
            fig_delta.update_xaxes(categoryorder="array", categoryarray=category_array)
            st.plotly_chart(fig_delta, use_container_width=True, theme="streamlit")
    else:
        st.info("Sin datos agregados para mostrar en el gráfico de temperatura.")

    if humidity_available and "Humedad_%" in aggregated.columns and aggregated["Humedad_%"].notna().any():
        fig_humidity = go.Figure()
        for city in selections.cities:
            city_data = aggregated[aggregated["Lugar"] == city]
            if city_data.empty or city_data["Humedad_%"].dropna().empty:
                continue
            x_values_city = format_axis_values(city_data[x_field], granularity)
            fig_humidity.add_trace(
                go.Scatter(x=x_values_city, y=city_data["Humedad_%"], mode="lines+markers", name=city)
            )
        if fig_humidity.data:
            fig_humidity.update_layout(
                title="Evolución de la humedad",
                xaxis_title=x_axis_title,
                yaxis_title="%",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(t=60, b=110, l=60, r=20),
                height=420,
            )
            fig_humidity.update_xaxes(categoryorder="array", categoryarray=category_array)
            st.plotly_chart(fig_humidity, use_container_width=True, theme="streamlit")

    if wind_available and "Viento (km/h)" in aggregated.columns and aggregated["Viento (km/h)"].notna().any():
        fig_wind = go.Figure()
        for city in selections.cities:
            city_data = aggregated[aggregated["Lugar"] == city]
            if city_data.empty or city_data["Viento (km/h)"].dropna().empty:
                continue
            x_values_city = format_axis_values(city_data[x_field], granularity)
            fig_wind.add_trace(
                go.Scatter(x=x_values_city, y=city_data["Viento (km/h)"], mode="lines+markers", name=city)
            )
        if fig_wind.data:
            fig_wind.update_layout(
                title="Evolución del viento",
                xaxis_title=x_axis_title,
                yaxis_title="km/h",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
                margin=dict(t=60, b=110, l=60, r=20),
                height=420,
            )
            fig_wind.update_xaxes(categoryorder="array", categoryarray=category_array)
            st.plotly_chart(fig_wind, use_container_width=True, theme="streamlit")

    if value_columns:
        missing_rows = aggregated[aggregated[value_columns].isna().all(axis=1)]
        for _, missing in missing_rows.iterrows():
            raw_value = missing.get(x_field)
            if pd.isna(raw_value):
                label = "sin tiempo"
            else:
                labels = format_axis_values(pd.Series([raw_value]), granularity)
                label = labels[0] if labels else str(raw_value)
            missing_logs.append(f"Sin datos para {missing['Lugar']} en {label} con los filtros actuales")

    if not aggregated.empty:
        suffix_map = {"monthly": "mensual", "daily": "diario", "hourly": "horario"}
        file_suffix = suffix_map[granularity]
        st.download_button(
            f"Descargar datos agregados ({file_suffix})",
            aggregated.to_csv(index=False).encode("utf-8"),
            file_name=f"datos_agregados_{file_suffix}.csv",
            mime="text/csv",
        )

    filters_description = {
        "Meses": ", ".join(selections.months) if selections.months else "Todos",
        "Horas": ", ".join(selections.hours) if selections.hours else "Todas",
        "Ciudades": ", ".join(selections.cities) if selections.cities else "Todas",
        "Escenarios": "Todos" if "Todos" in selections.scenarios else ", ".join(selections.scenarios),
    }

    summary_md = build_summary_markdown(
        period_label,
        agg_label,
        filters_description,
        kpi_data,
        missing_logs,
        len(filtered),
    )
    st.download_button(
        "Descargar resumen (Markdown)",
        summary_md.encode("utf-8"),
        file_name="resumen_meteostat.md",
        mime="text/markdown",
    )

    if selections.show_distribution:
        st.subheader("Distribución de temperatura y sensación térmica")
        if filtered.empty:
            st.info("No hay datos para el gráfico de distribución.")
        else:
            temp_traces = []
            feel_traces = []
            for city in selections.cities:
                temp_values = pd.to_numeric(filtered.loc[filtered["Lugar"] == city, "temp"], errors="coerce").dropna()
                feel_values = pd.to_numeric(
                    filtered.loc[filtered["Lugar"] == city, "feels_like"], errors="coerce"
                ).dropna()
                if not temp_values.empty:
                    temp_traces.append((city, temp_values))
                if not feel_values.empty:
                    feel_traces.append((city, feel_values))

            if temp_traces or feel_traces:
                fig_dist = make_subplots(
                    rows=1,
                    cols=2,
                    shared_yaxes=True,
                    subplot_titles=["Temp (°C)", "Se siente (°C)"],
                )
                for city, values in temp_traces:
                    fig_dist.add_trace(go.Box(y=values, name=city, boxpoints="outliers"), row=1, col=1)
                for city, values in feel_traces:
                    fig_dist.add_trace(go.Box(y=values, name=city, boxpoints="outliers"), row=1, col=2)
                fig_dist.update_layout(
                    legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
                    margin=dict(t=80, b=100, l=60, r=20),
                )
                st.plotly_chart(fig_dist, use_container_width=True, theme="streamlit")
            else:
                st.info("No hay suficientes datos para el gráfico de distribución.")

            if {"datetime", "Lugar", "temp", "feels_like", "escenario"}.issubset(filtered.columns):
                distribution_export = filtered[["datetime", "Lugar", "temp", "feels_like", "escenario"]].copy()
                st.download_button(
                    "Descargar datos de distribución (CSV)",
                    distribution_export.to_csv(index=False).encode("utf-8"),
                    file_name="distribucion.csv",
                    mime="text/csv",
                )

    if missing_logs:
        with st.expander("Incidencias de datos", expanded=False):
            for message in sorted(set(missing_logs)):
                st.markdown(f"- {message}")

    table_columns = ["datetime", "Lugar", "temp", "humedad_%", "wspd_kmh", "feels_like", "escenario"]
    available_columns = [col for col in table_columns if col in filtered.columns]
    table_df = filtered[available_columns].sort_values("datetime") if not filtered.empty else filtered[available_columns]

    st.subheader("Datos crudos filtrados")
    st.caption(f"Fuente: {source_label}")
    st.dataframe(table_df, use_container_width=True)
    st.download_button(
        "Descargar tabla filtrada (CSV)",
        table_df.to_csv(index=False).encode("utf-8"),
        file_name="datos_filtrados.csv",
        mime="text/csv",
    )
