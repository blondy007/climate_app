import pandas as pd

from climate_app.shared.models import FilterSelections


def apply_filters(df: pd.DataFrame, selections: FilterSelections, date_range) -> pd.DataFrame:
    filtered = df.copy()

    if date_range:
        fecha_series = filtered.get("Fecha")
        if fecha_series is None:
            return filtered.iloc[0:0]
        start_date, end_date = date_range
        filtered = filtered[(fecha_series >= start_date) & (fecha_series <= end_date)]

    if selections.months:
        months_series = filtered.get("Mes")
        if months_series is None:
            return filtered.iloc[0:0]
        filtered = filtered[months_series.isin(selections.months)]

    if selections.hours:
        hours_series = filtered.get("Hora")
        if hours_series is None:
            return filtered.iloc[0:0]
        filtered = filtered[hours_series.isin(selections.hours)]

    if selections.cities:
        cities_series = filtered.get("Lugar")
        if cities_series is None:
            return filtered.iloc[0:0]
        filtered = filtered[cities_series.isin(selections.cities)]

    if "Todos" not in selections.scenarios and selections.scenarios:
        scenario_series = filtered.get("escenario")
        if scenario_series is None:
            return filtered.iloc[0:0]
        filtered = filtered[scenario_series.isin(selections.scenarios)]

    filtered = filtered.copy()
    if "Mes" in filtered.columns:
        filtered["Mes"] = pd.Categorical(filtered["Mes"], categories=selections.months, ordered=True)
    if "Lugar" in filtered.columns:
        filtered["Lugar"] = pd.Categorical(filtered["Lugar"], categories=selections.cities, ordered=False)
    return filtered
