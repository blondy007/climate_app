import pandas as pd

from climate_app.shared.models import FilterSelections


def apply_filters(df: pd.DataFrame, selections: FilterSelections, date_range) -> pd.DataFrame:
    filtered = df.copy()
    if date_range:
        start_date, end_date = date_range
        filtered = filtered[(filtered["Fecha"] >= start_date) & (filtered["Fecha"] <= end_date)]

    if selections.months:
        filtered = filtered[filtered["Mes"].isin(selections.months)]
    else:
        filtered = filtered.iloc[0:0]

    if selections.hours:
        filtered = filtered[filtered["Hora"].isin(selections.hours)]
    else:
        filtered = filtered.iloc[0:0]

    if selections.cities:
        filtered = filtered[filtered["Lugar"].isin(selections.cities)]
    else:
        filtered = filtered.iloc[0:0]

    if "Todos" not in selections.scenarios and selections.scenarios:
        filtered = filtered[filtered["escenario"].isin(selections.scenarios)]

    filtered = filtered.copy()
    if "Mes" in filtered.columns:
        filtered["Mes"] = pd.Categorical(filtered["Mes"], categories=selections.months, ordered=True)
    if "Lugar" in filtered.columns:
        filtered["Lugar"] = pd.Categorical(filtered["Lugar"], categories=selections.cities, ordered=False)
    return filtered
