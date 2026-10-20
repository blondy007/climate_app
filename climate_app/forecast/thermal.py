"""Comparable apparent temperature derived from each forecast's own inputs."""
from __future__ import annotations

import numpy as np
import pandas as pd

METHOD = "Steadman · sombra"
SOURCE = "https://www.bom.gov.au/info/thermal_stress/#atapproximation"


def with_feels_like(frame):
    """Return a copy with Steadman/BOM apparent temperature in Celsius.

    AT = T + 0.33 e - 0.70 v - 4; e = RH/100 * 6.105 exp(17.27 T/(237.7+T)).
    The archived 10 m wind is km/h and must be converted to m/s. No solar
    radiation is assumed. Missing or invalid inputs remain missing, including
    when only temperature is available. No provider or time interpolation.
    """
    result = frame.copy()
    def numbers(column):
        return pd.to_numeric(result.get(column, pd.Series(np.nan, index=result.index)), errors="coerce")
    temperature, humidity, wind = (numbers(c) for c in ("temperature", "humidity", "wind_speed"))
    valid = (np.isfinite(temperature) & temperature.between(-100, 70) &
             np.isfinite(humidity) & humidity.between(0, 100) &
             np.isfinite(wind) & (wind >= 0))
    result["feels_like"] = np.nan
    t, rh, speed = temperature[valid], humidity[valid], wind[valid] / 3.6
    vapour = rh / 100 * 6.105 * np.exp(17.27 * t / (237.7 + t))
    result.loc[valid, "feels_like"] = t + .33 * vapour - .70 * speed - 4
    return result
