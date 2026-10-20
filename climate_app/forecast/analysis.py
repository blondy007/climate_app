from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from climate_app.forecast.domain import VARIABLES, horizon, utcnow

EVALUATED = ("temperature", "humidity", "precipitation", "wind_speed", "wind_direction")
FLOORS = {"temperature": 0.5, "humidity": 5, "precipitation": 0.2,
          "wind_speed": 3, "wind_direction": 15, "cloud_cover": 10,
          "wind_gust": 4, "precip_probability": 0.05}


def paired_samples(forecasts, observations, as_of=None):
    """One snapshot per target and horizon. Never validate a hindcast as a forecast."""
    if forecasts.empty or observations.empty:
        return pd.DataFrame()
    as_of = pd.Timestamp(as_of or utcnow())
    f = forecasts[(forecasts.fetched_at < forecasts.valid_time) & (forecasts.valid_time < as_of)].copy()
    o = observations[(observations.valid_time < as_of) & (observations.fetched_at <= as_of)].copy()
    if f.empty or o.empty:
        return pd.DataFrame()
    f["lead_hours"] = (f.valid_time - f.fetched_at).dt.total_seconds() / 3600
    f["horizon"] = f.lead_hours.map(horizon)
    f = f.sort_values("fetched_at").drop_duplicates(["provider", "valid_time", "horizon"], keep="last")
    # A changed observing station cannot silently produce duplicate truth records.
    o = o.sort_values("fetched_at").drop_duplicates("valid_time", keep="last")
    return f.merge(o, on=["location_id", "valid_time"], suffixes=("", "_observed"), how="inner")


def scores(forecasts, observations, as_of=None, common_only=False):
    pairs = paired_samples(forecasts, observations, as_of)
    records = []
    if pairs.empty:
        return pd.DataFrame()
    for variable in EVALUATED:
        good = pairs.dropna(subset=[variable, variable + "_observed"]).copy()
        # Common-time comparison per horizon; providers with no data aren't ranked.
        if common_only and not good.empty:
            counts = good.groupby("horizon").provider.nunique()
            coverage = good.groupby(["horizon", "valid_time"]).provider.nunique()
            allowed = coverage[coverage == coverage.index.get_level_values("horizon").map(counts)].index
            good = good[good.set_index(["horizon", "valid_time"]).index.isin(allowed)]
        for (provider, bucket), group in good.groupby(["provider", "horizon"]):
            error = group[variable] - group[variable + "_observed"]
            if variable == "wind_direction":
                # Direction in calm conditions is not a useful verification target.
                group = group[group.wind_speed_observed >= 3]
                error = ((group[variable] - group[variable + "_observed"] + 180) % 360) - 180
            if not len(error):
                continue
            row = {"provider": provider, "variable": variable, "horizon": bucket,
                   "mae": float(error.abs().mean()), "rmse": float(np.sqrt((error ** 2).mean())),
                   "bias": float(error.mean()), "n": len(error),
                   "days": group.valid_time.dt.date.nunique(), "brier": np.nan,
                   "interval_coverage": np.nan}
            if variable == "precipitation" and "precip_probability" in group:
                rainy = group.dropna(subset=["precip_probability"])
                if not rainy.empty:
                    row["brier"] = float(((rainy.precip_probability / 100 -
                                            (rainy.precipitation_observed >= 0.1).astype(float)) ** 2).mean())
            if variable == "temperature":
                intervals = group.dropna(subset=["temperature_p10", "temperature_p90"])
                if not intervals.empty:
                    row["interval_coverage"] = float(((intervals.temperature_observed >= intervals.temperature_p10) &
                                                       (intervals.temperature_observed <= intervals.temperature_p90)).mean())
            records.append(row)
    return pd.DataFrame(records)


def combine(forecasts, training_scores=None, as_of=None):
    """Regularized inverse-MAE weights, independently for each variable/horizon.

    Missing providers/variables are renormalized at each valid hour. The spread is
    disagreement between models, never labeled as a calibrated confidence interval.
    """
    if forecasts.empty:
        return pd.DataFrame()
    now = pd.Timestamp(as_of or utcnow())
    source = forecasts[~forecasts.provider.isin(["climateapp", "climateapp_private"])].copy()
    source = source[source.valid_time > now]
    source = source[source.fetched_at >= now - pd.Timedelta(hours=9)]
    stats = training_scores if training_scores is not None else pd.DataFrame()
    records = []
    for when, group in source.groupby("valid_time"):
        bucket = horizon((when - now).total_seconds() / 3600)
        row = {"valid_time": when, "provider": "climateapp", "fetched_at": now,
               "location_id": group.location_id.iloc[0], "init_time": pd.NaT,
               "temperature_p10": np.nan, "temperature_p90": np.nan,
               "model_count": group.temperature.notna().sum(), "weights": {}, "trained": False}
        for variable in VARIABLES:
            good = group.dropna(subset=[variable])
            if good.empty:
                row[variable] = np.nan
                continue
            weights, trained = [], []
            candidates = pd.DataFrame()
            if not stats.empty and variable in EVALUATED:
                candidates = stats[(stats.variable == variable) & (stats.horizon == bucket) &
                                   (stats.n >= 30) & (stats.days >= 3) & (stats.provider != "climateapp")]
            # Cold start is equal for ALL models until each contributor has enough evidence.
            use_skill = not candidates.empty and set(good.provider).issubset(set(candidates.provider))
            for provider in good.provider:
                record = candidates[candidates.provider == provider] if use_skill else pd.DataFrame()
                if record.empty:
                    weights.append(1.0)
                else:
                    weights.append(1 / (float(record.mae.iloc[0]) + FLOORS[variable]))
            weights = np.array(weights, dtype=float)
            weights /= weights.sum()
            # Shrink towards equal weights: a small winning sample cannot dominate.
            weights = 0.75 * weights + 0.25 / len(weights)
            values = good[variable].to_numpy(dtype=float)
            if variable == "wind_direction":
                sine = np.sum(weights * np.sin(np.radians(values)))
                cosine = np.sum(weights * np.cos(np.radians(values)))
                row[variable] = np.degrees(np.arctan2(sine, cosine)) % 360 if np.hypot(sine, cosine) > 0.01 else np.nan
            else:
                row[variable] = float(np.sum(weights * values))
            row["weights"][variable] = dict(zip(good.provider, weights.tolist()))
            row["trained"] = row["trained"] or use_skill
            if variable == "temperature":
                row["model_min"] = float(values.min())
                row["model_max"] = float(values.max())
                row["model_std"] = float(np.sqrt(np.sum(weights * (values - row[variable]) ** 2)))
        records.append(row)
    return pd.DataFrame(records)


def climatology(observations, target, forecast_temperature, timezone="Europe/Madrid"):
    if observations.empty or pd.isna(forecast_temperature):
        return None
    local = observations.valid_time.dt.tz_convert(timezone)
    target = pd.Timestamp(target).tz_convert(timezone)
    # A fixed leap year gives a stable calendar index across leap/non-leap years.
    ordinal = pd.to_datetime("2000-" + local.dt.strftime("%m-%d")).dt.dayofyear
    target_day = pd.Timestamp(year=2000, month=target.month, day=target.day).dayofyear
    delta = (ordinal - target_day).abs()
    distance = np.minimum(delta, 366 - delta)
    candidates = observations[(distance <= 15) & (local.dt.hour == target.hour) &
                              (local.dt.year < target.year)].copy()
    candidates = candidates.dropna(subset=["temperature"])
    if len(candidates) < 20:
        return None
    values = candidates.temperature
    return {"mean": float(values.mean()), "delta": float(forecast_temperature - values.mean()),
            "percentile": float(100 * (values < forecast_temperature).mean()), "n": len(values),
            "years": sorted(candidates.valid_time.dt.tz_convert(timezone).dt.year.unique().tolist())}


def daily_summary(frame, timezone):
    if frame.empty:
        return pd.DataFrame()
    frame = frame.copy()
    frame["date"] = frame.valid_time.dt.tz_convert(timezone).dt.date
    return frame.groupby("date").agg(
        minimum=("temperature", "min"), maximum=("temperature", "max"),
        precipitation=("precipitation", lambda s: s.sum(min_count=1)),
        rain_hours=("precipitation", "count"), temperature_hours=("temperature", "count"),
        wind=("wind_speed", "max"), gust=("wind_gust", "max"),
        probability=("precip_probability", "max"),
    ).reset_index()
