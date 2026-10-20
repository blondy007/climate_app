"""Predeclared radar baseline, evaluated offline before any product integration."""
from __future__ import annotations

import contextlib
from datetime import datetime, timedelta
import io
import math

import numpy as np
from scipy.ndimage import map_coordinates

with contextlib.redirect_stdout(io.StringIO()):
    from pysteps.motion.lucaskanade import dense_lucaskanade
    from pysteps.extrapolation.semilagrangian import extrapolate

ALGORITHM = "opera-lk-rate-v1"
RAIN_THRESHOLD = .1  # mm/h, fixed before examining forecast performance
LEADS = (15, 30, 60)


def availability(metadata, replay=False):
    """Live uses first retrieval; replay uses archive publication + 2 minute margin."""
    end = datetime.fromisoformat(metadata["nominal_end"])
    if replay:
        published = datetime.fromisoformat(metadata["catalog_last_modified"].replace("Z", "+00:00"))
        return max(end, published) + timedelta(minutes=2)
    return max(end, datetime.fromisoformat(metadata["retrieved_at"]))


def time_steps(source_time, issued_at, leads):
    if source_time > issued_at or any(lead <= 0 for lead in leads):
        raise ValueError("Cannot issue a forecast using future observations")
    aligned_issue = datetime.fromtimestamp(math.ceil(issued_at.timestamp()/300)*300, issued_at.tzinfo)
    return [(aligned_issue + timedelta(minutes=lead) - source_time).total_seconds() / 300 for lead in leads]


def advect(field, velocity, source_time, issued_at, leads):
    return extrapolate(field, velocity, time_steps(source_time, issued_at, leads),
                       allow_nonfinite_values=True, outval=np.nan)


def rate_grid_velocity(velocity, dbzh_meta, rate_meta, site):
    """Resample physical motion onto the rain-rate grid, accounting for pixel centres."""
    if dbzh_meta["projection"] != rate_meta["projection"]:
        raise ValueError("Radar projections changed")
    origin = rate_meta["sites"][site]["crop_origin_rc"]
    shape = rate_meta["sites"][site]["shape"]
    yy, xx = np.indices(shape)
    step = rate_meta["grid_km"] * 1000
    x = rate_meta["global_origin_xy"][0] + (xx + origin[1] + .5) * step
    y = rate_meta["global_origin_xy"][1] + (rate_meta["global_shape"][0] - yy - origin[0] - .5) * step
    db_step = dbzh_meta["grid_km"] * 1000
    db_origin = dbzh_meta["sites"][site]["crop_origin_rc"]
    x = (x - dbzh_meta["global_origin_xy"][0]) / db_step - db_origin[1] - .5
    y = dbzh_meta["global_shape"][0] - (y - dbzh_meta["global_origin_xy"][1]) / db_step - db_origin[0] - .5
    # Motion is in 1 km pixels per 5 min; output must be in rain-grid pixels per 5 min.
    return np.stack([map_coordinates(component, [y, x], order=1, mode="nearest") * db_step / step
                     for component in velocity])


def forecast(dbzh_frames, dbzh_metadata, rate_frame, rate_metadata, site, issued_at, leads=LEADS, replay=False):
    if len(dbzh_frames) != 3 or len(dbzh_metadata) != 3:
        raise ValueError("Exactly three radar images required")
    times = [datetime.fromisoformat(meta["nominal_end"]) for meta in dbzh_metadata]
    rate_time = datetime.fromisoformat(rate_metadata["nominal_end"])
    if any(availability(meta, replay) > issued_at for meta in [*dbzh_metadata, rate_metadata]):
        raise ValueError("Source had not yet been available at issue time")
    if any(right-left != timedelta(minutes=5) for left, right in zip(times, times[1:])):
        raise ValueError("Missing or irregular radar sequence")
    if issued_at-times[-1] > timedelta(minutes=20) or issued_at-rate_time > timedelta(minutes=45):
        raise ValueError("Stale radar observations")
    if issued_at-datetime.fromisoformat(dbzh_metadata[-1]["observation_start"]) > timedelta(minutes=30):
        raise ValueError("Reflectivity scan window too old")
    stack = np.stack(dbzh_frames).astype(float)
    centre = rate_metadata["sites"][site]["centre"]
    db_row, db_col = dbzh_metadata[-1]["sites"][site]["centre"]
    if any(np.isfinite(frame[db_row-10:db_row+11, db_col-10:db_col+11]).mean() < .95 for frame in stack):
        raise ValueError("Insufficient radar coverage")
    if not np.isfinite(rate_frame[centre[0]-1:centre[0]+2, centre[1]-1:centre[1]+2]).all():
        raise ValueError("No coverage at research location")
    # Keep missing regions masked in optical flow and advection, never fill them as dry.
    images = np.ma.masked_invalid(np.clip(stack, 0, 60))
    velocity = dense_lucaskanade(images)
    if not np.isfinite(velocity).all() or np.percentile(np.hypot(*velocity), 99) * 12 * dbzh_metadata[-1]["grid_km"] > 150:
        raise ValueError("Unreliable motion estimate")
    rate_velocity = rate_grid_velocity(velocity, dbzh_metadata[-1], rate_metadata, site)
    predicted = advect(rate_frame, rate_velocity, rate_time, issued_at, leads)
    values = []
    aligned_issue = datetime.fromtimestamp(math.ceil(issued_at.timestamp()/300)*300, issued_at.tzinfo)
    row, col = centre
    for lead, field in zip(leads, predicted):
        point = field[row, col]
        nearby = field[row-1:row+2, col-1:col+2]
        valid_time = aligned_issue+timedelta(minutes=lead)
        values.append({"lead_minutes": lead, "effective_lead_minutes": (valid_time-issued_at).total_seconds()/60,
                       "valid_time": valid_time.isoformat(),
                       "rain_rate": float(point) if np.isfinite(point) else None,
                       "persistence_rate": float(rate_frame[row, col]),
                       "nearby_rain_rate": float(np.max(nearby)) if np.isfinite(nearby).all() else None,
                       "nearby_persistence_rate": float(np.max(rate_frame[row-1:row+2,col-1:col+2]))})
    return {"algorithm": ALGORITHM, "site": site, "issued_at": issued_at.isoformat(),
            "mode": "retrospective" if replay else "prospective", "approved_for_public_use": False,
            "source_dbzh": [meta["nominal_end"] for meta in dbzh_metadata], "source_rate": rate_metadata["nominal_end"],
            "source_crop_hashes": [meta["sha256_crop"] for meta in [*dbzh_metadata, rate_metadata]],
            "rain_source_age_minutes": (issued_at-rate_time).total_seconds()/60,
            "velocity_p95_kmh": float(np.percentile(np.hypot(*velocity),95) * 12 * dbzh_metadata[-1]["grid_km"]),
            "points": values}


def categorical(rows, field="rain_rate", threshold=RAIN_THRESHOLD):
    counts = dict(hits=0, misses=0, false_alarms=0, correct_dry=0, unavailable=0)
    for row in rows:
        predicted, observed = row.get(field), row.get("observed_rate")
        if predicted is None or observed is None or not math.isfinite(predicted) or not math.isfinite(observed):
            counts["unavailable"] += 1
        elif observed >= threshold:
            counts["hits" if predicted >= threshold else "misses"] += 1
        else:
            counts["false_alarms" if predicted >= threshold else "correct_dry"] += 1
    h, m, f = (counts[key] for key in ("hits", "misses", "false_alarms"))
    return {**counts, "csi": h/(h+m+f) if h+m+f else None,
            "probability_of_detection": h/(h+m) if h+m else None,
            "false_alarm_ratio": f/(h+f) if h+f else None}
