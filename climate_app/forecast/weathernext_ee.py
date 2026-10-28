"""Small native-grid point series from the noncommercial Earth Engine service."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from climate_app.forecast.domain import ProviderError, finite, stamp, utcnow
from climate_app.forecast.weathernext import parse_rows

COLLECTION = "projects/gcp-public-data-weathernext/assets/weathernext_3_0_0_"
SURFACE = {
    "temperature_2m_p50": "temp_k", "temperature_2m_p10": "temp_p10_k",
    "temperature_2m_p90": "temp_p90_k", "dewpoint_temperature_2m_p50": "dew_k",
    "wind_speed_10m_p50": "wind_mps", "u_component_of_wind_10m_mean": "u",
    "v_component_of_wind_10m_mean": "v", "total_precipitation_1hr_mean": "precip_m",
    "total_cloud_cover_p50": "cloud_fraction",
}
STATION = {
    "station_head_temperature_2m_p50": "temp_k",
    "station_head_temperature_2m_p10": "temp_p10_k",
    "station_head_temperature_2m_p90": "temp_p90_k",
    "station_head_dewpoint_temperature_2m_p50": "dew_k",
}


def require_noncommercial(session, project):
    response = session.get(f"https://earthengine.googleapis.com/v1/projects/{project}/config", timeout=15)
    if response.status_code != 200 or response.json().get("registrationState") != "REGISTERED_NOT_COMMERCIALLY":
        raise ProviderError("WeatherNext bloqueado: Earth Engine debe estar registrado para uso no comercial")


def point_series(ee, location, suffix, bands, now):
    point = ee.Geometry.Point([location.longitude, location.latitude])
    iso = lambda value: stamp(value).replace("+00:00", "Z")
    collection = (ee.ImageCollection(COLLECTION + suffix)
                  .filter(ee.Filter.gte("start_time", iso(now - timedelta(hours=24))))
                  .filter(ee.Filter.lte("start_time", iso(now)))
                  # Actual assets index system:time_start by initialization;
                  # end_time, not that index, is the forecast's valid timestamp.
                  .filter(ee.Filter.gt("end_time", iso(now)))
                  .filter(ee.Filter.lte("end_time", iso(now + timedelta(hours=360))))
                  .filterBounds(point))
    # Ordered join chooses the newest init independently for every valid hour,
    # including older full runs after the newest interim run's 48-hour horizon.
    latest = ee.Join.saveFirst(matchKey="_latest", ordering="start_time", ascending=False).apply(
        primary=collection.distinct("end_time"), secondary=collection,
        condition=ee.Filter.equals(leftField="end_time", rightField="end_time"))
    images = ee.ImageCollection(latest.map(lambda image: ee.Image(image.get("_latest"))))

    def select(image):
        return image.select(list(bands)).addBands(
            ee.Image.constant(ee.Date(image.get("start_time")).millis()).toDouble().rename("source_init_ms")).set(
                "system:time_start", ee.Date(image.get("end_time")).millis())

    selected = images.sort("end_time").limit(360).map(select)
    projection = ee.Image(selected.first()).select(0).projection().getInfo()
    # Supplying the native transform avoids rescaling/interpolating the source grid.
    return selected.getRegion(point, crs=projection["crs"], crsTransform=projection["transform"]).getInfo()


def decode_region(table, bands, now):
    if not table:
        return []
    header, *values = table
    required = {"longitude", "latitude", "time", "source_init_ms", *bands}
    if not required.issubset(header) or len(values) > 360:
        raise ProviderError("WeatherNext: respuesta puntual incompatible")
    rows = []
    for values_row in values:
        if len(values_row) != len(header):
            raise ProviderError("WeatherNext: respuesta puntual incompleta")
        record = dict(zip(header, values_row))
        valid = datetime.fromtimestamp(record["time"] / 1000, timezone.utc)
        init = datetime.fromtimestamp(record["source_init_ms"] / 1000, timezone.utc)
        if not (now < valid <= now + timedelta(hours=360) and now - timedelta(hours=24) <= init <= now):
            raise ProviderError("WeatherNext: fechas fuera del intervalo solicitado")
        rows.append({"time": valid, "init_time": init,
                     "latitude": finite(record["latitude"]), "longitude": finite(record["longitude"]),
                     **{target: finite(record.get(source)) for source, target in bands.items()}})
    return rows


def merge_grids(surface, station):
    stations = {row["time"]: row for row in station}
    result = []
    for row in surface:
        row = dict(row)
        row["surface_latitude"], row["surface_longitude"] = row.pop("latitude"), row.pop("longitude")
        head = stations.get(row["time"])
        if head and head.get("temp_k") is not None:
            row.update({key: head.get(key) for key in STATION.values()})
            row["temperature_init_time"] = head["init_time"]
            row["temperature_latitude"], row["temperature_longitude"] = head["latitude"], head["longitude"]
        else:
            row["temperature_init_time"] = row["init_time"]
            row["temperature_latitude"], row["temperature_longitude"] = row["surface_latitude"], row["surface_longitude"]
        result.append(row)
    return result


def fetch_point(location, store, credentials, project, now):
    import ee

    # A shared transactional limit also bounds repeated failures and all saved places.
    if not store.reserve("earthengine-forecasts", now.strftime("%Y-%m-%d"), 1, 240):
        raise ProviderError("WeatherNext: límite preventivo diario de consultas agotado")
    ee.Initialize(credentials=credentials, project=project)
    ee.data.setDeadline(90000)
    ee.data.setMaxRetries(1)
    surface = decode_region(point_series(ee, location, "0p1deg", SURFACE, now), SURFACE, now)
    station = decode_region(point_series(ee, location, "0p05deg", STATION, now), STATION, now)
    forecast = parse_rows(merge_grids(surface, station), fetched_at=utcnow())
    forecast.metadata.update({"access": "earthengine", "grid_sampling": "native pixel; no interpolation",
                              "registration": "noncommercial", "billing_enabled": False})
    return forecast
