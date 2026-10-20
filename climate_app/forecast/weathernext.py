"""WeatherNext 3 adapter. All query paths fail closed if billing is not disabled."""
from __future__ import annotations

import math
import os
import re
from datetime import timedelta

from climate_app.forecast.domain import Forecast, ProviderError, finite, utcnow


def settings():
    return {key: os.environ.get(key, "").strip() for key in (
        "WEATHERNEXT_PROJECT", "WEATHERNEXT_DATASET", "WEATHERNEXT_LOCATION")}


def configured():
    config = settings()
    return bool(config["WEATHERNEXT_PROJECT"] and config["WEATHERNEXT_DATASET"])


def require_unbilled(session, project: str):
    response = session.get(f"https://cloudbilling.googleapis.com/v1/projects/{project}/billingInfo", timeout=15)
    if response.status_code != 200:
        raise ProviderError("WeatherNext bloqueado: no se pudo verificar que Google Cloud tenga la facturación deshabilitada")
    info = response.json()
    if info.get("billingEnabled") is not False:
        raise ProviderError("WeatherNext bloqueado por la política de 0 €: el proyecto tiene facturación habilitada o desconocida")


def query_sql(dataset: str) -> str:
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,62}\.[A-Za-z0-9_]+", dataset):
        raise ProviderError("WEATHERNEXT_DATASET debe ser proyecto.dataset, sin SQL ni caracteres especiales")
    # Select station-head and grid data independently; choose the latest run that
    # actually covers each valid hour (interim runs only cover 48 hours).
    return f"""
    WITH surface AS (
      SELECT t.init_time, t.geography, f.time,
        f.temperature_2m_p50, f.temperature_2m_p10, f.temperature_2m_p90,
        f.dewpoint_temperature_2m_p50, f.wind_speed_10m_p50,
        f.u_component_of_wind_10m_mean, f.v_component_of_wind_10m_mean,
        f.total_precipitation_1hr_mean, f.total_cloud_cover_p50
      FROM `{dataset}.weathernext_3_0_0_0p1deg` t, UNNEST(t.forecast) f
      WHERE t.init_time BETWEEN @cutoff AND @now
        AND ST_DWITHIN(t.geography, ST_GEOGPOINT(@lon,@lat), 20000)
        AND ST_INTERSECTS(t.geography_polygon, ST_GEOGPOINT(@lon,@lat))
        AND f.time > @now AND f.time <= TIMESTAMP_ADD(@now, INTERVAL 360 HOUR)
      QUALIFY ROW_NUMBER() OVER (PARTITION BY f.time ORDER BY t.init_time DESC,
        ST_DISTANCE(t.geography,ST_GEOGPOINT(@lon,@lat)))=1
    ), station AS (
      SELECT t.init_time, t.geography, f.time,
        f.station_head_temperature_2m_p50, f.station_head_temperature_2m_p10,
        f.station_head_temperature_2m_p90, f.station_head_dewpoint_temperature_2m_p50
      FROM `{dataset}.weathernext_3_0_0_0p05deg` t, UNNEST(t.forecast) f
      WHERE t.init_time BETWEEN @cutoff AND @now
        AND ST_DWITHIN(t.geography, ST_GEOGPOINT(@lon,@lat), 20000)
        AND ST_INTERSECTS(t.geography_polygon, ST_GEOGPOINT(@lon,@lat))
        AND f.time > @now AND f.time <= TIMESTAMP_ADD(@now, INTERVAL 360 HOUR)
      QUALIFY ROW_NUMBER() OVER (PARTITION BY f.time ORDER BY t.init_time DESC,
        ST_DISTANCE(t.geography,ST_GEOGPOINT(@lon,@lat)))=1
    )
    SELECT s.time, s.init_time,
      IF(h.station_head_temperature_2m_p50 IS NOT NULL,h.init_time,s.init_time) AS temperature_init_time,
      ST_Y(s.geography) AS surface_latitude, ST_X(s.geography) AS surface_longitude,
      ST_Y(IF(h.station_head_temperature_2m_p50 IS NOT NULL,h.geography,s.geography)) AS temperature_latitude,
      ST_X(IF(h.station_head_temperature_2m_p50 IS NOT NULL,h.geography,s.geography)) AS temperature_longitude,
      IF(h.station_head_temperature_2m_p50 IS NOT NULL,h.station_head_temperature_2m_p50,s.temperature_2m_p50) AS temp_k,
      IF(h.station_head_temperature_2m_p50 IS NOT NULL,h.station_head_temperature_2m_p10,s.temperature_2m_p10) AS temp_p10_k,
      IF(h.station_head_temperature_2m_p50 IS NOT NULL,h.station_head_temperature_2m_p90,s.temperature_2m_p90) AS temp_p90_k,
      IF(h.station_head_temperature_2m_p50 IS NOT NULL,h.station_head_dewpoint_temperature_2m_p50,s.dewpoint_temperature_2m_p50) AS dew_k,
      s.wind_speed_10m_p50 AS wind_mps,
      s.u_component_of_wind_10m_mean AS u, s.v_component_of_wind_10m_mean AS v,
      s.total_precipitation_1hr_mean AS precip_m,
      s.total_cloud_cover_p50 AS cloud_fraction
    FROM surface s LEFT JOIN station h USING(time)
    ORDER BY s.time
    """


def parse_rows(rows, fetched_at=None) -> Forecast:
    points, inits = [], set()
    for row in rows:
        row = dict(row)
        temp, dew = finite(row.get("temp_k")), finite(row.get("dew_k"))
        if temp is None:
            continue
        temp -= 273.15
        humidity = None
        if dew is not None:
            dew -= 273.15
            humidity = min(100, max(0, 100 * math.exp(17.625 * dew / (243.04 + dew)
                                                     - 17.625 * temp / (243.04 + temp))))
        u, v = finite(row.get("u")), finite(row.get("v"))
        point = {"valid_time": row["time"], "temperature": temp, "humidity": humidity}
        point["source_init_time"] = row.get("init_time")
        point["temperature_init_time"] = row.get("temperature_init_time", row.get("init_time"))
        for coordinate in ("surface_latitude", "surface_longitude", "temperature_latitude", "temperature_longitude"):
            point[coordinate] = finite(row.get(coordinate))
        for source, target, scale, offset in (
            ("temp_p10_k", "temperature_p10", 1, -273.15),
            ("temp_p90_k", "temperature_p90", 1, -273.15),
            ("wind_mps", "wind_speed", 3.6, 0), ("precip_m", "precipitation", 1000, 0),
            ("cloud_fraction", "cloud_cover", 100, 0)):
            value = finite(row.get(source))
            point[target] = value * scale + offset if value is not None else None
        point["wind_direction"] = (math.degrees(math.atan2(-u, -v)) + 360) % 360 if u is not None and v is not None and (u or v) else None
        points.append(point)
        if row.get("init_time"):
            inits.add(str(row["init_time"]))
    if not points:
        raise ProviderError("WeatherNext: todavía no hay datos disponibles para este lugar")
    return Forecast("weathernext3", fetched_at or utcnow(), points, metadata={
        "source": "Google DeepMind WeatherNext 3", "initializations": sorted(inits),
        "timing": "snapshot", "temperature_statistic": "p50", "rain_statistic": "mean",
        "humidity": "Aproximación derivada de medianas de temperatura y punto de rocío",
        "wind_direction": "Dirección del vector medio; velocidad p50",
        "precipitation_interval": "preceding_hour", "uncertainty": "Native ensemble p10/p90",
    })


def fetch_weathernext(location, store):
    if not configured():
        raise ProviderError("WeatherNext 3 pendiente de acceso y configuración de Google")
    from google import auth
    from google.auth.transport.requests import AuthorizedSession
    from google.cloud import bigquery

    config, now = settings(), utcnow()
    project = config["WEATHERNEXT_PROJECT"]
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,62}", project):
        raise ProviderError("Identificador del proyecto Google inválido")
    try:
        credentials, _ = auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        with AuthorizedSession(credentials) as session:
            require_unbilled(session, project)
        client = bigquery.Client(project=project, credentials=credentials,
                                 location=config["WEATHERNEXT_LOCATION"] or "US")
        parameters = [bigquery.ScalarQueryParameter("lat", "FLOAT64", location.latitude),
                      bigquery.ScalarQueryParameter("lon", "FLOAT64", location.longitude),
                      bigquery.ScalarQueryParameter("now", "TIMESTAMP", now),
                      bigquery.ScalarQueryParameter("cutoff", "TIMESTAMP", now - timedelta(hours=24))]
        sql = query_sql(config["WEATHERNEXT_DATASET"])
        cap = 2 * 1024 ** 3
        dry = client.query(sql, job_config=bigquery.QueryJobConfig(
            dry_run=True, use_query_cache=False, query_parameters=parameters,
            maximum_bytes_billed=cap), timeout=45)
        estimate = dry.total_bytes_processed
        if estimate is None or estimate > cap:
            raise ProviderError("WeatherNext: consulta bloqueada por el límite preventivo de 2 GiB")
        # Reserve the entire possible billable cap, not the optimistic dry-run estimate.
        period = now.strftime("%Y-%m")
        if not store.reserve("bigquery-bytes", period, cap, 900 * 1024 ** 3):
            raise ProviderError("WeatherNext: presupuesto preventivo mensual de consultas agotado")
        job = client.query(sql, job_config=bigquery.QueryJobConfig(
            maximum_bytes_billed=cap, query_parameters=parameters, use_query_cache=True), timeout=45)
        result = parse_rows(job.result(timeout=90), fetched_at=utcnow())
        result.metadata["bytes_processed"] = job.total_bytes_processed
        billed = job.total_bytes_billed
        if billed is not None:
            store.release("bigquery-bytes", period, cap - min(cap, billed))
        return result
    except ProviderError:
        raise
    except Exception:
        # Google errors can include principal identifiers and credential paths.
        raise ProviderError("WeatherNext no disponible: comprobar acceso, credenciales privadas, permisos y esquema BigQuery") from None
