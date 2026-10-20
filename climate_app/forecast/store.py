from __future__ import annotations

import hashlib
import json
import os
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict
from datetime import timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from climate_app.data.paths import DATA_DIR
from climate_app.forecast.domain import Forecast, Location, finite, stamp, utcnow

SCHEMA = """
CREATE TABLE IF NOT EXISTS locations (
 id TEXT PRIMARY KEY, name TEXT NOT NULL, latitude REAL NOT NULL, longitude REAL NOT NULL,
 timezone TEXT NOT NULL, station_id TEXT, station_name TEXT, station_distance_km REAL,
 active INTEGER NOT NULL DEFAULT 1);
CREATE TABLE IF NOT EXISTS runs (
 id INTEGER PRIMARY KEY, location_id TEXT NOT NULL REFERENCES locations(id),
 provider TEXT NOT NULL, fetched_at TEXT NOT NULL, init_time TEXT, metadata TEXT NOT NULL,
 UNIQUE(location_id,provider,fetched_at));
CREATE TABLE IF NOT EXISTS points (
 run_id INTEGER NOT NULL REFERENCES runs(id), valid_time TEXT NOT NULL,
 temperature REAL, humidity REAL, precipitation REAL, wind_speed REAL, wind_direction REAL,
 wind_gust REAL, cloud_cover REAL, precip_probability REAL,
 temperature_p10 REAL, temperature_p90 REAL,
 PRIMARY KEY(run_id,valid_time));
CREATE INDEX IF NOT EXISTS runs_lookup ON runs(location_id,provider,fetched_at);
CREATE INDEX IF NOT EXISTS points_time ON points(valid_time);
CREATE TABLE IF NOT EXISTS observations (
 location_id TEXT NOT NULL, valid_time TEXT NOT NULL, station_id TEXT NOT NULL,
 fetched_at TEXT NOT NULL, temperature REAL, humidity REAL, precipitation REAL,
 wind_speed REAL, wind_direction REAL, PRIMARY KEY(location_id,valid_time,station_id));
CREATE TABLE IF NOT EXISTS state (
 key TEXT PRIMARY KEY, value TEXT NOT NULL, updated_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS budget (
 service TEXT NOT NULL, period TEXT NOT NULL, used INTEGER NOT NULL,
 PRIMARY KEY(service,period));
PRAGMA user_version=1;
"""
POINT_COLUMNS = ("temperature", "humidity", "precipitation", "wind_speed", "wind_direction",
                 "wind_gust", "cloud_cover", "precip_probability", "temperature_p10", "temperature_p90")


class Store:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or os.environ.get("CLIMATE_FORECAST_DB", DATA_DIR / "forecasts.sqlite3"))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            if version > 1:
                raise RuntimeError("Base de datos más reciente que esta versión de la app")
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self):
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def seed(self):
        # Inactive favorites stay inactive across restarts.
        if self.locations(include_inactive=True):
            return
        self.add_location("Los Narejos · Los Alcázares", 37.758, -0.837, "Europe/Madrid",
                          "08433", "San Javier")
        self.add_location("Torrejón de Ardoz", 40.4594, -3.4697, "Europe/Madrid",
                          "08227", "Madrid / Torrejón")

    def locations(self, include_inactive=False) -> list[Location]:
        with self.connect() as conn:
            rows = conn.execute("SELECT * FROM locations" + ("" if include_inactive else " WHERE active=1")
                                + " ORDER BY rowid").fetchall()
        return [Location(**{key: row[key] for key in Location.__dataclass_fields__}) for row in rows]

    def add_location(self, name, latitude, longitude, timezone="Europe/Madrid",
                     station_id=None, station_name=None, station_distance_km=None) -> str:
        latitude, longitude = finite(latitude), finite(longitude)
        if latitude is None or longitude is None or not (-90 <= latitude <= 90 and -180 <= longitude <= 180):
            raise ValueError("Coordenadas inválidas")
        ZoneInfo(timezone)
        if not name.strip() or len(name.strip()) > 120:
            raise ValueError("El nombre debe tener entre 1 y 120 caracteres")
        location_id = hashlib.sha256(f"{latitude:.4f},{longitude:.4f}".encode()).hexdigest()[:16]
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            existing = conn.execute("SELECT active FROM locations WHERE id=?", (location_id,)).fetchone()
            if (existing is None or not existing[0]) and conn.execute(
                    "SELECT count(*) FROM locations WHERE active=1").fetchone()[0] >= 10:
                raise ValueError("Máximo de 10 lugares activos para mantener las cuotas gratuitas")
            conn.execute("""INSERT INTO locations VALUES (?,?,?,?,?,?,?,?,1)
                ON CONFLICT(id) DO UPDATE SET active=1""",
                (location_id, name.strip(), latitude, longitude, timezone, station_id,
                 station_name, finite(station_distance_km)))
        return location_id

    def set_station(self, location_id, station_id, name, distance):
        with self.connect() as conn:
            conn.execute("UPDATE locations SET station_id=?,station_name=?,station_distance_km=? WHERE id=?",
                         (station_id, name, distance, location_id))

    def deactivate(self, location_id):
        with self.connect() as conn:
            conn.execute("UPDATE locations SET active=0 WHERE id=?", (location_id,))

    def save_forecast(self, location_id: str, forecast: Forecast):
        # Immutable snapshots: duplicates cannot rewrite an already collected run.
        points = [p for p in forecast.points if stamp(p["valid_time"]) > stamp(forecast.fetched_at)]
        if not points:
            raise ValueError("El proveedor no devolvió previsiones futuras")
        with self.connect() as conn:
            cursor = conn.execute("""INSERT OR IGNORE INTO runs
                (location_id,provider,fetched_at,init_time,metadata) VALUES (?,?,?,?,?)""",
                (location_id, forecast.provider, stamp(forecast.fetched_at),
                 stamp(forecast.init_time) if forecast.init_time else None,
                 json.dumps(forecast.metadata, ensure_ascii=False)))
            if not cursor.rowcount:
                return
            conn.executemany("INSERT INTO points VALUES (" + ",".join("?" for _ in range(12)) + ")",
                [(cursor.lastrowid, stamp(p["valid_time"]), *[finite(p.get(col)) for col in POINT_COLUMNS])
                 for p in points])

    def forecasts(self, location_id, since=None, latest=True) -> pd.DataFrame:
        since = stamp(since or utcnow() - timedelta(days=90))
        sql = """SELECT r.location_id,r.provider,r.fetched_at,r.init_time,r.metadata,p.*
                 FROM runs r JOIN points p ON p.run_id=r.id WHERE r.location_id=? AND p.valid_time>=?"""
        args = [location_id, since]
        if latest:
            # One complete snapshot per provider; never splice an old tail into a fresh run.
            sql += " AND r.id IN (SELECT MAX(id) FROM runs WHERE location_id=? GROUP BY provider)"
            args.append(location_id)
        with self.connect() as conn:
            df = pd.read_sql_query(sql + " ORDER BY p.valid_time", conn, params=args)
        for column in ("valid_time", "fetched_at", "init_time"):
            df[column] = pd.to_datetime(df[column], utc=True)
        return df

    def save_observations(self, location_id, station_id, points, fetched_at=None):
        fetched = stamp(fetched_at or utcnow())
        with self.connect() as conn:
            conn.executemany("""INSERT INTO observations VALUES (?,?,?,?,?,?,?,?,?)
                ON CONFLICT(location_id,valid_time,station_id) DO UPDATE SET
                fetched_at=excluded.fetched_at,temperature=excluded.temperature,humidity=excluded.humidity,
                precipitation=excluded.precipitation,wind_speed=excluded.wind_speed,wind_direction=excluded.wind_direction""",
                [(location_id, stamp(p["valid_time"]), station_id, fetched,
                  *[finite(p.get(col)) for col in ("temperature", "humidity", "precipitation", "wind_speed", "wind_direction")])
                 for p in points])

    def observations(self, location_id, since=None):
        with self.connect() as conn:
            df = pd.read_sql_query("""SELECT * FROM observations WHERE location_id=? AND valid_time>=?
                ORDER BY valid_time""", conn, params=(location_id, stamp(since or utcnow() - timedelta(days=90))))
        for column in ("valid_time", "fetched_at"):
            df[column] = pd.to_datetime(df[column], utc=True)
        return df

    def state(self, key, default=None):
        with self.connect() as conn:
            row = conn.execute("SELECT value FROM state WHERE key=?", (key,)).fetchone()
        return json.loads(row[0]) if row else default

    def set_state(self, key, value):
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO state VALUES (?,?,?)",
                         (key, json.dumps(value, ensure_ascii=False), stamp(utcnow())))

    def reserve(self, service: str, period: str, amount: int, limit: int) -> bool:
        with self.connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT used FROM budget WHERE service=? AND period=?", (service, period)).fetchone()
            used = row[0] if row else 0
            if amount < 0 or used + amount > limit:
                return False
            conn.execute("INSERT OR REPLACE INTO budget VALUES (?,?,?)", (service, period, used + amount))
        return True

    def backup(self, destination=None):
        target = Path(destination or self.path.with_name("forecasts-backup.sqlite3"))
        temp = target.with_suffix(".sqlite3.tmp")
        with self.connect() as source:
            dest = sqlite3.connect(temp)
            try:
                source.backup(dest)
                if dest.execute("PRAGMA integrity_check").fetchone()[0] != "ok":
                    raise RuntimeError("Snapshot SQLite inválido")
            finally:
                dest.close()
        temp.replace(target)
        return target

    def release(self, service, period, amount):
        with self.connect() as conn:
            conn.execute("UPDATE budget SET used=MAX(0,used-?) WHERE service=? AND period=?",
                         (max(0, int(amount)), service, period))
