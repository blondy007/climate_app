"""Bounded, anonymous OPERA downloads with immutable regional observations."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import io
import json
import math
import os
from pathlib import Path
import re
import time
import xml.etree.ElementTree as ET

import h5py
import numpy as np
from pyproj import Transformer
import requests

UTC = timezone.utc
HOST = "https://s3.waw3-1.cloudferro.com"
BUCKETS = ("openradar-24h", "openradar-archive")
KEY = re.compile(r"^(\d{4})/(\d{2})/(\d{2})/OPERA/COMP/OPERA@(\d{8}T\d{4})@0@(DBZH|RATE|ACRR)\.h5$")
USER_AGENT = "ClimateApp nowcast evaluation (https://github.com/blondy007/climate_app)"


@dataclass(frozen=True)
class Site:
    id: str
    latitude: float
    longitude: float


# Public town centres; this experiment never imports the app's location database.
SITES = (Site("los-alcazares", 37.744, -.850), Site("torrejon", 40.456, -3.475))


def utcnow():
    return datetime.now(UTC)


def stamp(when):
    return when.astimezone(UTC).strftime("%Y%m%dT%H%M")


def parse_stamp(value):
    return datetime.strptime(value, "%Y%m%dT%H%M").replace(tzinfo=UTC)


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)


def request_bytes(url, params=None, maximum=16_000_000):
    if not url.startswith(HOST + "/"):
        raise ValueError("Only the official public radar host is allowed")
    error = None
    for attempt in range(3):
        try:
            with requests.get(url, params=params, headers={"User-Agent": USER_AGENT},
                              timeout=(10, 45), stream=True, allow_redirects=False) as response:
                response.raise_for_status()
                if response.is_redirect:
                    raise ValueError("Unexpected download redirect")
                if int(response.headers.get("Content-Length", 0)) > maximum:
                    raise ValueError("Radar response exceeds size cap")
                body = bytearray()
                for chunk in response.iter_content(65536):
                    body.extend(chunk)
                    if len(body) > maximum:
                        raise ValueError("Radar response exceeds size cap")
                return bytes(body)
        except requests.RequestException as exc:
            error = exc
            if attempt < 2:
                time.sleep(attempt + 1)
    raise RuntimeError("Public radar download unavailable") from error


def list_day(day, bucket="openradar-archive"):
    if bucket not in BUCKETS:
        raise ValueError("Unknown public bucket")
    prefix = day.strftime("%Y/%m/%d/OPERA/COMP/")
    body = request_bytes(HOST + "/" + bucket + "/", {"list-type": "2", "prefix": prefix, "max-keys": "1000"}, 1_000_000)
    root = ET.fromstring(body)
    ns = {"s": "http://s3.amazonaws.com/doc/2006-03-01/"}
    if root.findtext("s:IsTruncated", namespaces=ns) != "false":
        raise ValueError("Unexpectedly truncated radar day catalog")
    result = []
    for item in root.findall("s:Contents", ns):
        key = item.findtext("s:Key", namespaces=ns)
        match = KEY.fullmatch(key or "")
        if match and key.startswith(prefix):
            result.append({"key": key, "bucket": bucket, "product": match[5], "stamp": match[4],
                           "bytes": int(item.findtext("s:Size", namespaces=ns)),
                           "last_modified": item.findtext("s:LastModified", namespaces=ns)})
    return sorted(result, key=lambda item: (item["stamp"], item["product"]))


def decode(value):
    return value.decode("ascii") if isinstance(value, bytes) else str(value)


def decode_frame(body, product, when, radius_km=128):
    """Keep physical values, missing coverage and undetect distinct in each crop."""
    arrays = {}
    metadata = {"product": product, "nominal_end": when.isoformat(), "sites": {},
                "sha256_source": hashlib.sha256(body).hexdigest(), "download_bytes": len(body),
                "license": "CC BY 4.0", "source": "EUMETNET OPERA"}
    with h5py.File(io.BytesIO(body), "r") as f:
        where = dict(f["where"].attrs)
        what = dict(f["what"].attrs)
        period = dict(f["dataset1/what"].attrs)
        field = dict(f["dataset1/data1/what"].attrs)
        if decode(what["object"]) != "COMP" or decode(field["quantity"]) != product:
            raise ValueError("Radar product does not match its catalog")
        def hdf_time(date, clock):
            return datetime.strptime(decode(date) + decode(clock), "%Y%m%d%H%M%S").replace(tzinfo=UTC)
        if hdf_time(what["date"], what["time"]) != when:
            raise ValueError("Internal radar timestamp differs from filename")
        begin, end = (hdf_time(period["startdate"], period["starttime"]), hdf_time(period["enddate"], period["endtime"]))
        if end != when or begin > end or end - begin > timedelta(hours=2):
            raise ValueError("Invalid internal radar observation interval")
        metadata.update(observation_start=begin.isoformat(), observation_end=end.isoformat(),
                        grid_km=float(where["xscale"]) / 1000,
                        projection=decode(where["projdef"]))
        transform = Transformer.from_crs("EPSG:4326", decode(where["projdef"]), always_xy=True)
        origin_x, origin_y = transform.transform(where["LL_lon"], where["LL_lat"])
        metadata["global_origin_xy"] = [origin_x, origin_y]
        metadata["global_shape"] = [int(where["ysize"]), int(where["xsize"])]
        dataset = f["dataset1/data1/data"]
        half = math.ceil(radius_km / metadata["grid_km"])
        for site in SITES:
            x, y = transform.transform(site.longitude, site.latitude)
            col = math.floor((x - origin_x) / where["xscale"])
            row = int(where["ysize"]) - 1 - math.floor((y - origin_y) / where["yscale"])
            if not (half <= row < dataset.shape[0] - half and half <= col < dataset.shape[1] - half):
                raise ValueError("Research window outside radar grid")
            raw = dataset[row-half:row+half, col-half:col+half]
            missing, undetect = raw == field["nodata"], raw == field["undetect"]
            data = np.asarray(raw * field["gain"] + field["offset"], dtype=np.float32)
            data[missing] = np.nan
            data[undetect] = -15.0 if product == "DBZH" else 0.0
            arrays[site.id] = data
            arrays[site.id + "_undetect"] = undetect
            near = data[half-2:half+3, half-2:half+3]
            metadata["sites"][site.id] = {"centre": [half, half], "shape": list(data.shape),
                "crop_origin_rc": [row-half, col-half],
                "coverage_fraction": float(np.isfinite(data).mean()),
                "nearby_coverage": float(np.isfinite(near).mean()),
                "nearby_max": float(np.nanmax(near)) if np.isfinite(near).any() else None}
    return arrays, metadata


def frame_path(root, product, value):
    return Path(root) / "frames" / product / (value + ".npz")


def collect_frame(root, item, max_storage_mb=750):
    root = Path(root)
    product, value = item["product"], item["stamp"]
    path = frame_path(root, product, value)
    metadata_path = path.with_suffix(".json")
    if path.exists() and metadata_path.exists():
        return json.loads(metadata_path.read_text(encoding="utf-8"))
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file()) if root.exists() else 0
    if used > max_storage_mb * 1024 ** 2:
        raise RuntimeError("Research storage cap reached; existing data retained")
    if not KEY.fullmatch(item["key"]) or item["bucket"] not in BUCKETS or item["bytes"] > 16_000_000:
        raise ValueError("Invalid radar download reference")
    body = request_bytes(HOST + "/" + item["bucket"] + "/" + item["key"])
    fetched = utcnow()
    arrays, metadata = decode_frame(body, product, parse_stamp(value))
    metadata.update(retrieved_at=fetched.isoformat(), catalog_last_modified=item["last_modified"],
                    url=HOST + "/" + item["bucket"] + "/" + item["key"])
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".npz.tmp")
    with temporary.open("wb") as out:
        np.savez_compressed(out, **arrays)
    os.replace(temporary, path)
    metadata["sha256_crop"] = hashlib.sha256(path.read_bytes()).hexdigest()
    atomic_json(metadata_path, metadata)
    return metadata


def load_frame(root, product, value):
    path = frame_path(root, product, value)
    metadata = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    with np.load(path, allow_pickle=False) as saved:
        data = {site.id: saved[site.id] for site in SITES}
    return data, metadata
