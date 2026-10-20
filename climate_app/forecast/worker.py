from __future__ import annotations

import argparse
import logging
import os
import random
import signal
import threading
from datetime import timedelta

from climate_app.forecast.analysis import combine, scores
from climate_app.forecast.domain import Forecast, ProviderError, instant, stamp, utcnow
from climate_app.forecast.observations import fetch_observations, resolve_station
from climate_app.forecast.providers import OPEN_MODELS, fetch_metno, fetch_open_meteo
from climate_app.forecast.store import Store
from climate_app.forecast.weathernext import configured, fetch_weathernext

log = logging.getLogger(__name__)


def due(store, key, now):
    state = store.state("task:" + key, {})
    return not state.get("next_due") or instant(state["next_due"]) <= now


def run_task(store, key, action, interval):
    now = utcnow()
    store.set_state("collector", {"heartbeat": stamp(now), "stage": key})
    try:
        count = action()
        store.set_state("task:" + key, {"status": "ok", "updated_at": stamp(utcnow()),
                        "next_due": stamp(now + interval), "count": count})
        return True
    except Exception as exc:
        message = str(exc) if isinstance(exc, ProviderError) else f"Error temporal de recopilación ({type(exc).__name__})"
        store.set_state("task:" + key, {"status": "error", "updated_at": stamp(utcnow()),
                        "next_due": stamp(now + timedelta(minutes=45)), "message": message})
        log.warning("%s: %s", key, message)
        return False


def collect_cycle(store: Store, include_climate=True):
    store.seed()
    now = utcnow()
    changed = []
    # Fetch forecasts for every location before the larger climatology downloads.
    for location in store.locations():
        location_changed = False
        for provider in (*OPEN_MODELS, "metno", "weathernext3"):
            key = f"{location.id}:{provider}"
            if provider == "weathernext3" and not configured():
                store.set_state("task:" + key, {"status": "pending", "message": "Pendiente de acceso a Google"})
                continue
            if not due(store, key, now):
                continue

            def collect(provider=provider):
                if provider in OPEN_MODELS:
                    forecast = fetch_open_meteo(provider, location, store)
                elif provider == "metno":
                    forecast = fetch_metno(location, store)
                else:
                    forecast = fetch_weathernext(location, store)
                store.save_forecast(location.id, forecast)
                return len(forecast.points)

            ok = run_task(store, key, collect, timedelta(hours=1 if provider == "weathernext3" else 3))
            location_changed |= ok
        if location_changed:
            changed.append(location)
    for location in changed:
        as_of = utcnow()
        archive = store.forecasts(location.id, latest=False)
        observations = store.observations(location.id)
        training = scores(archive, observations, as_of=as_of)
        current = store.forecasts(location.id, since=as_of)
        variants = [("climateapp", current[current.provider != "weathernext3"])]
        if (current.provider == "weathernext3").any():
            variants.insert(0, ("climateapp_private", current))
        for consensus_provider, inputs in variants:
            merged = combine(inputs, training, as_of=as_of)
            if not merged.empty:
                weights = {}
                from climate_app.forecast.domain import horizon
                for row in merged.to_dict("records"):
                    bucket = horizon((row["valid_time"] - as_of).total_seconds() / 3600)
                    weights.setdefault(bucket, row["weights"])
                store.save_forecast(location.id, Forecast(consensus_provider, as_of, merged.to_dict("records"),
                    metadata={"policy": "inverse MAE + regularization; equal cold start; 30 samples/3 days",
                              "weights": weights, "trained": bool(merged.trained.any()),
                              "training_cutoff": stamp(as_of), "precipitation_interval": "preceding_hour"}))
    for location in store.locations():
        key = f"{location.id}:observations"
        if due(store, key, now):
            def observe():
                resolved = resolve_station(location, store)
                points = fetch_observations(resolved, now - timedelta(days=14), now - timedelta(hours=2))
                store.save_observations(resolved.id, resolved.station_id, points)
                return len(points)
            run_task(store, key, observe, timedelta(hours=6))
        key = f"{location.id}:climate"
        if include_climate and due(store, key, now):
            def baseline():
                resolved = resolve_station(location, store)
                points = fetch_observations(resolved, now - timedelta(days=5 * 366), now - timedelta(days=14))
                store.save_observations(resolved.id, resolved.station_id, points)
                return len(points)
            run_task(store, key, baseline, timedelta(days=30))
    store.backup()
    store.set_state("collector", {"heartbeat": stamp(utcnow()), "stage": "idle",
                    "last_cycle": stamp(utcnow())})


def run_forever(store=None, stop=None):
    store, stop = store or Store(), stop or threading.Event()
    lock = store.path.with_suffix(".lock").open("a+")
    try:
        if os.name == "nt":
            import msvcrt
            lock.seek(0)
            lock.write("0")
            lock.flush()
            lock.seek(0)
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while not stop.is_set():
            try:
                collect_cycle(store)
            except Exception as exc:
                log.error("Collector cycle failed (%s)", type(exc).__name__)
                store.set_state("collector", {"heartbeat": stamp(utcnow()), "stage": "error",
                                "message": "Error del recolector; se reintentará"})
            stop.wait(60 + random.randint(1, 30))
    finally:
        lock.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--no-climate", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.once:
        collect_cycle(Store(), include_climate=not args.no_climate)
    else:
        stop = threading.Event()
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda *_: stop.set())
        run_forever(stop=stop)


if __name__ == "__main__":
    main()
