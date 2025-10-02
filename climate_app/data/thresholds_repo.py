import json
from typing import Dict

from climate_app.data.paths import THRESHOLDS_FILE, ensure_data_dir
from climate_app.shared.constants import DEFAULT_THRESHOLDS


def load_wind_thresholds() -> Dict[str, Dict[str, float]]:
    ensure_data_dir()
    if not THRESHOLDS_FILE.exists():
        save_wind_thresholds(DEFAULT_THRESHOLDS)
        return {city: values.copy() for city, values in DEFAULT_THRESHOLDS.items()}
    with THRESHOLDS_FILE.open("r", encoding="utf-8-sig") as fp:
        data = json.load(fp)
    parsed: Dict[str, Dict[str, float]] = {}
    for city, values in data.items():
        calma = float(values.get("calma", 10.0))
        ventoso = float(values.get("ventoso", max(calma, 25.0)))
        parsed[city] = {"calma": calma, "ventoso": ventoso}
    return parsed


def save_wind_thresholds(data: Dict[str, Dict[str, float]]) -> None:
    ensure_data_dir()
    with THRESHOLDS_FILE.open("w", encoding="utf-8") as fp:
        json.dump(data, fp, ensure_ascii=False, indent=2)
