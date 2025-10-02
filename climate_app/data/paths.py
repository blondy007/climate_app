from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_FILE = BASE_DIR / "meteostat_raw_oct-abr_torrejon08227_sanjavier08433.csv"
DATA_DIR = BASE_DIR / "data"
MASTER_FILE = DATA_DIR / "meteostat_master.csv"
THRESHOLDS_FILE = DATA_DIR / "wind_thresholds.json"


def ensure_data_dir() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
