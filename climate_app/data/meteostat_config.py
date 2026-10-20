"""Keep Meteostat's disposable cache inside the application's writable volume."""
from meteostat import Hourly, Stations

from climate_app.data.paths import DATA_DIR

Hourly.cache_dir = str(DATA_DIR / "meteostat-cache")
Stations.cache_dir = str(DATA_DIR / "meteostat-cache")
