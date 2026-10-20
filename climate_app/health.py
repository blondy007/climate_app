from datetime import timedelta
from urllib.request import urlopen

from climate_app.forecast.domain import instant, utcnow
from climate_app.forecast.store import Store


def main():
    with urlopen("http://127.0.0.1:8501/_stcore/health", timeout=3) as response:
        assert response.status == 200
    state = Store().state("collector", {})
    assert state.get("heartbeat"), "Collector has not started"
    assert utcnow() - instant(state["heartbeat"]) < timedelta(minutes=15), "Collector heartbeat is stale"


if __name__ == "__main__":
    main()
