from typing import Dict, List

STATION_NAME_MAP: Dict[str, str] = {
    "torrejon_08227": "Torrejón de Ardoz",
    "08227": "Torrejón de Ardoz",
    "san_javier_08433": "Los Alcázares (San Javier)",
    "08433": "Los Alcázares (San Javier)",
}

STATION_ID_OVERRIDES: Dict[str, str] = {
    "torrejon_08227": "08227",
    "08227": "08227",
    "san_javier_08433": "08433",
    "08433": "08433",
}

DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "Torrejón de Ardoz": {"calma": 10.0, "ventoso": 25.0},
    "Los Alcázares (San Javier)": {"calma": 10.0, "ventoso": 35.0},
}

MONTH_NAMES: Dict[int, str] = {
    1: "Enero",
    2: "Febrero",
    3: "Marzo",
    4: "Abril",
    5: "Mayo",
    6: "Junio",
    7: "Julio",
    8: "Agosto",
    9: "Septiembre",
    10: "Octubre",
    11: "Noviembre",
    12: "Diciembre",
}

MONTH_NAME_TO_NUM: Dict[str, int] = {name: number for number, name in MONTH_NAMES.items()}

SCENARIO_CATEGORIES: List[str] = ["Calma", "Ventoso", "Viento fuerte", "Sin dato"]
SCENARIO_OPTIONS: List[str] = SCENARIO_CATEGORIES + ["Todos"]

COUNTRY_OPTIONS: List[str] = ["Todos", "ES", "US", "FR", "PT", "IT", "DE", "GB", "CA"]
