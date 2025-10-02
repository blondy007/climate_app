from dataclasses import dataclass
from typing import List, Optional, Tuple, Any


@dataclass
class FilterSelections:
    cities: List[str]
    agg_option: str
    raw_date_value: Any
    months: List[str]
    hours: List[str]
    scenarios: List[str]
    show_distribution: bool
