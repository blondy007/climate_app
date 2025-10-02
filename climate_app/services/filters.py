from datetime import date
from typing import Optional, Tuple


def parse_date_range(value) -> Optional[Tuple[date, date]]:
    start: Optional[date] = None
    end: Optional[date] = None
    if isinstance(value, tuple) and len(value) == 2:
        start, end = value
    elif isinstance(value, list) and len(value) == 2:
        start, end = value[0], value[1]
    elif value:
        start = value
    if start is None:
        return None
    if end is None:
        end = start
    if start > end:
        end = start
    return start, end

def resolve_active_date_range(raw_value, session_state) -> Optional[Tuple[date, date]]:
    parsed = parse_date_range(raw_value)
    if parsed is None:
        session_state.pop("active_date_range", None)
        return None
    previous = session_state.get("active_date_range")
    if previous != parsed:
        session_state["active_date_range"] = parsed
    return session_state.get("active_date_range")
