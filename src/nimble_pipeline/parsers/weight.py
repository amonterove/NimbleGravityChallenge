"""Weight string parser. Converts diverse formats to kilograms.

Handles inputs like: '12.5 kg', '1.8kg', '250 grams', '370gr', '5.8 Kilograms',
'141 G.', '32.9 grs', '6.8 g', '309GR', '2.3 Kilos', '0.027 KG'.
"""

from __future__ import annotations

import re

# Order matters: longer tokens first so 'kilograms' matches before 'kg'.
_UNIT_TO_KG: list[tuple[str, float]] = [
    ("kilograms", 1.0),
    ("kilogram", 1.0),
    ("kilos", 1.0),
    ("kilo", 1.0),
    ("grams", 0.001),
    ("gram", 0.001),
    ("grs", 0.001),
    ("gr", 0.001),
    ("kg", 1.0),
    ("g", 0.001),
]

_NUMBER_RE = re.compile(r"[-+]?\d*[.,]?\d+")


def parse_weight_kg(raw: str | None) -> float | None:
    """Parse a weight string and return value in kilograms.

    Returns None when input is null/empty or unparseable.
    """
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text:
        return None

    match = _NUMBER_RE.search(text)
    if not match:
        return None

    number_str = match.group(0).replace(",", ".")
    try:
        value = float(number_str)
    except ValueError:
        return None

    remainder = text[match.end():].strip()
    remainder = re.sub(r"[^a-z]", "", remainder)

    if not remainder:
        return None

    multiplier: float | None = None
    for unit, factor in _UNIT_TO_KG:
        if remainder.startswith(unit):
            multiplier = factor
            break

    if multiplier is None:
        return None

    return round(value * multiplier, 6)
