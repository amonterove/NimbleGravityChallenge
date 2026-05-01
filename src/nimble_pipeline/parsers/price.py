"""Price string parser. Converts noisy price strings to a USD float.

Handles inputs like: '$499.99', '749', '$ 349.99', '999.00 USD',
'$ 249.00 dollars', '149,99' (european decimal), '349,00 usd',
'1,099.00 USD' (thousands comma), and OCR-like errors '59 99', '$4 50.00'.
"""

from __future__ import annotations

import re

_CURRENCY_TOKENS_RE = re.compile(r"\b(usd|dollars?|us)\b", re.IGNORECASE)
_NON_NUMERIC_RE = re.compile(r"[^\d.,\s-]")


def parse_price_usd(raw: str | None) -> float | None:
    """Parse a noisy price string and return value as USD float.

    Strategy:
    1. Strip currency symbols, USD/dollars suffixes, and any non-numeric chars
       (except dot, comma, space, minus).
    2. Decide between US format (1,099.99) and EU format (149,99) based on the
       relative position and count of '.' and ','.
    3. Repair OCR artefacts where a decimal separator was lost ('59 99' -> 59.99,
       '$4 50.00' -> 450.00).
    """
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None

    text = _CURRENCY_TOKENS_RE.sub("", text)
    text = _NON_NUMERIC_RE.sub("", text).strip()
    if not text:
        return None

    has_dot = "." in text
    has_comma = "," in text

    if has_dot and has_comma:
        # Whichever appears LAST is the decimal separator (US: 1,099.99 | EU: 1.099,99).
        if text.rfind(".") > text.rfind(","):
            text = text.replace(",", "")
        else:
            text = text.replace(".", "").replace(",", ".")
    elif has_comma:
        # Comma-only — treat as European decimal if it looks like one ('149,99'),
        # otherwise as a thousands separator ('1,099').
        comma_parts = text.split(",")
        if len(comma_parts) == 2 and 1 <= len(comma_parts[1].strip()) <= 2:
            text = text.replace(",", ".")
        else:
            text = text.replace(",", "")

    # At this point the only non-digit chars left should be '.' (decimal) and spaces.
    text = text.strip()
    if " " in text:
        text = _repair_internal_space(text)

    try:
        return round(float(text), 2)
    except ValueError:
        return None


def _repair_internal_space(text: str) -> str:
    """Repair OCR-like errors where a decimal point became a space.

    Examples:
        '59 99'      -> '59.99'   (no dot at all, space splits int+decimals)
        '$4 50.00'   -> '450.00'  (space splits an integer; '.00' already present)
        '$3 99.00'   -> '399.00'
    """
    parts = text.split()
    if not parts:
        return text
    if "." in text:
        # Decimal already present elsewhere; just glue the integer parts together.
        return "".join(parts)
    # No decimal: assume the LAST space is the decimal separator iff the trailing
    # chunk looks like 1-2 digits (cents).
    if len(parts[-1]) in (1, 2) and parts[-1].isdigit():
        return "".join(parts[:-1]) + "." + parts[-1]
    return "".join(parts)
