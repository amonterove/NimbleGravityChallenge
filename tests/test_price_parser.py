"""Unit tests for the price parser. Cases are taken directly from the dataset."""

import pytest

from nimble_pipeline.parsers.price import parse_price_usd


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        # Plain dollar-prefixed
        ("$499.99", 499.99),
        ("$ 349.99", 349.99),
        ("$149.50", 149.50),
        ("$329.00", 329.00),
        ("$99.00", 99.00),
        ("$649.00", 649.00),
        ("$249.99", 249.99),
        ("$ 139.99", 139.99),
        ("$109.00", 109.00),
        ("$44.99", 44.99),
        ("$ 99.00", 99.00),
        ("$34.99", 34.99),
        ("$ 75.00", 75.00),
        ("$79.99", 79.99),
        ("$ 39.00", 39.00),
        # No prefix
        ("749", 749.00),
        ("179.0", 179.00),
        ("59.99", 59.99),
        ("1499.99", 1499.99),
        # USD / dollars suffix
        ("999.00 USD", 999.00),
        ("$ 249.00 dollars", 249.00),
        ("$ 49.99 dollars", 49.99),
        ("349,00 usd", 349.00),
        ("1,099.00 USD", 1099.00),
        # European decimal
        ("149,99", 149.99),
        ("79,99", 79.99),
        # OCR-style errors (space replaced decimal)
        ("59 99", 59.99),
        ("67 50", 67.50),
        ("$4 50.00", 450.00),
        ("$3 99.00", 399.00),
    ],
)
def test_parse_price_known_cases(raw: str, expected: float) -> None:
    actual = parse_price_usd(raw)
    assert actual is not None
    assert abs(actual - expected) < 1e-2, f"{raw!r} -> {actual} (expected {expected})"


@pytest.mark.parametrize("raw", [None, "", "   ", "abc", "USD"])
def test_parse_price_invalid_returns_none(raw: str | None) -> None:
    assert parse_price_usd(raw) is None
