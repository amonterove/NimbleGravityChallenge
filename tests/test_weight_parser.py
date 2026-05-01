"""Unit tests for the weight parser. Cases are taken directly from the dataset."""

import pytest

from nimble_pipeline.parsers.weight import parse_weight_kg


@pytest.mark.parametrize(
    ("raw", "expected_kg"),
    [
        # Direct kg variants
        ("12.5 kg", 12.5),
        ("1.8kg", 1.8),
        ("0.206 KG", 0.206),
        ("0.96 Kg.", 0.96),
        ("8.2Kg", 8.2),
        ("0.32 KG", 0.32),
        ("0.43 Kg", 0.43),
        ("0.356 KG.", 0.356),
        ("0.027 KG", 0.027),
        ("0.215 kg", 0.215),
        ("2.6 kg.", 2.6),
        # Spelled-out kilograms / kilos
        ("5.8 Kilograms", 5.8),
        ("2.3 Kilos", 2.3),
        # Grams variants
        ("250 grams", 0.25),
        ("370gr", 0.370),
        ("900g", 0.9),
        ("141 G.", 0.141),
        ("32.9 grs", 0.0329),
        ("343 grams", 0.343),
        ("1100 G", 1.1),
        ("205 Grs.", 0.205),
        ("6.8 g", 0.0068),
        ("120 g.", 0.12),
        ("309GR", 0.309),
        ("98 grams", 0.098),
        ("304 g", 0.304),
        ("820 G.", 0.82),
        ("46 grs", 0.046),
        ("249 grams", 0.249),
        ("162g.", 0.162),
    ],
)
def test_parse_weight_known_cases(raw: str, expected_kg: float) -> None:
    actual = parse_weight_kg(raw)
    assert actual is not None
    assert abs(actual - expected_kg) < 1e-6, f"{raw!r} -> {actual} (expected {expected_kg})"


@pytest.mark.parametrize("raw", [None, "", "   ", "abc", "12", "kg", "12 unknown"])
def test_parse_weight_invalid_returns_none(raw: str | None) -> None:
    assert parse_weight_kg(raw) is None
