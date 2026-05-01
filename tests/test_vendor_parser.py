"""Unit tests for the vendor normalizer + canonical clustering."""

from nimble_pipeline.parsers.vendor import canonicalize_vendor, normalize_vendor_name


def test_normalize_strips_legal_suffixes() -> None:
    assert normalize_vendor_name("Samsung Electronics Co.") == "samsung electronics"
    assert normalize_vendor_name("SAMSUNG ELECTRONICS") == "samsung electronics"
    assert normalize_vendor_name("Apple Inc") == "apple"
    assert normalize_vendor_name("APPLE INC.") == "apple"
    assert normalize_vendor_name("Sony Corporation") == "sony"
    assert normalize_vendor_name("Sony Corp.") == "sony"
    assert normalize_vendor_name("sony corporation") == "sony"
    assert normalize_vendor_name("Logitech International S.A") == "logitech"
    assert normalize_vendor_name("logitech") == "logitech"
    # Dots are removed (not replaced with space) so 'Amazon.com' becomes 'amazoncom'
    # — this is intentional and lets WRatio still cluster it with plain 'amazon'.
    assert normalize_vendor_name("Amazon.com Inc") == "amazoncom"
    assert normalize_vendor_name("Amazon") == "amazon"


def test_normalize_handles_empty_or_none() -> None:
    assert normalize_vendor_name(None) == ""
    assert normalize_vendor_name("") == ""
    assert normalize_vendor_name("   ") == ""


def test_canonicalize_clusters_apple_variants() -> None:
    raw = ["Apple Inc", "APPLE INC.", "Samsung Electronics Co.", "SAMSUNG ELECTRONICS"]
    mapping = canonicalize_vendor(raw)

    assert mapping["Apple Inc"] == mapping["APPLE INC."]
    assert mapping["Samsung Electronics Co."] == mapping["SAMSUNG ELECTRONICS"]
    assert mapping["Apple Inc"] != mapping["Samsung Electronics Co."]


def test_canonicalize_clusters_asus_variants() -> None:
    raw = ["Asus tek computer inc.", "ASUSTeK Computer Inc.", "Asus Tek"]
    mapping = canonicalize_vendor(raw)

    canonical = {mapping[r] for r in raw}
    assert len(canonical) == 1, f"expected 1 canonical, got {canonical}"


def test_canonicalize_picks_longest_form_as_display() -> None:
    raw = ["Sony Corp.", "sony corporation", "Sony Corporation"]
    mapping = canonicalize_vendor(raw)
    canonical = {mapping[r] for r in raw}
    assert len(canonical) == 1
    # The longest raw form wins as display.
    assert next(iter(canonical)) in {"Sony Corporation", "sony corporation"}


def test_canonicalize_full_dataset_reduces_count() -> None:
    raw = [
        "Samsung Electronics Co.",
        "LENOVO GROUP LTD",
        "sony corporation",
        "Apple Inc",
        "JBL / Harman Intl.",
        "Wacom Co., Ltd.",
        "LG electronics",
        "Redragon  Inc.",
        "Logitech International S.A",
        "APPLE INC.",
        "HP Inc",
        "Sony Corp.",
        "Seagate Technology",
        "Asus tek computer inc.",
        "BenQ Corporation",
        "Amazon.com Inc",
        "Nintendo co ltd",
        "SAMSUNG ELECTRONICS",
        "Belkin International Inc.",
        "Hikvision Digital Tech Co.",
        "HyperX  (Kingston)",
        "ASUSTeK Computer Inc.",
        "UGREEN Group Limited",
        "Amazon",
        "Anker Innovations",
        "Asus Tek",
        "raspberry pi ltd",
        "DJI technology co. ltd",
        "logitech",
        "Xiaomi Corporation",
    ]
    mapping = canonicalize_vendor(raw)
    distinct_canonicals = {mapping[r] for r in raw}
    # 30 raw vendors should collapse to fewer canonical groups.
    assert len(distinct_canonicals) < len(raw)
    # Specifically, these pairs/triples MUST share a canonical:
    assert mapping["Apple Inc"] == mapping["APPLE INC."]
    assert mapping["Samsung Electronics Co."] == mapping["SAMSUNG ELECTRONICS"]
    assert mapping["sony corporation"] == mapping["Sony Corp."]
    assert mapping["Asus tek computer inc."] == mapping["ASUSTeK Computer Inc."] == mapping["Asus Tek"]
    assert mapping["Logitech International S.A"] == mapping["logitech"]
    assert mapping["Amazon.com Inc"] == mapping["Amazon"]
