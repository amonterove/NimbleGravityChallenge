"""Vendor name normalization and canonical-name resolution.

The dataset contains the same vendor written multiple ways:
    'Samsung Electronics Co.' vs 'SAMSUNG ELECTRONICS'
    'Apple Inc' vs 'APPLE INC.'
    'ASUSTeK Computer Inc.' vs 'Asus Tek' vs 'Asus tek computer inc.'

We split this in two layers:
1. ``normalize_vendor_name`` — deterministic cleanup (case, suffixes, punctuation).
2. ``canonicalize_vendor`` — fuzzy clustering across normalized names.
"""

from __future__ import annotations

import re
from collections.abc import Iterable

from rapidfuzz import fuzz

# Common legal suffixes / corporate qualifiers to drop during normalization.
_LEGAL_SUFFIXES = [
    "incorporated",
    "corporation",
    "international",
    "technologies",
    "technology",
    "company",
    "limited",
    "group",
    "tech",
    "corp",
    "intl",
    "inc",
    "ltd",
    "llc",
    "co",
    "sa",
    "ag",
]

# Words that often appear inside vendor names but shouldn't drive matching.
_NOISE_WORDS = {"the", "and", "of"}

_DOT_RE = re.compile(r"\.")
_PUNCT_RE = re.compile(r"[^\w\s]")
_MULTI_SPACE_RE = re.compile(r"\s+")


def normalize_vendor_name(raw: str | None) -> str:
    """Return a normalized form of a vendor name (lowercase, no suffixes/punct).

    Dots are removed (not replaced with space) so that 'S.A' -> 'sa' and 'Inc.' -> 'inc'
    keep their token shape; other punctuation becomes a space.
    """
    if raw is None:
        return ""
    text = str(raw).lower().strip()
    text = _DOT_RE.sub("", text)
    text = _PUNCT_RE.sub(" ", text)
    text = _MULTI_SPACE_RE.sub(" ", text).strip()
    if not text:
        return ""

    tokens = [t for t in text.split() if t and t not in _NOISE_WORDS]
    # Strip legal suffix tokens from the right repeatedly.
    while tokens and tokens[-1] in _LEGAL_SUFFIXES:
        tokens.pop()
    return " ".join(tokens)


def _compact(name: str) -> str:
    """Remove all whitespace — collapses 'asustek' vs 'asus tek' into one form."""
    return name.replace(" ", "")


def canonicalize_vendor(
    names: Iterable[str],
    threshold: int = 85,
) -> dict[str, str]:
    """Cluster a list of raw vendor names into canonical groups.

    Args:
        names: iterable of raw vendor strings.
        threshold: rapidfuzz WRatio score above which two normalized names are
            considered the same vendor. WRatio combines several scorers and is
            robust to substring/spacing differences ('asus tek' vs 'asustek').

    Returns:
        Mapping from each input name to its canonical name (the longest raw
        form within the cluster, which usually carries the most info).
    """
    raw_to_norm: dict[str, str] = {n: normalize_vendor_name(n) for n in names}

    canonical_for_norm: dict[str, str] = {}
    cluster_members: dict[str, list[str]] = {}

    for raw, norm in raw_to_norm.items():
        if not norm:
            canonical_for_norm[norm] = raw
            cluster_members.setdefault(norm, []).append(raw)
            continue

        # Match against both the spaced and the compacted form of each cluster head
        # so 'asus tek' clusters with 'asustek'.
        matched_key: str | None = None
        for key in canonical_for_norm:
            score = max(
                fuzz.WRatio(norm, key),
                fuzz.WRatio(_compact(norm), _compact(key)),
            )
            if score >= threshold:
                matched_key = key
                break

        if matched_key is None:
            canonical_for_norm[norm] = raw
            cluster_members[norm] = [raw]
        else:
            cluster_members[matched_key].append(raw)

    # Pick the longest raw name within each cluster as the display canonical.
    canonical_display: dict[str, str] = {
        key: max(members, key=len) for key, members in cluster_members.items()
    }

    result: dict[str, str] = {}
    for raw in raw_to_norm:
        # Find the cluster this raw landed in.
        for key, members in cluster_members.items():
            if raw in members:
                result[raw] = canonical_display[key]
                break
    return result
