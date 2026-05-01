"""Extractor task: pulls Name, Sub-Category, Brand from a product description."""

from __future__ import annotations

from dataclasses import dataclass

from nimble_pipeline.llm.client import LLMClient, LLMResponse, Prompt


@dataclass
class ExtractionResult:
    """Output of a single extraction call, carrying observability metadata."""

    name: str
    sub_category: str
    brand: str
    prompt_fingerprint: str
    model: str
    latency_ms: float
    raw_response: str
    parse_ok: bool


def extract_product_attributes(
    description: str,
    client: LLMClient,
    prompt: Prompt,
) -> ExtractionResult:
    """Extract structured attributes from a product description.

    The prompt is responsible for constraining sub_category to a RESTRICTED list
    (intentionally narrower than the Judge's full taxonomy).
    """
    restricted = prompt.metadata.get("restricted_sub_categories", [])
    user = prompt.user_template.format(
        restricted_sub_categories=", ".join(restricted),
        description=description,
    )
    response: LLMResponse = client.chat_json(system=prompt.system, user=user)
    parsed = response.parsed
    parse_ok = "_parse_error" not in parsed

    return ExtractionResult(
        name=str(parsed.get("name", "")) if parse_ok else "",
        sub_category=str(parsed.get("sub_category", "")) if parse_ok else "",
        brand=str(parsed.get("brand", "Unknown")) if parse_ok else "Unknown",
        prompt_fingerprint=prompt.fingerprint,
        model=response.model,
        latency_ms=response.latency_ms,
        raw_response=response.raw_text,
        parse_ok=parse_ok,
    )
