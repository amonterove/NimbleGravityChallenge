"""LLM-as-Judge task: validates extractor output against the FULL taxonomy."""

from __future__ import annotations

from dataclasses import dataclass

from nimble_pipeline.llm.client import LLMClient, LLMResponse, Prompt


@dataclass
class JudgeResult:
    """Output of a single judge call."""

    verdict: str  # "pass" | "fail"
    suggested_sub_category: str
    reason: str
    prompt_fingerprint: str
    model: str
    latency_ms: float
    raw_response: str
    parse_ok: bool


def judge_extraction(
    description: str,
    candidate_sub_category: str,
    client: LLMClient,
    prompt: Prompt,
) -> JudgeResult:
    """Validate an extractor's sub_category against the full approved taxonomy.

    Records that come back with verdict='fail' should be routed to a manual
    review table rather than into the Silver Taxonomy.
    """
    approved = prompt.metadata.get("approved_taxonomy", [])
    user = prompt.user_template.format(
        approved_taxonomy=", ".join(approved),
        description=description,
        candidate_sub_category=candidate_sub_category,
    )
    response: LLMResponse = client.chat_json(system=prompt.system, user=user)
    parsed = response.parsed
    parse_ok = "_parse_error" not in parsed

    verdict = str(parsed.get("verdict", "fail")).lower() if parse_ok else "fail"
    if verdict not in {"pass", "fail"}:
        verdict = "fail"

    return JudgeResult(
        verdict=verdict,
        suggested_sub_category=str(parsed.get("suggested_sub_category", "")) if parse_ok else "",
        reason=str(parsed.get("reason", "parse_error")) if parse_ok else "parse_error",
        prompt_fingerprint=prompt.fingerprint,
        model=response.model,
        latency_ms=response.latency_ms,
        raw_response=response.raw_text,
        parse_ok=parse_ok,
    )
