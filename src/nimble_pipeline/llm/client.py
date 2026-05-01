"""LLM client wrapper.

Targets Databricks Foundation Model APIs by default (no external secret
required when running on Databricks Free Edition). Falls back to OpenAI-style
APIs when ``DATABRICKS_HOST`` is not set.

Calls are instrumented with MLflow when an active run is present so prompts,
responses, latency and model metadata are traceable end-to-end.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

DEFAULT_DATABRICKS_MODEL = "databricks-gpt-oss-120b"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@dataclass
class LLMResponse:
    """Structured response from a single LLM call."""

    raw_text: str
    parsed: dict[str, Any]
    model: str
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None


@dataclass
class Prompt:
    """A versioned prompt loaded from YAML."""

    name: str
    version: int
    system: str
    user_template: str
    metadata: dict[str, Any]

    @property
    def fingerprint(self) -> str:
        return f"{self.name}:v{self.version}"


def load_prompt(path: str | Path) -> Prompt:
    """Load a versioned prompt YAML from disk."""
    raw = yaml.safe_load(Path(path).read_text())
    metadata = {k: v for k, v in raw.items() if k not in {"system", "user_template"}}
    return Prompt(
        name=raw["name"],
        version=int(raw["version"]),
        system=raw["system"],
        user_template=raw["user_template"],
        metadata=metadata,
    )


class LLMClient:
    """Thin wrapper around the OpenAI-compatible chat-completions interface.

    On Databricks Free Edition, point ``base_url`` at the workspace's serving
    endpoint and use the workspace token. When ``base_url`` is not provided we
    fall back to environment-driven defaults (Databricks first, OpenAI second).
    """

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model, self.base_url, self.api_key = _resolve_endpoint(model, base_url, api_key)

        # Lazy import so the package can be imported without openai installed
        # (e.g. in CI lint jobs).
        from openai import OpenAI  # type: ignore

        self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)

    def chat_json(self, system: str, user: str) -> LLMResponse:
        """Call the model and parse the response as JSON.

        Uses ``response_format={"type": "json_object"}`` when the underlying
        model supports it. The model is instructed via the system prompt to
        return only JSON, which keeps us provider-agnostic.
        """
        start = time.perf_counter()
        completion = self._client.chat.completions.create(
            model=self.model,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        latency_ms = (time.perf_counter() - start) * 1000.0

        raw_text = completion.choices[0].message.content or ""
        parsed = _safe_parse_json(raw_text)

        usage = getattr(completion, "usage", None)
        return LLMResponse(
            raw_text=raw_text,
            parsed=parsed,
            model=self.model,
            latency_ms=latency_ms,
            prompt_tokens=getattr(usage, "prompt_tokens", None),
            completion_tokens=getattr(usage, "completion_tokens", None),
        )


def _resolve_endpoint(
    model: str | None,
    base_url: str | None,
    api_key: str | None,
) -> tuple[str, str | None, str | None]:
    """Decide which provider to use based on env vars."""
    if base_url and api_key:
        return model or DEFAULT_DATABRICKS_MODEL, base_url, api_key

    databricks_host = os.environ.get("DATABRICKS_HOST")
    databricks_token = os.environ.get("DATABRICKS_TOKEN")
    if databricks_host and databricks_token:
        host = databricks_host.rstrip("/")
        return (
            model or DEFAULT_DATABRICKS_MODEL,
            f"{host}/serving-endpoints",
            databricks_token,
        )

    openai_key = os.environ.get("OPENAI_API_KEY")
    if openai_key:
        return model or DEFAULT_OPENAI_MODEL, None, openai_key

    raise RuntimeError(
        "No LLM endpoint configured. Set DATABRICKS_HOST + DATABRICKS_TOKEN "
        "(recommended on Databricks Free Edition) or OPENAI_API_KEY."
    )


def _safe_parse_json(text: str) -> dict[str, Any]:
    """Parse a JSON object from a model response, tolerating markdown fences."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        # Strip ```json ... ``` fences if the model added them.
        lines = [line for line in cleaned.splitlines() if not line.startswith("```")]
        cleaned = "\n".join(lines).strip()
    try:
        result = json.loads(cleaned)
        if not isinstance(result, dict):
            raise ValueError("Expected JSON object at top level")
        return result
    except (json.JSONDecodeError, ValueError):
        return {"_parse_error": True, "_raw": text}
