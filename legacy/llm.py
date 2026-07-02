"""Thin wrapper around LangChain chat providers.

Exposes a single `call_model` coroutine the agents use to talk to an LLM.
Multi-provider support is delegated to LangChain; we only translate from
OpenAI-style message dicts and optionally parse JSON responses with
`json-repair` for robustness against unquoted keys / trailing commas.
"""
from __future__ import annotations

import importlib
import os
from typing import Any

import json_repair
from langchain_community.adapters.openai import convert_openai_messages
from langchain_core.utils.json import parse_json_markdown
from loguru import logger


# OpenAI reasoning models that reject custom temperature / max_tokens.
_NO_TEMPERATURE_MODELS = {
    "o1", "o1-mini", "o1-preview",
    "o3", "o3-mini",
    "o4-mini",
    "gpt-5", "gpt-5-mini",
}

_REASONING_EFFORT_MODELS = {"o3-mini", "o3", "o4-mini"}


def _check_pkg(pkg: str) -> None:
    if importlib.util.find_spec(pkg) is None:
        raise ImportError(
            f"Provider requires `{pkg.replace('_', '-')}`. "
            f"Install it (e.g. `uv add {pkg.replace('_', '-')}`) and retry."
        )


def _get_llm(provider: str, **kwargs: Any):
    if provider == "openai":
        _check_pkg("langchain_openai")
        from langchain_openai import ChatOpenAI

        base_url = os.environ.get("OPENAI_BASE_URL")
        if base_url:
            kwargs["openai_api_base"] = base_url
        return ChatOpenAI(**kwargs)

    if provider == "anthropic":
        _check_pkg("langchain_anthropic")
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(**kwargs)

    if provider == "ollama":
        _check_pkg("langchain_ollama")
        from langchain_ollama import ChatOllama

        return ChatOllama(base_url=os.environ["OLLAMA_BASE_URL"], **kwargs)

    if provider == "openrouter":
        _check_pkg("langchain_openai")
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(
            openai_api_base="https://openrouter.ai/api/v1",
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            **kwargs,
        )

    raise ValueError(f"Unsupported LLM provider: {provider!r}")


async def call_model(
    messages: list[dict[str, str]],
    model: str,
    *,
    temperature: float | None = 0.0,
    response_format: str | None = None,
    llm_provider: str = "openai",
    max_tokens: int | None = None,
) -> Any:
    """Call an LLM and optionally parse a JSON response.

    Returns the raw string content, or — when `response_format='json'` — the
    parsed JSON object (dict / list).
    """
    provider_kwargs: dict[str, Any] = {"model": model}
    if model in _NO_TEMPERATURE_MODELS:
        provider_kwargs["temperature"] = None
        provider_kwargs["max_tokens"] = None
    else:
        provider_kwargs["temperature"] = temperature
        provider_kwargs["max_tokens"] = max_tokens
    if model in _REASONING_EFFORT_MODELS:
        provider_kwargs["reasoning_effort"] = "medium"

    llm = _get_llm(llm_provider, **provider_kwargs)
    lc_messages = convert_openai_messages(messages)

    try:
        result = await llm.ainvoke(lc_messages)
    except Exception as e:  # pragma: no cover — pass through to caller
        logger.error(f"LLM call failed ({llm_provider}/{model}): {e}")
        raise

    response = result.content
    if response_format == "json":
        return parse_json_markdown(response, parser=json_repair.loads)
    return response
