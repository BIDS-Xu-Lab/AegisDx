# libraries
from __future__ import annotations

import logging
import json_repair
from typing import Any
from langchain_community.adapters.openai import convert_openai_messages
from langchain_core.utils.json import parse_json_markdown
from loguru import logger


from llm_provider.generic.base import (
    NO_SUPPORT_TEMPERATURE_MODELS,
    SUPPORT_REASONING_EFFORT_MODELS,
    ReasoningEfforts,
)

import os


async def call_model(
    prompt: list,
    model: str,
    response_format: str | None = None,
    temperature: float | None = 0,
    llm_provider: str | None = "openai",
):

    lc_messages = convert_openai_messages(prompt)

    try:
        response = await create_chat_completion(
            model=model,
            messages=lc_messages,
            temperature=temperature,
            llm_provider=llm_provider,
        )

        if response_format == "json":
            return parse_json_markdown(response, parser=json_repair.loads)

        return response

    except Exception as e:
        print("⚠️ Error in calling model")
        logger.error(f"Error in calling model: {e}")


def get_llm(llm_provider, **kwargs):
    from llm_provider import GenericLLMProvider

    return GenericLLMProvider.from_provider(llm_provider, **kwargs)


async def create_chat_completion(
    messages: list[dict[str, str]],
    model: str | None = None,
    temperature: float | None = 0.4,
    max_tokens: int | None = None,
    llm_provider: str | None = None,
    stream: bool = False,
    websocket: Any | None = None,
    llm_kwargs: dict[str, Any] | None = None,
    reasoning_effort: str | None = ReasoningEfforts.Medium.value,
    **kwargs,
) -> str:
    """Create a chat completion using the OpenAI API
    Args:
        messages (list[dict[str, str]]): The messages to send to the chat completion.
        model (str, optional): The model to use. Defaults to None.
        temperature (float, optional): The temperature to use. Defaults to 0.4.
        max_tokens (int, optional): The max tokens to use. Defaults to 4000.
        llm_provider (str, optional): The LLM Provider to use.
        stream (bool): Whether to stream the response. Defaults to False.
        webocket (WebSocket): The websocket used in the currect request,
        llm_kwargs (dict[str, Any], optional): Additional LLM keyword arguments. Defaults to None.
        reasoning_effort (str, optional): Reasoning effort for OpenAI's reasoning models. Defaults to 'low'.
        **kwargs: Additional keyword arguments.
    Returns:
        str: The response from the chat completion.
    """
    # validate input
    if model is None:
        raise ValueError("Model cannot be None")
    if max_tokens is not None and max_tokens > 32001:
        raise ValueError(f"Max tokens cannot be more than 32,000, but got {max_tokens}")

    # Get the provider from supported providers
    provider_kwargs = {"model": model}

    if llm_kwargs:
        provider_kwargs.update(llm_kwargs)

    if model in SUPPORT_REASONING_EFFORT_MODELS:
        provider_kwargs["reasoning_effort"] = reasoning_effort

    if model not in NO_SUPPORT_TEMPERATURE_MODELS:
        provider_kwargs["temperature"] = temperature
        provider_kwargs["max_tokens"] = max_tokens
    else:
        provider_kwargs["temperature"] = None
        provider_kwargs["max_tokens"] = None

    if llm_provider == "openai":
        base_url = os.environ.get("OPENAI_BASE_URL", None)
        if base_url:
            provider_kwargs["openai_api_base"] = base_url
    
    provider = get_llm(llm_provider, **provider_kwargs)
    response = ""
    # create response
    for _ in range(10):  # maximum of 10 attempts
        response = await provider.get_chat_response(
            messages, stream, websocket, **kwargs
        )

        return response

    logging.error(f"Failed to get response from {llm_provider} API")
    raise RuntimeError(f"Failed to get response from {llm_provider} API")
