from typing import List
import os
import backoff
import instructor
from openai import AsyncOpenAI
from pydantic import BaseModel


def is_reasoning_model(model: str) -> bool:
    return model in [
        "o3-mini-2025-01-31",
        "o3-mini",
        "o1-preview-2024-09-12",
        "o1-preview",
        "o1-mini",
        "o1-mini-2024-09-12",
        "o1",
        "o1-2024-12-17",
    ]


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
async def run_llm_with_instructor(
    api_key: str,
    model: str,
    messages: List,
    response_model: BaseModel,
    max_completion_tokens: int,
):
    client = instructor.from_openai(AsyncOpenAI(api_key=api_key))

    model_kwargs = {}

    if not is_reasoning_model(model):
        model_kwargs["temperature"] = 0

    return await client.chat.completions.create(
        model=model,
        messages=messages,
        response_model=response_model,
        max_completion_tokens=max_completion_tokens,
        store=True,
        **model_kwargs,
    )


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
async def stream_llm_with_instructor(
    api_key: str,
    model: str,
    messages: List,
    response_model: BaseModel,
    max_completion_tokens: int,
    **kwargs,
):
    client = instructor.from_openai(AsyncOpenAI(api_key=api_key))

    model_kwargs = {}

    if not is_reasoning_model(model):
        model_kwargs["temperature"] = 0

    model_kwargs.update(kwargs)

    return client.chat.completions.create_partial(
        model=model,
        messages=messages,
        response_model=response_model,
        stream=True,
        max_completion_tokens=max_completion_tokens,
        store=True,
        **model_kwargs,
    )


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
async def stream_llm_responses_with_instructor(
    api_key: str,
    model: str,
    input: List,
    response_model: BaseModel,
    max_output_tokens: int,
    **kwargs,
):
    os.environ["OPENAI_API_KEY"] = api_key

    client = instructor.from_provider(
        model=f"openai/{model}",
        mode=instructor.Mode.RESPONSES_TOOLS,
        async_client=True,
    )

    model_kwargs = {}

    if not is_reasoning_model(model):
        model_kwargs["temperature"] = 0

    model_kwargs.update(kwargs)

    return client.responses.create_partial(
        input=input,
        response_model=response_model,
        stream=True,
        max_output_tokens=max_output_tokens,
        store=True,
        **model_kwargs,
    )


@backoff.on_exception(backoff.expo, Exception, max_tries=5, factor=2)
async def run_llm_responses_with_instructor(
    api_key: str,
    model: str,
    input: List,
    response_model: BaseModel,
    max_output_tokens: int,
):
    os.environ["OPENAI_API_KEY"] = api_key

    client = instructor.from_provider(
        model=f"openai/{model}",
        mode=instructor.Mode.RESPONSES_TOOLS,
        async_client=True,
    )

    model_kwargs = {}

    if not is_reasoning_model(model):
        model_kwargs["temperature"] = 0

    return await client.responses.create(
        input=input,
        response_model=response_model,
        max_output_tokens=max_output_tokens,
        store=True,
        **model_kwargs,
    )
