from __future__ import annotations

import json
import os

from openai import OpenAI
from pydantic import BaseModel

from eq_generation.generators.base import BaseEQGenerator
from eq_generation.generators.prompts import format_prompt, get_system_prompt
from eq_generation.query_types import QueryType


_SAMPLING_UNSUPPORTED_MODEL_PREFIXES = (
    "gpt-5-mini-",
    "gpt-5-nano-",
    "o4-",
)

_MIN_REASONING_MODEL_TOKENS = 1024


class AnswerFormat(BaseModel):
    explanation: str
    answer: str


class AnswerOnlyFormat(BaseModel):
    answer: str


def _response_format_for(query_type: QueryType) -> type[BaseModel]:
    if query_type in {QueryType.QUESTION, QueryType.COMMAND, QueryType.INDIRECT}:
        return AnswerFormat
    return AnswerOnlyFormat


def _parse_model_content(content: str) -> dict[str, str]:
    text = content.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        return {"generated_query": text, "explanation": ""}
    if not isinstance(parsed, dict):
        return {"generated_query": text, "explanation": ""}
    answer = parsed.get("answer", parsed.get("generated_query", ""))
    return {
        "generated_query": str(answer).strip(),
        "explanation": str(parsed.get("explanation", "")).strip(),
    }


class GPTEQGenerator(BaseEQGenerator):
    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        batch_size: int = 10,
        max_tokens: int = 1024,
        temperature: float = 0.35,
        top_p: float = 0.9,
    ) -> None:
        super().__init__(batch_size, max_tokens, temperature, top_p)
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
        self.base_url = base_url or os.environ.get("OPENAI_BASE_URL") or os.environ.get("OPENROUTER_BASE_URL")
        if self.base_url is None and os.environ.get("OPENROUTER_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            self.base_url = "https://openrouter.ai/api/v1"
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            default_headers = {}
            site_url = os.environ.get("OPENROUTER_SITE_URL")
            app_name = os.environ.get("OPENROUTER_APP_NAME")
            if site_url:
                default_headers["HTTP-Referer"] = site_url
            if app_name:
                default_headers["X-Title"] = app_name
            self._client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
                default_headers=default_headers or None,
            )
        return self._client

    @property
    def supports_sampling_params(self) -> bool:
        return not self.model.startswith(_SAMPLING_UNSUPPORTED_MODEL_PREFIXES)

    @property
    def uses_reasoning_budget(self) -> bool:
        return self.model.startswith(_SAMPLING_UNSUPPORTED_MODEL_PREFIXES)

    @property
    def uses_openrouter(self) -> bool:
        return bool(self.base_url and "openrouter.ai" in self.base_url)

    def _generate_single(
        self,
        caption: str,
        query_type: QueryType,
        hard_negative_caption: str | None = None,
    ) -> dict[str, str]:
        del hard_negative_caption
        prompt = format_prompt(query_type, caption)
        max_completion_tokens = self.max_tokens
        if self.uses_reasoning_budget:
            max_completion_tokens = max(max_completion_tokens, _MIN_REASONING_MODEL_TOKENS)

        response_format = _response_format_for(query_type)
        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": get_system_prompt(query_type, backend="gpt")},
                {"role": "user", "content": prompt},
            ],
            "n": 1,
        }
        if self.uses_openrouter:
            request["max_tokens"] = max_completion_tokens
            if response_format is AnswerFormat:
                request["response_format"] = {"type": "json_object"}
        else:
            request["max_completion_tokens"] = max_completion_tokens
            request["response_format"] = response_format
        if self.supports_sampling_params:
            request["temperature"] = self.temperature
            request["top_p"] = self.top_p
        if self.uses_reasoning_budget:
            request["reasoning_effort"] = "low"

        if self.uses_openrouter:
            response = self.client.chat.completions.create(**request)
            message = response.choices[0].message
            parsed = _parse_model_content(message.content or "")
        else:
            response = self.client.beta.chat.completions.parse(**request)
            message = response.choices[0].message
            if message.parsed is not None:
                parsed = {
                    "generated_query": message.parsed.answer.strip(),
                    "explanation": getattr(message.parsed, "explanation", "").strip(),
                }
            else:
                parsed = _parse_model_content(message.content or "")
        if not parsed["generated_query"]:
            finish_reason = response.choices[0].finish_reason
            raise RuntimeError(f"OpenAI returned an empty query (finish_reason={finish_reason})")
        return parsed
