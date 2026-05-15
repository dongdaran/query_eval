from __future__ import annotations

import os

from openai import OpenAI

from eq_generation.generators.base import BaseEQGenerator
from eq_generation.generators.prompts import format_prompt, get_system_prompt
from eq_generation.query_types import QueryType


_SAMPLING_UNSUPPORTED_MODEL_PREFIXES = (
    "gpt-5-mini-",
    "gpt-5-nano-",
    "o4-",
)

_MIN_REASONING_MODEL_TOKENS = 1024


class GPTEQGenerator(BaseEQGenerator):
    def __init__(
        self,
        model: str = "gpt-5.4-mini",
        api_key: str | None = None,
        batch_size: int = 10,
        max_tokens: int = 256,
        temperature: float = 0.35,
        top_p: float = 0.9,
    ) -> None:
        super().__init__(batch_size, max_tokens, temperature, top_p)
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client: OpenAI | None = None

    @property
    def client(self) -> OpenAI:
        if self._client is None:
            self._client = OpenAI(api_key=self.api_key)
        return self._client

    @property
    def supports_sampling_params(self) -> bool:
        return not self.model.startswith(_SAMPLING_UNSUPPORTED_MODEL_PREFIXES)

    @property
    def uses_reasoning_budget(self) -> bool:
        return self.model.startswith(_SAMPLING_UNSUPPORTED_MODEL_PREFIXES)

    def _generate_single(
        self,
        caption: str,
        query_type: QueryType,
        hard_negative_caption: str | None = None,
    ) -> str:
        del hard_negative_caption
        prompt = format_prompt(query_type, caption)
        max_completion_tokens = self.max_tokens
        if self.uses_reasoning_budget:
            max_completion_tokens = max(max_completion_tokens, _MIN_REASONING_MODEL_TOKENS)

        request = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": get_system_prompt(query_type, backend="gpt")},
                {"role": "user", "content": prompt},
            ],
            "max_completion_tokens": max_completion_tokens,
            "n": 1,
        }
        if self.supports_sampling_params:
            request["temperature"] = self.temperature
            request["top_p"] = self.top_p
        if self.uses_reasoning_budget:
            request["reasoning_effort"] = "low"

        response = self.client.chat.completions.create(**request)
        content = response.choices[0].message.content or ""
        query = content.strip()
        if not query:
            finish_reason = response.choices[0].finish_reason
            raise RuntimeError(f"OpenAI returned an empty query (finish_reason={finish_reason})")
        return query
