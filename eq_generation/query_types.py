from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class QueryType(Enum):
    KEY_PHRASE = "key_phrase"
    STATEMENT = "statement"
    QUESTION = "question"
    COMMAND = "command"
    INDIRECT = "indirect"
    FULL_CAPTION = "full_caption"

    @classmethod
    def from_string(cls, value: str) -> "QueryType":
        for member in cls:
            if member.value == value.lower():
                return member
        raise ValueError(f"Unknown QueryType: {value}")


@dataclass
class QueryResult:
    audio_id: str
    dataset: str
    dataset_slug: str
    query_type: QueryType
    generated_query: str
    original_captions: list[str]
    explanation: str = ""
    metadata: Optional[dict] = field(default_factory=dict)
    source_model: str = "gpt-5.4-mini"
    regen_model: str = "gpt-5.4-mini"

    def to_dict(self) -> dict:
        return {
            "audio_id": self.audio_id,
            "dataset": self.dataset,
            "dataset_slug": self.dataset_slug,
            "query_type": self.query_type.value,
            "generated_query": self.generated_query,
            "original_captions": self.original_captions,
            "explanation": self.explanation,
            "metadata": self.metadata,
            "source_model": self.source_model,
            "regen_model": self.regen_model,
        }
