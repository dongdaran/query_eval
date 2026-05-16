from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional, Sequence

from tqdm import tqdm

from eq_generation.query_types import QueryResult, QueryType


class BaseEQGenerator(ABC):
    def __init__(
        self,
        batch_size: int = 10,
        max_tokens: int = 1024,
        temperature: float = 0.35,
        top_p: float = 0.9,
    ) -> None:
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p

    @abstractmethod
    def _generate_single(
        self,
        caption: str,
        query_type: QueryType,
        hard_negative_caption: Optional[str] = None,
    ) -> Any:
        raise NotImplementedError

    @staticmethod
    def _normalize_generation(raw_output: Any) -> tuple[str, str]:
        if isinstance(raw_output, dict):
            generated_query = str(raw_output.get("generated_query", "")).strip()
            explanation = str(raw_output.get("explanation", "")).strip()
            return generated_query, explanation
        return str(raw_output).strip(), ""

    def generate(
        self,
        captions: Sequence[str],
        query_type: QueryType,
        clip_ids: Optional[Sequence[str]] = None,
        show_progress: bool = True,
    ) -> list[QueryResult]:
        if clip_ids is None:
            clip_ids = [f"clip_{index}" for index in range(len(captions))]

        results = []
        iterator = range(len(captions))
        if show_progress:
            iterator = tqdm(iterator, desc=f"Generating {query_type.value}", unit="query")

        for index in iterator:
            try:
                raw_output = self._generate_single(
                    caption=captions[index],
                    query_type=query_type,
                )
                query, explanation = self._normalize_generation(raw_output)
                results.append(
                    QueryResult(
                        audio_id=clip_ids[index],
                        dataset="unknown",
                        dataset_slug="unknown",
                        original_captions=[captions[index]],
                        query_type=query_type,
                        generated_query=query.strip(),
                        explanation=explanation,
                    )
                )
            except Exception as exc:
                print(f"[WARN] Failed to generate query for clip {clip_ids[index]}: {exc}")
                results.append(
                    QueryResult(
                        audio_id=clip_ids[index],
                        dataset="unknown",
                        dataset_slug="unknown",
                        original_captions=[captions[index]],
                        query_type=query_type,
                        generated_query="",
                        metadata={"error": str(exc)},
                    )
                )

        return results

    def save_results(
        self,
        results: list[QueryResult],
        output_path: Path,
        format: str = "jsonl",
    ) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if format == "jsonl":
            with output_path.open("w", encoding="utf-8") as handle:
                for result in results:
                    handle.write(json.dumps(result.to_dict()) + "\n")
        elif format == "json":
            with output_path.open("w", encoding="utf-8") as handle:
                json.dump([result.to_dict() for result in results], handle, indent=2)
        else:
            raise ValueError(f"Unknown format: {format}")

        print(f"[INFO] Saved {len(results)} results to {output_path}")
