from __future__ import annotations

import json
import random

from eq_generation.query_types import QueryType

START_WORD_COUNT_RANGE = (3, 15)
START_WORD_VARIATION_TYPES = {
    QueryType.STATEMENT,
    QueryType.QUESTION,
    QueryType.COMMAND,
    QueryType.INDIRECT,
}

FULL_CAPTION_PROMPT_TEMPLATE = """Caption Set:
{caption}
Full_caption:"""

SINGLE_REFERENCE_SUFFIX = {
    QueryType.KEY_PHRASE: "Query:",
    QueryType.STATEMENT: "Statement:",
    QueryType.QUESTION: "Question:",
    QueryType.COMMAND: "Command:",
    QueryType.INDIRECT: "Indirect:",
}

SYSTEM_PROMPTS = {
    QueryType.KEY_PHRASE: """You are an expert at optimizing audio captions for search engines. Analyze the given caption and compress it into a tight Key Phrase.""",
    QueryType.STATEMENT: """You are an expert at objectively describing audio scenarios. Analyze the given caption and write a full, descriptive Statement.""",
    QueryType.FULL_CAPTION: """You are an expert at combining fragmented audio captions into one vivid, unified scene. Analyze multiple captions and write a Full-caption.""",
}


def _with_output_schema(prompt: str, query_type: QueryType) -> str:
    start_word_instruction = ""
    if query_type in START_WORD_VARIATION_TYPES:
        start_word_count = random.randint(*START_WORD_COUNT_RANGE)
        start_word_instruction = (
            "\n\nIn \"explanation\", first generate "
            f"{start_word_count} plausible starting words or phrases for the final query, "
            "then choose a final starting word or phrase that is not in that generated list. "
            "The \"answer\" must start with that non-listed choice. Explain briefly that the "
            "final query starts with the non-listed choice."
        )

    return f"""{prompt}

[Output Schema]
Return only a valid JSON object with exactly these string fields:
- "answer": the final query text.
- "explanation": a concise explanation of how you chose the final query.
{start_word_instruction}

Do not include any field named "reasoning"."""


def format_prompt(query_type: QueryType, caption: str) -> str:
    safe_caption = json.dumps(caption)

    if query_type == QueryType.FULL_CAPTION:
        return FULL_CAPTION_PROMPT_TEMPLATE.format(caption=safe_caption)

    suffix = SINGLE_REFERENCE_SUFFIX[query_type]
    return f"Source Caption: {safe_caption}\n{suffix}"


def get_system_prompt(query_type: QueryType, backend: str = "gpt") -> str:
    del backend

    if query_type == QueryType.QUESTION:
        return _with_output_schema(
            """You are an expert at generating questions to verify the presence of audio content. Analyze the given caption and create a natural Yes/No Question.""",
            query_type,
        )

    elif query_type == QueryType.COMMAND:
        return _with_output_schema(
            """You are an expert at crafting precise instructions for search systems or agents. Analyze the given caption and generate a direct Command.""",
            query_type,
        )

    elif query_type == QueryType.INDIRECT:
        return _with_output_schema(
            """You are an expert in highly polite and conversational communication. Create an Indirect/Polite Request asking to find the sounds described in the caption.""",
            query_type,
        )

    else:
        return _with_output_schema(SYSTEM_PROMPTS[query_type], query_type)
