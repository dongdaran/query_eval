from __future__ import annotations

import json

from eq_generation.query_types import QueryType

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


def format_prompt(query_type: QueryType, caption: str) -> str:
    safe_caption = json.dumps(caption)

    if query_type == QueryType.FULL_CAPTION:
        return FULL_CAPTION_PROMPT_TEMPLATE.format(caption=safe_caption)

    suffix = SINGLE_REFERENCE_SUFFIX[query_type]
    return f"Source Caption: {safe_caption}\n{suffix}"


def get_system_prompt(query_type: QueryType, backend: str = "gpt") -> str:
    del backend

    if query_type == QueryType.QUESTION:
        return """You are an expert at generating questions to verify the presence of audio content. Analyze the given caption and create a natural Yes/No Question."""

    elif query_type == QueryType.COMMAND:
        return """You are an expert at crafting precise instructions for search systems or agents. Analyze the given caption and generate a direct Command."""

    elif query_type == QueryType.INDIRECT:
        return """You are an expert in highly polite and conversational communication. Create an Indirect/Polite Request asking to find the sounds described in the caption."""

    else:
        return SYSTEM_PROMPTS[query_type]
