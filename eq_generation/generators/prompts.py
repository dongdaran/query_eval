from __future__ import annotations

import json

from eq_generation.query_types import QueryType

START_WORD_COUNT_RANGE = (3, 15)


def set_start_word_count_range(min_count: int, max_count: int) -> None:
    """Keep CLI compatibility; baseline prompts no longer use start-word sampling."""
    if min_count < 0:
        raise ValueError("min_count must be non-negative")
    if max_count < min_count:
        raise ValueError("max_count must be greater than or equal to min_count")
    global START_WORD_COUNT_RANGE
    START_WORD_COUNT_RANGE = (min_count, max_count)

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
    QueryType.KEY_PHRASE: """You are an expert at optimizing audio captions for search engines. Analyze the given caption and compress it into a tight Key Phrase.
    
    [Constraints]
    1. Strictly NO complete sentences (subject + verb). You must use a Noun Phrase format.
    2. Remove unnecessary prepositions and conjunctions (e.g., 'while', 'from a').
    3. Convert long background descriptions containing verbs into concise modifiers.
    4. Maintain the texture and action of the sound by combining the sound's subject with a present participle (-ing) instead of a regular verb.
    5. Avoid using 'while' or 'and' to connect background sounds/environments. Instead, compress and connect them using prepositions like 'with', 'on', 'in', or adjectives like 'distant'.""",
    QueryType.STATEMENT: """You are an expert at objectively describing audio scenarios. Analyze the given caption and write a full, descriptive Statement.
    
    [Constraints]
    1. Reconstruct the description into a grammatically perfect sentence complete with a subject and a verb.
    2. Soften stiff noun phrases (e.g., 'repeated barking') into verb-modifying structures (e.g., 'barking repeatedly').
    3. When describing background sounds or simultaneous events, do not compress them. Use conjunctions like 'while' and 'and' to naturally connect two clauses.
    4. Since this is a complete sentence, it must end with a period (.).""",
    QueryType.FULL_CAPTION: """You are an expert at combining fragmented audio captions into one vivid, unified scene. Analyze multiple captions and write a Full-caption.
    
    [Constraints]
    1. Gather all unique details scattered across each caption (e.g., 'storm', 'steady', 'pattering') **without leaving any out**.
    2. Consolidate phrases that have different wording but the same meaning into a single, clean expression.
    3. Do not just list fragmented noun phrases. Reconstruct them into a complete, vivid sentence (e.g., 'is barking', 'is falling', 'is brewing') as if a video is playing right before the eyes.
    4. Do not stiffly glue the collected sound information (main sound, background, texture) together. Weave them into a fluent, single-breath sentence using connectors like 'accompanied by', 'while', 'duri
    ng', or ', producing'.
    5. Do not add environmental details, emotions, or causal interpretations that are not supported by the original captions.
    """,
}


def _with_output_schema(prompt: str, query_type: QueryType) -> str:
    del query_type

    return f"""{prompt}

[Output Schema]
Return only a valid JSON object with exactly these string fields:
- "answer": the final query text.
Do not include an explanation field or any additional fields."""


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
            """You are an expert at generating questions to verify the presence of audio content. Analyze the given caption and create a natural Yes/No Question.
            
            [Constraints]
            1. Do not severely compress or break apart the original sound description; preserve the important audible content.
            2. Convert the caption into a fluent yes/no question using natural wording.
            3. Slightly polish any overly mechanical expressions caused by the question format into natural wording (e.g., 'producing' -> 'brewing').
            4. Since this is a complete question, it must end with a question mark (?).""",
            query_type,
        )

    elif query_type == QueryType.COMMAND:
        return _with_output_schema(
            """You are an expert at crafting precise instructions for search systems or agents. Analyze the given caption and generate a direct Command.
            
            [Constraints]
            1. Do not use a subject. Start the sentence directly with a natural action verb to instruct the system.
            2. Include only search/extraction directives aimed at the system, removing any unnecessary modifiers or predicates.
            3. Do not overly compress or excessively describe the original text. Group the description into one large noun phrase and provide it as the direct object of the command.
            4. Preserve the important audible content from the caption.
            5. This is a complete command sentence, so it must end with a period (.).""",
            query_type,
        )

    elif query_type == QueryType.INDIRECT:
        return _with_output_schema(
            """You are an expert in highly polite and conversational communication. Create an Indirect/Polite Request asking to find the sounds described in the caption.
            
            [Constraints]
            1. Start with a natural polite or indirect expression that asks for the other party's willingness or assistance.
            2. Smoothly connect the polite expression to the task of finding/locating the sound described in the original caption.
            3. Preserve the important audible content from the caption.
            4. Structure the sentence appropriately as a statement or a question, depending on the polite introduction, and end with the correct punctuation (. or ?)""",
            query_type,
        )

    else:
        return _with_output_schema(SYSTEM_PROMPTS[query_type], query_type)
