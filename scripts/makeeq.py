#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent.parent))

from eq_generation import (
    EQGenerator,
    QueryResult,
    QueryType,
    load_audiocaps,
    load_clotho,
    load_config,
    load_macs,
    load_mecat,
)
from eq_generation.generators.prompts import set_start_word_count_range

EQ_QUERY_TYPES = [
    QueryType.KEY_PHRASE,
    QueryType.STATEMENT,
    QueryType.QUESTION,
    QueryType.COMMAND,
    QueryType.INDIRECT,
    QueryType.FULL_CAPTION,
]

EQ_OUTPUT_FILENAMES = {
    QueryType.KEY_PHRASE: "eq_key_phrase.jsonl",
    QueryType.STATEMENT: "eq_statement.jsonl",
    QueryType.QUESTION: "eq_question.jsonl",
    QueryType.COMMAND: "eq_command.jsonl",
    QueryType.INDIRECT: "eq_indirect.jsonl",
    QueryType.FULL_CAPTION: "eq_full_caption.jsonl",
}

QUERY_TYPES_BY_VALUE = {query_type.value: query_type for query_type in EQ_QUERY_TYPES}


def format_caption_set(captions: list[str]) -> str:
    return "\n".join(f"- {caption}" for caption in captions)


def middle_caption(captions: list[str]) -> str:
    """Index len//2 (e.g. 5 captions -> index 2)."""
    if not captions:
        return ""
    return captions[len(captions) // 2]


def append_log(log_lines: list[str], message: str) -> None:
    print(message)
    log_lines.append(message)


def prepare_entries_from_records(
    records: list[dict[str, Any]],
    log_lines: list[str],
) -> list[dict[str, Any]]:
    """One row per unique audio_id. full_caption uses all captions; other types use middle caption only."""
    prepared: list[dict[str, Any]] = []
    for idx, record in enumerate(records):
        caption_set = record.get("original_captions") or []
        if not caption_set:
            append_log(
                log_lines,
                f"[WARN] Skipping record {idx} audio_id={record.get('audio_id')!r}: no captions.",
            )
            continue
        prepared.append(
            {
                "audio_id": str(record["audio_id"]).strip(),
                "dataset": record["dataset"],
                "dataset_slug": record["dataset_slug"],
                "caption_set": caption_set,
                "metadata": {
                    **record.get("metadata", {}),
                    "caption_count": len(caption_set),
                },
            }
        )
    return prepared


def build_results_for_query_type(
    prepared_entries: list[dict[str, Any]],
    generator: Any,
    query_type: QueryType,
    source_model: str,
    regen_model: str,
) -> list[QueryResult]:
    if query_type == QueryType.FULL_CAPTION:
        prompts = [format_caption_set(entry["caption_set"]) for entry in prepared_entries]
    else:
        prompts = [middle_caption(entry["caption_set"]) for entry in prepared_entries]

    raw_results = generator.generate(
        captions=prompts,
        query_type=query_type,
        clip_ids=[entry["audio_id"] for entry in prepared_entries],
        show_progress=True,
    )

    results = []
    for entry, raw_result in zip(prepared_entries, raw_results):
        caps = list(entry["caption_set"])
        if query_type == QueryType.FULL_CAPTION:
            original_captions = caps
            meta = dict(entry["metadata"])
        else:
            mid = middle_caption(caps)
            original_captions = [mid]
            meta = {
                **entry["metadata"],
                "eq_reference": "middle_caption",
                "middle_caption_index": len(caps) // 2 if caps else 0,
                "full_caption_count": len(caps),
            }
        results.append(
            QueryResult(
                audio_id=entry["audio_id"],
                dataset=entry["dataset"],
                dataset_slug=entry["dataset_slug"],
                query_type=query_type,
                generated_query=raw_result.generated_query,
                original_captions=original_captions,
                explanation=raw_result.explanation,
                metadata=meta,
                source_model=source_model,
                regen_model=regen_model,
            )
        )

    return results


def validate_outputs(
    prepared_entries: list[dict[str, Any]],
    outputs_by_type: dict[QueryType, list[QueryResult]],
    query_types: list[QueryType],
    log_lines: list[str],
) -> None:
    counts = Counter()
    expected_by_audio_id = {entry["audio_id"]: entry for entry in prepared_entries}

    for query_type, results in outputs_by_type.items():
        if len(results) != len(prepared_entries):
            n_prep = len(prepared_entries)
            append_log(
                log_lines,
                f"[ERROR] {query_type.value} produced {len(results)} outputs for {n_prep} prepared entries.",
            )

        for result in results:
            counts[result.audio_id] += 1
            expected_entry = expected_by_audio_id.get(result.audio_id)
            if expected_entry is None:
                append_log(
                    log_lines,
                    f"[ERROR] Unexpected output audio_id={result.audio_id} in {query_type.value}.",
                )
                continue

            caps = list(expected_entry["caption_set"])
            if query_type == QueryType.FULL_CAPTION:
                expected_oc = caps
            else:
                expected_oc = [middle_caption(caps)]
            if result.original_captions != expected_oc:
                append_log(
                    log_lines,
                    f"[ERROR] {query_type.value} output for audio_id={result.audio_id} "
                    "did not retain expected original_captions (full set for full_caption, else middle only).",
                )

    for entry in prepared_entries:
        produced = counts[entry["audio_id"]]
        if produced != len(query_types):
            append_log(
                log_lines,
                f"[ERROR] audio_id={entry['audio_id']} produced {produced} outputs; "
                f"expected {len(query_types)}.",
            )


def write_validation_log(output_dir: Path, log_lines: list[str]) -> Path:
    log_path = output_dir / "eq_validation.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return log_path


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="EQ generation from dataset captions or MECAT short-caption JSON files.",
    )
    parser.add_argument(
        "--dataset",
        required=True,
        choices=["audiocaps", "clotho", "mecat", "macs"],
        help="Dataset type used to choose the input loader.",
    )
    parser.add_argument(
        "--captions-csv",
        "--captions-path",
        dest="captions_path",
        required=True,
        help="Path to the caption CSV, MACS YAML, or MECAT JSON directory/JSONL file.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for per-type EQ JSONL files")
    parser.add_argument(
        "--split",
        default="test",
        help="Split label stored in output metadata. Default: test",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        help="Optional limit on clips (CSV / record order after grouping).",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="Zero-based inclusive start index after grouping. Default: 0",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Zero-based inclusive end index after grouping. Default: no end limit",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    parser.add_argument(
        "--query-types",
        nargs="+",
        choices=list(QUERY_TYPES_BY_VALUE),
        default=list(QUERY_TYPES_BY_VALUE),
        help="Query types to generate. Default: all query types.",
    )
    parser.add_argument(
        "--start-word-count-min",
        type=int,
        default=None,
        help="Minimum number of plausible starting words/phrases requested in prompt explanations.",
    )
    parser.add_argument(
        "--start-word-count-max",
        type=int,
        default=None,
        help="Maximum number of plausible starting words/phrases requested in prompt explanations.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config.get("model", {})
    source_model = model_config.get("source_model", "gpt-4.1-2025-04-14")
    regen_model = model_config.get("regen_model", "gpt-4.1-2025-04-14")
    backend = model_config.get("backend", "gpt")
    base_url = model_config.get("base_url") or model_config.get("api_base_url")
    temperature = model_config.get("temperature", 1.0)
    batch_size = model_config.get("batch_size", 2)
    max_tokens = model_config.get("max_tokens", 1024)
    top_p = model_config.get("top_p", 0.9)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    query_types = [QUERY_TYPES_BY_VALUE[value] for value in args.query_types]

    log_lines: list[str] = []
    append_log(log_lines, "[INFO] Query types: " + ", ".join(query_type.value for query_type in query_types))
    if args.start_word_count_min is not None or args.start_word_count_max is not None:
        if args.start_word_count_min is None or args.start_word_count_max is None:
            raise SystemExit("--start-word-count-min and --start-word-count-max must be provided together.")
        set_start_word_count_range(args.start_word_count_min, args.start_word_count_max)
        append_log(
            log_lines,
            "[INFO] Using start-word candidate count range: "
            f"{args.start_word_count_min}-{args.start_word_count_max}",
        )
    append_log(log_lines, f"[INFO] Loading records from {args.captions_path}")
    append_log(log_lines, f"[INFO] Using dataset loader: {args.dataset}")
    if args.dataset == "clotho":
        records = load_clotho(args.captions_path, split=args.split)
    elif args.dataset == "macs":
        records = load_macs(args.captions_path, split=args.split)
    elif args.dataset == "mecat":
        records = load_mecat(args.captions_path, split=args.split)
    else:
        records = load_audiocaps(args.captions_path, split=args.split)
    append_log(log_lines, f"[INFO] Loaded {len(records)} unique audio_id groups")

    if args.start_index < 0:
        raise SystemExit("--start-index must be >= 0.")
    if args.end_index is not None and args.end_index < args.start_index:
        raise SystemExit("--end-index must be >= --start-index.")

    if args.start_index or args.end_index is not None:
        slice_end = args.end_index + 1 if args.end_index is not None else None
        records = records[args.start_index : slice_end]
        end_label = args.end_index if args.end_index is not None else "end"
        append_log(
            log_lines,
            f"[INFO] Limited to index range {args.start_index}-{end_label}; "
            f"{len(records)} clips remain.",
        )

    if args.num_queries is not None and len(records) > args.num_queries:
        records = records[: args.num_queries]
        append_log(log_lines, f"[INFO] Limited to first {len(records)} clips via --num-queries.")

    prepared_entries = prepare_entries_from_records(records, log_lines)
    append_log(log_lines, f"[INFO] Prepared {len(prepared_entries)} entries.")
    if not prepared_entries:
        append_log(log_lines, "[ERROR] No prepared entries; aborting.")
        log_path = write_validation_log(output_dir, log_lines)
        raise SystemExit(f"See validation log: {log_path}")

    generator = EQGenerator(
        backend=backend,
        model=source_model,
        base_url=base_url,
        batch_size=batch_size,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    outputs_by_type: dict[QueryType, list[QueryResult]] = {}
    for query_type in query_types:
        append_log(log_lines, f"[INFO] Generating {query_type.value} for {len(prepared_entries)} entries")
        results = build_results_for_query_type(
            prepared_entries=prepared_entries,
            generator=generator,
            query_type=query_type,
            source_model=source_model,
            regen_model=regen_model,
        )
        outputs_by_type[query_type] = results
        generator.save_results(results, output_dir / EQ_OUTPUT_FILENAMES[query_type], format="jsonl")

    validate_outputs(prepared_entries, outputs_by_type, query_types, log_lines)

    error_count = sum(1 for line in log_lines if line.startswith("[ERROR]"))
    if error_count:
        append_log(log_lines, f"[ERROR] Validation finished with {error_count} error(s).")
        log_path = write_validation_log(output_dir, log_lines)
        raise SystemExit(f"See validation log: {log_path}")

    append_log(
        log_lines,
        f"[INFO] EQ generation complete. Produced {len(prepared_entries) * len(query_types)} outputs.",
    )
    log_path = write_validation_log(output_dir, log_lines)
    print(f"[INFO] Validation log saved to {log_path}")


if __name__ == "__main__":
    main()
