#!/usr/bin/env -S uv run python
"""AudioCaps-style CSV → EQ JSONL (key_phrase, statement, question, command, indirect only).

Expects columns compatible with ``load_audiocaps`` (at minimum ``youtube_id``, ``start_time``, ``caption``).
Extra columns (e.g. ``audiocap_id``) are ignored by the loader.

Example:
  uv run python scripts/makeeq_audiocaps_csv_five.py \\
    --csv input/audiocaps/train1000.csv \\
    --output-dir output/eq_train1000 \\
    --split train
"""
from __future__ import annotations

import argparse
import importlib.util
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from eq_generation import (  # noqa: E402
    EQGenerator,
    QueryResult,
    QueryType,
    load_audiocaps,
    load_config,
)

FIVE_EQ_TYPES: list[QueryType] = [
    QueryType.KEY_PHRASE,
    QueryType.STATEMENT,
    QueryType.QUESTION,
    QueryType.COMMAND,
    QueryType.INDIRECT,
]

EQ_OUTPUT_FILENAMES: dict[QueryType, str] = {
    QueryType.KEY_PHRASE: "eq_key_phrase.jsonl",
    QueryType.STATEMENT: "eq_statement.jsonl",
    QueryType.QUESTION: "eq_question.jsonl",
    QueryType.COMMAND: "eq_command.jsonl",
    QueryType.INDIRECT: "eq_indirect.jsonl",
}


def _load_makeeq_helpers() -> Any:
    path = Path(__file__).resolve().parent / "makeeq.py"
    spec = importlib.util.spec_from_file_location("makeeq_helpers", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load makeeq.py from {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def append_log(log_lines: list[str], message: str) -> None:
    print(message)
    log_lines.append(message)


def validate_outputs_five(
    prepared_entries: list[dict[str, Any]],
    outputs_by_type: dict[QueryType, list[QueryResult]],
    log_lines: list[str],
    middle_caption: Callable[[list[str]], str],
) -> None:
    counts: Counter[str] = Counter()
    expected_by_audio_id = {entry["audio_id"]: entry for entry in prepared_entries}

    for query_type, results in outputs_by_type.items():
        if len(results) != len(prepared_entries):
            append_log(
                log_lines,
                f"[ERROR] {query_type.value} produced {len(results)} outputs "
                f"for {len(prepared_entries)} prepared entries.",
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
            expected_oc = [middle_caption(caps)]
            if result.original_captions != expected_oc:
                append_log(
                    log_lines,
                    f"[ERROR] {query_type.value} output for audio_id={result.audio_id} "
                    "did not retain expected original_captions (middle caption only).",
                )

    expected_n = len(FIVE_EQ_TYPES)
    for entry in prepared_entries:
        produced = counts[entry["audio_id"]]
        if produced != expected_n:
            append_log(
                log_lines,
                f"[ERROR] audio_id={entry['audio_id']} produced {produced} outputs; "
                f"expected {expected_n}.",
            )


def write_validation_log(output_dir: Path, log_lines: list[str]) -> Path:
    log_path = output_dir / "eq_validation.log"
    log_path.write_text("\n".join(log_lines) + "\n", encoding="utf-8")
    return log_path


def main() -> None:
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="Generate five EQ types from an AudioCaps-format caption CSV (no full_caption).",
    )
    parser.add_argument(
        "--csv",
        required=True,
        help="Path to CSV (youtube_id, start_time, caption, …).",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for five EQ JSONL files.")
    parser.add_argument(
        "--split",
        default="train",
        help="Split label stored in metadata (dataset_slug). Default: train",
    )
    parser.add_argument(
        "--num-queries",
        type=int,
        help="Optional limit: first N unique clips after grouping (CSV order).",
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration file")
    args = parser.parse_args()

    config = load_config(args.config)
    model_config = config.get("model", {})
    source_model = model_config.get("source_model", "gpt-5.4-mini")
    regen_model = model_config.get("regen_model", "gpt-5.4-mini")
    backend = model_config.get("backend", "gpt")
    temperature = model_config.get("temperature", 0.35)
    batch_size = model_config.get("batch_size", 2)
    max_tokens = model_config.get("max_tokens", 1024)
    top_p = model_config.get("top_p", 0.9)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_lines: list[str] = []
    makeeq = _load_makeeq_helpers()
    prepare_entries_from_records = makeeq.prepare_entries_from_records
    build_results_for_query_type = makeeq.build_results_for_query_type

    append_log(log_lines, f"[INFO] Loading records from {args.csv}")
    records = load_audiocaps(args.csv, split=args.split)
    append_log(log_lines, f"[INFO] Loaded {len(records)} unique audio_id groups")

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
        batch_size=batch_size,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
    )

    outputs_by_type: dict[QueryType, list[QueryResult]] = {}
    for query_type in FIVE_EQ_TYPES:
        append_log(
            log_lines,
            f"[INFO] Generating {query_type.value} for {len(prepared_entries)} entries",
        )
        results = build_results_for_query_type(
            prepared_entries=prepared_entries,
            generator=generator,
            query_type=query_type,
            source_model=source_model,
            regen_model=regen_model,
        )
        outputs_by_type[query_type] = results
        generator.save_results(
            results,
            output_dir / EQ_OUTPUT_FILENAMES[query_type],
            format="jsonl",
        )

    validate_outputs_five(
        prepared_entries,
        outputs_by_type,
        log_lines,
        makeeq.middle_caption,
    )

    error_count = sum(1 for line in log_lines if line.startswith("[ERROR]"))
    if error_count:
        append_log(log_lines, f"[ERROR] Validation finished with {error_count} error(s).")
        log_path = write_validation_log(output_dir, log_lines)
        raise SystemExit(f"See validation log: {log_path}")

    append_log(
        log_lines,
        f"[INFO] EQ generation complete. "
        f"Produced {len(prepared_entries) * len(FIVE_EQ_TYPES)} outputs.",
    )
    log_path = write_validation_log(output_dir, log_lines)
    print(f"[INFO] Validation log saved to {log_path}")


if __name__ == "__main__":
    main()
