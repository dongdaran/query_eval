#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Any, Iterator

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
QUERY_TYPE_ORDER = [
    "full_caption",
    "key_phrase",
    "statement",
    "question",
    "command",
    "indirect",
]


def iter_json_objects(path: Path) -> Iterator[dict[str, Any]]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return

    try:
        for line_number, line in enumerate(text.splitlines(), start=1):
            stripped = line.strip()
            if stripped:
                obj = json.loads(stripped)
                if not isinstance(obj, dict):
                    raise ValueError(f"Expected JSON object at {path}:{line_number}")
                yield obj
        return
    except json.JSONDecodeError:
        pass

    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        while idx < len(text) and text[idx].isspace():
            idx += 1
        if idx >= len(text):
            break
        obj, end = decoder.raw_decode(text, idx)
        if not isinstance(obj, dict):
            raise ValueError(f"Expected JSON object in {path} at offset {idx}")
        yield obj
        idx = end


def tokenize(text: str) -> list[str]:
    return text.lower().split()


def ngram_counts(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    if n <= 0 or len(tokens) < n:
        return Counter()
    return Counter(tuple(tokens[idx : idx + n]) for idx in range(len(tokens) - n + 1))


def closest_ref_length(hyp_len: int, ref_lens: list[int]) -> int:
    return min(ref_lens, key=lambda ref_len: (abs(ref_len - hyp_len), ref_len))


def sentence_bleu(
    hypothesis: str,
    references: list[str],
    max_order: int = 4,
    smooth: float = 1.0,
) -> float:
    hyp_tokens = tokenize(hypothesis)
    ref_tokens = [tokenize(ref) for ref in references if ref.strip()]
    if not hyp_tokens or not ref_tokens:
        return 0.0

    precisions: list[float] = []
    for n in range(1, max_order + 1):
        hyp_counts = ngram_counts(hyp_tokens, n)
        total = sum(hyp_counts.values())
        if total == 0:
            precisions.append(0.0)
            continue

        max_ref_counts: Counter[tuple[str, ...]] = Counter()
        for ref in ref_tokens:
            ref_counts = ngram_counts(ref, n)
            for gram, count in ref_counts.items():
                if count > max_ref_counts[gram]:
                    max_ref_counts[gram] = count

        overlap = sum(min(count, max_ref_counts[gram]) for gram, count in hyp_counts.items())
        precisions.append((overlap + smooth) / (total + smooth))

    if any(precision <= 0.0 for precision in precisions):
        return 0.0

    hyp_len = len(hyp_tokens)
    ref_len = closest_ref_length(hyp_len, [len(ref) for ref in ref_tokens])
    brevity_penalty = 1.0 if hyp_len > ref_len else math.exp(1.0 - (ref_len / hyp_len))
    return brevity_penalty * math.exp(sum(math.log(precision) for precision in precisions) / max_order)


def self_bleu(responses: list[str], max_order: int = 4, smooth: float = 1.0) -> float | None:
    cleaned = [response.strip() for response in responses if response.strip()]
    if len(cleaned) < 2:
        return None
    scores = []
    for idx, response in enumerate(cleaned):
        refs = cleaned[:idx] + cleaned[idx + 1 :]
        scores.append(sentence_bleu(response, refs, max_order=max_order, smooth=smooth))
    return sum(scores) / len(scores)


def pairwise_embedding_distance(embeddings: np.ndarray) -> float | None:
    if len(embeddings) < 2:
        return None
    distances = []
    for left, right in combinations(range(len(embeddings)), 2):
        cosine = float(np.dot(embeddings[left], embeddings[right]))
        distances.append(1.0 - cosine)
    return sum(distances) / len(distances)


def query_type_sort_key(query_type: str) -> tuple[int, str]:
    try:
        return (QUERY_TYPE_ORDER.index(query_type), query_type)
    except ValueError:
        return (len(QUERY_TYPE_ORDER), query_type)


def extract_responses(row: dict[str, Any]) -> list[tuple[str, str]]:
    generated_queries = row.get("generated_queries", {})
    if not isinstance(generated_queries, dict):
        return []
    items = [
        (str(query_type), str(query_text).strip())
        for query_type, query_text in generated_queries.items()
        if str(query_text).strip()
    ]
    return sorted(items, key=lambda item: query_type_sort_key(item[0]))


def mean_present(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return sum(present) / len(present)


def load_sentence_transformer(model_name: str, device: str | None):
    os.environ.setdefault("HF_HOME", str(PROJECT_ROOT / "checkpoints" / ".cache" / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(PROJECT_ROOT / "input" / "datasets"))
    from sentence_transformers import SentenceTransformer

    kwargs = {"device": device} if device else {}
    return SentenceTransformer(model_name, **kwargs)


def compute_semantic_diversity(
    per_record: list[dict[str, Any]],
    model_name: str,
    batch_size: int,
    device: str | None,
) -> None:
    model = load_sentence_transformer(model_name, device)
    all_texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for record in per_record:
        start = len(all_texts)
        all_texts.extend(record["responses"])
        spans.append((start, len(all_texts)))

    if not all_texts:
        return

    embeddings = model.encode(
        all_texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=True,
    )
    for record, (start, end) in zip(per_record, spans):
        record["semantic_diversity"] = pairwise_embedding_distance(embeddings[start:end])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute per-record self-BLEU and semantic diversity over generated_queries "
            "in a WOCOT/WCOT JSONL file."
        )
    )
    parser.add_argument(
        "--input",
        nargs="+",
        default=["results/wocottest100.jsonl"],
        help="One or more input JSONL paths.",
    )
    parser.add_argument(
        "--output-json",
        default="results/wocottest100_diversity.json",
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--output-csv",
        default="results/wocottest100_diversity_by_record.csv",
        help="Per-record CSV output path.",
    )
    parser.add_argument("--max-bleu-order", type=int, default=4, help="Maximum n-gram order for self-BLEU.")
    parser.add_argument("--bleu-smooth", type=float, default=1.0, help="Additive smoothing for BLEU precision.")
    parser.add_argument(
        "--embedding-model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model for semantic diversity.",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Embedding batch size.")
    parser.add_argument("--device", default=None, help="SentenceTransformer device, e.g. cpu, cuda, cuda:0.")
    parser.add_argument(
        "--group-by",
        choices=["row", "source_caption", "source_caption_query_type"],
        default="row",
        help=(
            "row: compare generated_queries inside each JSONL row. "
            "source_caption: pool responses from rows with the same source_caption. "
            "source_caption_query_type: pool only same query_type responses under each source_caption."
        ),
    )
    parser.add_argument(
        "--k-values",
        nargs="+",
        type=int,
        default=None,
        help="Response counts to evaluate per group, e.g. --k-values 1 5 10 20 100.",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=1,
        help="Number of random samples per group/k when more than k responses are available.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for k-response sampling.")
    parser.add_argument(
        "--skip-semantic",
        action="store_true",
        help="Only compute self-BLEU; do not load the embedding model.",
    )
    return parser.parse_args()


def build_row_records(
    rows: list[dict[str, Any]],
    max_bleu_order: int,
    bleu_smooth: float,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        response_items = extract_responses(row)
        responses = [text for _, text in response_items]
        bleu = self_bleu(responses, max_order=max_bleu_order, smooth=bleu_smooth)
        records.append(
            {
                "row_index": idx,
                "audio_id": str(row.get("audio_id", "")),
                "group_key": str(row.get("audio_id", "")),
                "source_caption": str(row.get("source_caption", "")),
                "dataset": str(row.get("dataset", "")),
                "dataset_slug": str(row.get("dataset_slug", "")),
                "k": len(responses),
                "query_types": [query_type for query_type, _ in response_items],
                "responses": responses,
                "self_bleu": bleu,
                "semantic_diversity": None,
            }
        )
    return records


def sample_responses(
    responses: list[str],
    k: int,
    rng: random.Random,
) -> list[str] | None:
    if k < 1 or len(responses) < k:
        return None
    if len(responses) == k:
        return list(responses)
    return rng.sample(responses, k)


def build_source_caption_records(
    rows: list[dict[str, Any]],
    k_values: list[int],
    trials: int,
    seed: int,
    max_bleu_order: int,
    bleu_smooth: float,
) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in rows:
        source_caption = str(row.get("source_caption", "")).strip()
        if not source_caption:
            continue
        group = grouped.setdefault(
            source_caption,
            {
                "source_caption": source_caption,
                "dataset": str(row.get("dataset", "")),
                "dataset_slug": str(row.get("dataset_slug", "")),
                "audio_ids": [],
                "responses": [],
                "query_types": [],
            },
        )
        group["audio_ids"].append(str(row.get("audio_id", "")))
        for query_type, text in extract_responses(row):
            group["responses"].append(text)
            group["query_types"].append(query_type)

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    row_index = 0
    for source_caption, group in sorted(grouped.items()):
        responses = group["responses"]
        for k in k_values:
            if len(responses) < k:
                continue
            for trial in range(max(1, trials)):
                sampled = sample_responses(responses, k, rng)
                if sampled is None:
                    continue
                row_index += 1
                records.append(
                    {
                        "row_index": row_index,
                        "audio_id": "|".join(group["audio_ids"]),
                        "group_key": source_caption,
                        "source_caption": source_caption,
                        "dataset": group["dataset"],
                        "dataset_slug": group["dataset_slug"],
                        "k": k,
                        "available_responses": len(responses),
                        "trial": trial,
                        "query_types": group["query_types"],
                        "responses": sampled,
                        "self_bleu": self_bleu(sampled, max_order=max_bleu_order, smooth=bleu_smooth),
                        "semantic_diversity": None,
                    }
                )
    return records


def build_source_caption_query_type_records(
    rows: list[dict[str, Any]],
    k_values: list[int],
    trials: int,
    seed: int,
    max_bleu_order: int,
    bleu_smooth: float,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        source_caption = str(row.get("source_caption", "")).strip()
        if not source_caption:
            continue
        for query_type, text in extract_responses(row):
            key = (source_caption, query_type)
            group = grouped.setdefault(
                key,
                {
                    "source_caption": source_caption,
                    "query_type": query_type,
                    "dataset": str(row.get("dataset", "")),
                    "dataset_slug": str(row.get("dataset_slug", "")),
                    "audio_ids": [],
                    "responses": [],
                },
            )
            group["audio_ids"].append(str(row.get("audio_id", "")))
            group["responses"].append(text)

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []
    row_index = 0
    for (source_caption, query_type), group in sorted(grouped.items()):
        responses = group["responses"]
        for k in k_values:
            if len(responses) < k:
                continue
            for trial in range(max(1, trials)):
                sampled = sample_responses(responses, k, rng)
                if sampled is None:
                    continue
                row_index += 1
                records.append(
                    {
                        "row_index": row_index,
                        "audio_id": "|".join(group["audio_ids"]),
                        "group_key": f"{source_caption} || {query_type}",
                        "source_caption": source_caption,
                        "dataset": group["dataset"],
                        "dataset_slug": group["dataset_slug"],
                        "k": k,
                        "available_responses": len(responses),
                        "trial": trial,
                        "query_types": [query_type],
                        "responses": sampled,
                        "self_bleu": self_bleu(sampled, max_order=max_bleu_order, smooth=bleu_smooth),
                        "semantic_diversity": None,
                    }
                )
    return records


def main() -> None:
    args = parse_args()
    input_paths = [Path(path) for path in args.input]
    rows: list[dict[str, Any]] = []
    for input_path in input_paths:
        rows.extend(iter_json_objects(input_path))

    if args.group_by == "source_caption_query_type":
        k_values = args.k_values or [1, 5, 10, 20, 100]
        per_record = build_source_caption_query_type_records(
            rows,
            k_values=k_values,
            trials=args.trials,
            seed=args.seed,
            max_bleu_order=args.max_bleu_order,
            bleu_smooth=args.bleu_smooth,
        )
    elif args.group_by == "source_caption":
        k_values = args.k_values or [1, 5, 10, 20, 100]
        per_record = build_source_caption_records(
            rows,
            k_values=k_values,
            trials=args.trials,
            seed=args.seed,
            max_bleu_order=args.max_bleu_order,
            bleu_smooth=args.bleu_smooth,
        )
    else:
        per_record = build_row_records(rows, args.max_bleu_order, args.bleu_smooth)

    if not args.skip_semantic:
        compute_semantic_diversity(per_record, args.embedding_model, args.batch_size, args.device)

    summary = {
        "input": [str(path) for path in input_paths],
        "group_by": args.group_by,
        "records": len(per_record),
        "self_bleu": mean_present([record["self_bleu"] for record in per_record]),
        "semantic_diversity": mean_present([record["semantic_diversity"] for record in per_record]),
        "max_bleu_order": args.max_bleu_order,
        "bleu_smooth": args.bleu_smooth,
        "embedding_model": None if args.skip_semantic else args.embedding_model,
        "k_values": sorted({record["k"] for record in per_record}),
        "by_dataset_slug": {},
        "by_k": {},
    }

    dataset_slugs = sorted({record["dataset_slug"] for record in per_record})
    for dataset_slug in dataset_slugs:
        records = [record for record in per_record if record["dataset_slug"] == dataset_slug]
        summary["by_dataset_slug"][dataset_slug] = {
            "records": len(records),
            "self_bleu": mean_present([record["self_bleu"] for record in records]),
            "semantic_diversity": mean_present([record["semantic_diversity"] for record in records]),
        }
    for k in sorted({record["k"] for record in per_record}):
        records = [record for record in per_record if record["k"] == k]
        summary["by_k"][str(k)] = {
            "records": len(records),
            "self_bleu": mean_present([record["self_bleu"] for record in records]),
            "semantic_diversity": mean_present([record["semantic_diversity"] for record in records]),
        }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        fieldnames = [
            "row_index",
            "audio_id",
            "group_key",
            "source_caption",
            "dataset",
            "dataset_slug",
            "k",
            "available_responses",
            "trial",
            "query_types",
            "self_bleu",
            "semantic_diversity",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for record in per_record:
            writer.writerow(
                {
                    "row_index": record["row_index"],
                    "audio_id": record["audio_id"],
                    "group_key": record["group_key"],
                    "source_caption": record["source_caption"],
                    "dataset": record["dataset"],
                    "dataset_slug": record["dataset_slug"],
                    "k": record["k"],
                    "available_responses": record.get("available_responses", record["k"]),
                    "trial": record.get("trial", 0),
                    "query_types": "|".join(record["query_types"]),
                    "self_bleu": "" if record["self_bleu"] is None else f"{record['self_bleu']:.8f}",
                    "semantic_diversity": ""
                    if record["semantic_diversity"] is None
                    else f"{record['semantic_diversity']:.8f}",
                }
            )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {output_json}")
    print(f"Wrote {output_csv}")


if __name__ == "__main__":
    main()
