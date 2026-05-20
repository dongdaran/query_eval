#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoTokenizer


QUERY_TYPE_ORDER = [
    "key_phrase",
    "statement",
    "question",
    "command",
    "indirect",
    "full_caption",
]


def iter_json_objects(path: Path) -> Iterator[dict[str, Any]]:
    decoder = json.JSONDecoder()
    text = path.read_text(encoding="utf-8")
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


def normalize_captions(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(item).strip() for item in value if str(item).strip()]


def parse_range_run(path: Path, input_root: Path) -> tuple[str, str, str]:
    rel = path.relative_to(input_root)
    if len(rel.parts) < 4:
        raise ValueError(f"Expected <range>/<run>/<dataset>/eq_*.jsonl under {input_root}: {path}")
    return rel.parts[0], rel.parts[1], rel.parts[2]


def query_type_key(query_type: str) -> tuple[int, str]:
    try:
        return (QUERY_TYPE_ORDER.index(query_type), query_type)
    except ValueError:
        return (len(QUERY_TYPE_ORDER), query_type)


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def format_float(value: float) -> str:
    return f"{value:.6f}"


def load_records(input_root: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in sorted(input_root.glob("range_*_*/run_*/**/eq_*.jsonl")):
        range_name, run_name, dataset_dir = parse_range_run(path, input_root)
        fallback_query_type = path.stem.removeprefix("eq_")
        for obj in iter_json_objects(path):
            generated_query = str(obj.get("generated_query", "")).strip()
            refs = normalize_captions(obj.get("original_captions"))
            if not generated_query or not refs:
                continue
            records.append(
                {
                    "range": range_name,
                    "run": run_name,
                    "dataset": str(obj.get("dataset_slug") or dataset_dir),
                    "query_type": str(obj.get("query_type") or fallback_query_type),
                    "audio_id": str(obj.get("audio_id", "")),
                    "candidate": generated_query,
                    "references": refs,
                    "path": str(path),
                }
            )
    return records


def embed_texts(
    texts: list[str],
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> dict[str, torch.Tensor]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    if config.model_type == "roberta":
        config.add_pooling_layer = False
    model = AutoModel.from_pretrained(model_name, config=config).to(device)
    model.eval()

    out: dict[str, torch.Tensor] = {}
    unique_texts = sorted(set(texts))
    with torch.no_grad():
        for start in range(0, len(unique_texts), batch_size):
            batch = unique_texts[start : start + batch_size]
            encoded = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_special_tokens_mask=True,
            )
            encoded = {key: value.to(device) for key, value in encoded.items()}
            hidden = model(
                input_ids=encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
            ).last_hidden_state
            hidden = F.normalize(hidden, p=2, dim=-1)
            valid_mask = encoded["attention_mask"].bool()
            if "special_tokens_mask" in encoded:
                valid_mask = valid_mask & ~encoded["special_tokens_mask"].bool()

            for text, token_embeddings, valid, attention in zip(
                batch,
                hidden,
                valid_mask,
                encoded["attention_mask"].bool(),
            ):
                kept = token_embeddings[valid].detach().cpu()
                if kept.numel() == 0:
                    kept = token_embeddings[attention].detach().cpu()
                out[text] = kept
    return out


def bertscore_precision(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    if candidate.numel() == 0 or reference.numel() == 0:
        return 0.0
    sim = candidate @ reference.T
    return float(sim.max(dim=1).values.mean().item())


def add_group_scores(
    group_scores: dict[tuple[str, str, str, str], list[float]],
    record: dict[str, Any],
    score: float,
) -> None:
    range_name = str(record["range"])
    run_name = str(record["run"])
    dataset = str(record["dataset"])
    query_type = str(record["query_type"])
    group_scores[(range_name, run_name, dataset, query_type)].append(score)
    group_scores[(range_name, run_name, "ALL", query_type)].append(score)


def write_by_run(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "range",
        "run",
        "dataset",
        "query_type",
        "records",
        "bertscore_precision_mean",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "range",
        "dataset",
        "query_type",
        "runs",
        "records_per_run_min",
        "records_per_run_max",
        "bertscore_precision_mean",
        "bertscore_precision_std",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute BERTScore precision for EQ prompt-range runs.",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/eq_top30_prompt_ranges"),
        help="Root containing range_*/run_*/dataset/eq_*.jsonl outputs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for CSV outputs. Default: <input-root>/bertscore_precision",
    )
    parser.add_argument("--model", default="roberta-large", help="HF encoder model for BERTScore.")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    args = parser.parse_args()

    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"

    records = load_records(args.input_root)
    if not records:
        raise SystemExit(f"No EQ prompt-range records found under {args.input_root}")

    output_dir = args.output_dir or (args.input_root / "bertscore_precision")
    output_dir.mkdir(parents=True, exist_ok=True)

    texts: list[str] = []
    for record in records:
        texts.append(str(record["candidate"]))
        texts.extend(str(ref) for ref in record["references"])

    embeddings = embed_texts(
        texts=texts,
        model_name=args.model,
        device=device,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )

    group_scores: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    for record in records:
        candidate_emb = embeddings[str(record["candidate"])]
        ref_scores = [
            bertscore_precision(candidate_emb, embeddings[str(ref)])
            for ref in record["references"]
        ]
        add_group_scores(group_scores, record, mean(ref_scores))

    by_run_rows = []
    for (range_name, run_name, dataset, query_type), scores in group_scores.items():
        by_run_rows.append(
            {
                "range": range_name,
                "run": run_name,
                "dataset": dataset,
                "query_type": query_type,
                "records": len(scores),
                "bertscore_precision_mean": format_float(mean(scores)),
            }
        )
    by_run_rows.sort(
        key=lambda row: (
            str(row["range"]),
            str(row["run"]),
            str(row["dataset"]),
            query_type_key(str(row["query_type"])),
        )
    )

    run_means: dict[tuple[str, str, str], list[tuple[float, int]]] = defaultdict(list)
    for row in by_run_rows:
        run_means[(str(row["range"]), str(row["dataset"]), str(row["query_type"]))].append(
            (float(row["bertscore_precision_mean"]), int(row["records"]))
        )

    summary_rows = []
    for (range_name, dataset, query_type), values in run_means.items():
        means = [value for value, _ in values]
        counts = [count for _, count in values]
        summary_rows.append(
            {
                "range": range_name,
                "dataset": dataset,
                "query_type": query_type,
                "runs": len(values),
                "records_per_run_min": min(counts),
                "records_per_run_max": max(counts),
                "bertscore_precision_mean": format_float(mean(means)),
                "bertscore_precision_std": format_float(sample_std(means)),
            }
        )
    summary_rows.sort(
        key=lambda row: (
            str(row["range"]),
            str(row["dataset"]),
            query_type_key(str(row["query_type"])),
        )
    )

    by_run_path = output_dir / "bertscore_precision_by_run.csv"
    summary_path = output_dir / "bertscore_precision_summary.csv"
    write_by_run(by_run_path, by_run_rows)
    write_summary(summary_path, summary_rows)

    print(f"Wrote {by_run_path}")
    print(f"Wrote {summary_path}")
    print("\n# ALL dataset summary")
    print("range\tquery_type\truns\tmean\tstd")
    for row in summary_rows:
        if row["dataset"] != "ALL":
            continue
        print(
            f"{row['range']}\t{row['query_type']}\t{row['runs']}\t"
            f"{row['bertscore_precision_mean']}\t{row['bertscore_precision_std']}"
        )


if __name__ == "__main__":
    main()
