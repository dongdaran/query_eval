#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
UIQ_ROOT = Path.home() / "UIQ"

MODELS = ("m2d", "msclap", "mga", "laion")
MODEL_LABELS = {
    "m2d": "M2D",
    "msclap": "MS",
    "mga": "MGA",
    "laion": "LAION",
}
DATASETS = ("mecat", "audiocaps", "clotho")
DATASET_ALIASES = {
    "mecat": ("mecat", "mecats", "mecat_test", "mecat_00A_train_top160"),
    "audiocaps": ("audiocaps", "audiocaps_test", "audiocaps_val"),
    "clotho": ("clotho", "clotho_test", "clotho_validation"),
}
CATEGORY_ORDER = (
    "all_wrong",
    "all_correct",
    "only_m2d",
    "only_msclap",
    "only_mga",
    "only_laion",
)
CATEGORY_LABELS = {
    "all_wrong": "4개 모델 모두 틀림",
    "all_correct": "4개 모델 모두 맞음",
    "only_m2d": "M2D만 맞음",
    "only_msclap": "MS만 맞음",
    "only_mga": "MGA만 맞음",
    "only_laion": "LAION만 맞음",
}


def iter_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at {path}:{line_number}") from exc
    return rows


def find_dataset_dir(dataset: str) -> Path | None:
    base = UIQ_ROOT / "results" / "retrieval"
    for alias in DATASET_ALIASES[dataset]:
        candidate = base / alias
        if candidate.is_dir():
            return candidate
    return None


def rank_is_hit_at5(row: dict[str, Any]) -> bool:
    if "rank" in row and row["rank"] is not None:
        return int(row["rank"]) <= 5
    for retrieved in row.get("retrieved_audios", [])[:5]:
        if retrieved.get("is_correct"):
            return True
    return False


def load_dataset_rows(dataset: str) -> tuple[list[dict[str, Any]], list[str]]:
    retrieval_dir = find_dataset_dir(dataset)
    if retrieval_dir is None:
        return [], [f"No retrieval directory found for {dataset} under {UIQ_ROOT / 'results' / 'retrieval'}"]

    warnings: list[str] = []
    by_audio: dict[str, dict[str, Any]] = {}

    for model in MODELS:
        path = retrieval_dir / f"{model}_original_retrieval.jsonl"
        if not path.is_file():
            warnings.append(f"Missing retrieval file: {path}")
            continue

        for row in iter_jsonl(path):
            audio_id = str(row.get("query_audio_id") or row.get("audio_id") or "").strip()
            if not audio_id:
                continue
            record = by_audio.setdefault(
                audio_id,
                {
                    "dataset": dataset,
                    "dataset_source_dir": str(retrieval_dir),
                    "audio_id": audio_id,
                    "file_name": row.get("query_file_name") or row.get("file_name"),
                    "source_caption": row.get("query_text") or row.get("source_caption"),
                    "ranks": {},
                    "hit_at5": {},
                },
            )
            record["ranks"][model] = row.get("rank")
            record["hit_at5"][model] = rank_is_hit_at5(row)

    rows: list[dict[str, Any]] = []
    for record in by_audio.values():
        if any(model not in record["hit_at5"] for model in MODELS):
            continue
        hits = [model for model in MODELS if record["hit_at5"][model]]
        if len(hits) == 0:
            category = "all_wrong"
        elif len(hits) == len(MODELS):
            category = "all_correct"
        elif len(hits) == 1:
            category = f"only_{hits[0]}"
        else:
            continue

        record["category"] = category
        record["category_label"] = CATEGORY_LABELS[category]
        rows.append(record)

    rows.sort(key=lambda item: (CATEGORY_ORDER.index(item["category"]), item["audio_id"]))
    return rows, warnings


def summarize(rows: list[dict[str, Any]], warnings: list[str]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        counts[row["dataset"]][row["category"]] += 1
    return {
        "metric": "source caption text-to-audio Recall@5",
        "models": list(MODELS),
        "model_labels": MODEL_LABELS,
        "requested_per_dataset_per_category": 3,
        "categories": CATEGORY_LABELS,
        "counts": {dataset: dict(category_counts) for dataset, category_counts in counts.items()},
        "warnings": warnings,
    }


def write_outputs(rows: list[dict[str, Any]], warnings: list[str]) -> None:
    output_dir = PROJECT_ROOT / "results" / "source_caption_recall5_cases"
    output_dir.mkdir(parents=True, exist_ok=True)

    selected: list[dict[str, Any]] = []
    for dataset in DATASETS:
        dataset_rows = [row for row in rows if row["dataset"] == dataset]
        for category in CATEGORY_ORDER:
            selected.extend([row for row in dataset_rows if row["category"] == category][:3])

    csv_path = output_dir / "source_caption_recall5_cases.csv"
    fieldnames = [
        "dataset",
        "category",
        "category_label",
        "audio_id",
        "file_name",
        "source_caption",
        "m2d_hit_at5",
        "msclap_hit_at5",
        "mga_hit_at5",
        "laion_hit_at5",
        "m2d_rank",
        "msclap_rank",
        "mga_rank",
        "laion_rank",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow(
                {
                    "dataset": row["dataset"],
                    "category": row["category"],
                    "category_label": row["category_label"],
                    "audio_id": row["audio_id"],
                    "file_name": row["file_name"],
                    "source_caption": row["source_caption"],
                    "m2d_hit_at5": row["hit_at5"]["m2d"],
                    "msclap_hit_at5": row["hit_at5"]["msclap"],
                    "mga_hit_at5": row["hit_at5"]["mga"],
                    "laion_hit_at5": row["hit_at5"]["laion"],
                    "m2d_rank": row["ranks"]["m2d"],
                    "msclap_rank": row["ranks"]["msclap"],
                    "mga_rank": row["ranks"]["mga"],
                    "laion_rank": row["ranks"]["laion"],
                }
            )

    json_path = output_dir / "source_caption_recall5_cases.json"
    payload = {
        "summary": summarize(rows, warnings),
        "selected": selected,
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary_path = output_dir / "source_caption_recall5_summary.json"
    summary_path.write_text(json.dumps(payload["summary"], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(selected)} selected rows -> {csv_path}")
    print(f"Wrote JSON -> {json_path}")
    print(f"Wrote summary -> {summary_path}")
    for warning in warnings:
        print(f"WARNING: {warning}")


def main() -> None:
    all_rows: list[dict[str, Any]] = []
    all_warnings: list[str] = []
    for dataset in DATASETS:
        rows, warnings = load_dataset_rows(dataset)
        all_rows.extend(rows)
        all_warnings.extend(warnings)
    write_outputs(all_rows, all_warnings)


if __name__ == "__main__":
    main()
