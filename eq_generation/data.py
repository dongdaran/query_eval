from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


def load_audiocaps(csv_path: str, split: str = "test") -> list[dict]:
    data: dict[str, dict] = {}
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            youtube_id = row.get("youtube_id", "").strip()
            start_time = row.get("start_time", "").strip()
            caption = row.get("caption", "").strip()
            if not youtube_id or not start_time or not caption:
                continue

            try:
                start_time_float = float(start_time)
                start_time_key = int(start_time_float)
            except ValueError:
                start_time_float = start_time
                start_time_key = start_time

            audio_id = f"{youtube_id}_{start_time_key}"
            record = data.setdefault(
                audio_id,
                {
                    "audio_id": audio_id,
                    "dataset": "audiocaps",
                    "dataset_slug": f"audiocaps_{split}",
                    "original_captions": [],
                    "metadata": {
                        "split": split,
                        "youtube_id": youtube_id,
                        "start_time": start_time_float,
                    },
                },
            )
            record["original_captions"].append(caption)

    return list(data.values())


def load_clotho(csv_path: str, split: str = "evaluation") -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            file_name = row.get("file_name", "").strip()
            if not file_name:
                continue

            original_captions = []
            for caption_index in range(1, 6):
                caption = row.get(f"caption_{caption_index}", "").strip()
                if caption:
                    original_captions.append(caption)

            if not original_captions:
                continue

            audio_id = Path(file_name).stem
            data.append(
                {
                    "audio_id": audio_id,
                    "dataset": "clotho",
                    "dataset_slug": f"clotho_{split}",
                    "original_captions": original_captions,
                    "metadata": {
                        "split": split,
                        "file_name": file_name,
                    },
                }
            )

    return data


def _mecat_record_from_payload(
    payload: dict[str, Any],
    split: str,
    *,
    fallback_audio_id: str | None = None,
    json_file: str | None = None,
) -> dict[str, Any] | None:
    caption_fields = (payload.get("short"), payload.get("original_captions"), payload.get("captions"))
    caption_set = next((captions for captions in caption_fields if isinstance(captions, list)), None)
    if caption_set is None:
        return None

    original_captions = [str(caption).strip() for caption in caption_set if str(caption).strip()]
    if not original_captions:
        return None

    audio_id = str(payload.get("audio_id") or fallback_audio_id or "").strip()
    if not audio_id:
        file_name = str(payload.get("file_name") or payload.get("json_file") or "").strip()
        audio_id = Path(file_name).stem
    if not audio_id:
        return None

    metadata = {
        "split": split,
    }
    if json_file:
        metadata["json_file"] = json_file

    for key in ("domain", "file_name", "json_file"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            metadata[key] = value.strip()

    return {
        "audio_id": audio_id,
        "dataset": "mecat",
        "dataset_slug": f"mecat_{split}",
        "original_captions": original_captions,
        "metadata": metadata,
    }


def load_mecat(json_path_or_dir: str, split: str = "default") -> list[dict[str, Any]]:
    data: list[dict[str, Any]] = []
    base_path = Path(json_path_or_dir)

    if base_path.is_file() and base_path.suffix == ".jsonl":
        with base_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    payload = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(payload, dict):
                    continue

                record = _mecat_record_from_payload(payload, split)
                if record is not None:
                    data.append(record)
        return data

    json_paths = [base_path] if base_path.is_file() else sorted(base_path.glob("*.json"))
    for json_path in json_paths:
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if not isinstance(payload, dict):
            continue

        record = _mecat_record_from_payload(
            payload,
            split,
            fallback_audio_id=json_path.stem,
            json_file=json_path.name,
        )
        if record is not None:
            data.append(record)

    return data


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
