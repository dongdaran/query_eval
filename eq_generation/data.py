from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


def _split_joined_captions(caption: str, separator: str = " | ") -> list[str]:
    captions = [part.strip() for part in caption.split(separator)]
    return [part for part in captions if part]


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
            record["original_captions"].extend(_split_joined_captions(caption))

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


def _first_non_empty_string(payload: dict[str, Any], keys: tuple[str, ...]) -> str:
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
        if value is not None and not isinstance(value, (dict, list, tuple, set)):
            text = str(value).strip()
            if text:
                return text
    return ""


def _extract_caption_strings(value: Any) -> list[str]:
    captions: list[str] = []
    if isinstance(value, str):
        caption = value.strip()
        if caption:
            captions.append(caption)
    elif isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                captions.extend(
                    _extract_caption_strings(
                        _first_non_empty_string(
                            item,
                            ("caption", "text", "sentence", "description", "annotated_caption"),
                        )
                    )
                )
            else:
                captions.extend(_extract_caption_strings(item))
    elif isinstance(value, dict):
        for key in ("caption", "text", "sentence", "description", "annotated_caption"):
            captions.extend(_extract_caption_strings(value.get(key)))

    return captions


def _macs_records_from_yaml_payload(payload: Any) -> list[tuple[str | None, Any]]:
    if isinstance(payload, list):
        return [(None, item) for item in payload]

    if not isinstance(payload, dict):
        return []

    for key in ("data", "records", "items", "clips", "files", "annotations"):
        value = payload.get(key)
        if isinstance(value, list):
            return [(None, item) for item in value]
        if isinstance(value, dict):
            return [(str(item_key), item_value) for item_key, item_value in value.items()]

    return [(str(item_key), item_value) for item_key, item_value in payload.items()]


def _macs_record_from_payload(
    payload: Any,
    split: str,
    *,
    fallback_audio_id: str | None = None,
) -> dict[str, Any] | None:
    if isinstance(payload, str):
        original_captions = _extract_caption_strings(payload)
        audio_id = str(fallback_audio_id or "").strip()
        metadata: dict[str, Any] = {"split": split}
    elif isinstance(payload, list):
        original_captions = _extract_caption_strings(payload)
        audio_id = str(fallback_audio_id or "").strip()
        metadata = {"split": split}
    elif isinstance(payload, dict):
        caption_keys = (
            "captions",
            "original_captions",
            "caption",
            "annotations",
            "annotated_captions",
            "sentences",
            "descriptions",
        )
        original_captions = []
        for key in caption_keys:
            original_captions.extend(_extract_caption_strings(payload.get(key)))

        audio_id = _first_non_empty_string(
            payload,
            ("audio_id", "id", "clip_id", "file_id", "sound_id", "youtube_id"),
        )
        file_name = _first_non_empty_string(
            payload,
            ("file_name", "filename", "file", "path", "audio", "audio_path", "wav", "mp3"),
        )
        if not audio_id:
            audio_id = str(fallback_audio_id or "").strip() or Path(file_name).stem

        metadata = {"split": split}
        for key, value in payload.items():
            if key in caption_keys:
                continue
            if isinstance(value, (str, int, float, bool)) or value is None:
                metadata[key] = value
        if file_name and "file_name" not in metadata:
            metadata["file_name"] = file_name
        annotations = payload.get("annotations")
        if isinstance(annotations, list):
            annotator_ids = []
            tags = []
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                annotator_id = annotation.get("annotator_id")
                if annotator_id is not None:
                    annotator_ids.append(annotator_id)
                annotation_tags = annotation.get("tags")
                if isinstance(annotation_tags, list):
                    tags.extend(str(tag).strip() for tag in annotation_tags if str(tag).strip())
            if annotator_ids:
                metadata["annotator_ids"] = annotator_ids
            if tags:
                metadata["tags"] = sorted(set(tags))
    else:
        return None

    original_captions = list(dict.fromkeys(caption for caption in original_captions if caption))
    if not audio_id or not original_captions:
        return None

    return {
        "audio_id": audio_id,
        "dataset": "macs",
        "dataset_slug": f"macs_{split}",
        "original_captions": original_captions,
        "metadata": metadata,
    }


def load_macs(yaml_path: str, split: str = "default") -> list[dict[str, Any]]:
    yaml_file = Path(yaml_path)
    with yaml_file.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}

    grouped: dict[str, dict[str, Any]] = {}
    for fallback_audio_id, record_payload in _macs_records_from_yaml_payload(payload):
        record = _macs_record_from_payload(record_payload, split, fallback_audio_id=fallback_audio_id)
        if record is None:
            continue

        existing = grouped.setdefault(record["audio_id"], record)
        if existing is record:
            continue

        seen = set(existing["original_captions"])
        for caption in record["original_captions"]:
            if caption not in seen:
                existing["original_captions"].append(caption)
                seen.add(caption)
        existing["metadata"].update(record.get("metadata", {}))

    return list(grouped.values())


def load_config(config_path: str = "config.yaml") -> dict:
    path = Path(config_path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}
