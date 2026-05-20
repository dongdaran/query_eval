#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG:-config.yaml}"
OUT_ROOT="${OUT_ROOT:-results/eq_renew}"
PROMPT_RANGES="${PROMPT_RANGES:-3:15}"
RUN_REPEATS="${RUN_REPEATS:-1}"
START_INDEX="${START_INDEX:-126}"
END_INDEX="${END_INDEX:-150}"
INDEX_END_LABEL="${END_INDEX:-end}"
INDEX_LABEL="idx${START_INDEX}_${INDEX_END_LABEL}"

run_dataset() {
  local dataset="$1"
  local captions_path="$2"
  local output_dir="$3"
  local split="$4"
  shift 4
  local index_args=(--start-index "${START_INDEX}")
  if [[ -n "${END_INDEX}" ]]; then
    index_args+=(--end-index "${END_INDEX}")
  fi

  UV_CACHE_DIR="${UV_CACHE_DIR:-.uv-cache}" uv run python scripts/makeeq.py \
    --dataset "${dataset}" \
    --captions-path "${captions_path}" \
    --output-dir "${output_dir}" \
    --split "${split}" \
    --config "${CONFIG}" \
    "${index_args[@]}" \
    "$@"
}

run_one_range() {
  local out_root="$1"
  shift

  run_dataset \
    audiocaps \
    input/audiocaps/test_merged.csv \
    "${out_root}/audiocaps_test_merged_${INDEX_LABEL}" \
    test \
    "$@"

  run_dataset \
    macs \
    input/MACS/MACS.yaml \
    "${out_root}/macs_test_${INDEX_LABEL}" \
    test \
    "$@"
}

run_repeated_range() {
  local min="$1"
  local max="$2"

  for run_idx in $(seq 1 "${RUN_REPEATS}"); do
    run_label=$(printf "run_%02d" "${run_idx}")
    run_one_range \
      "${OUT_ROOT}/range_${min}_${max}/${run_label}" \
      --start-word-count-min "${min}" \
      --start-word-count-max "${max}"
  done
}

if [[ -n "${PROMPT_RANGES:-}" ]]; then
  for range in ${PROMPT_RANGES}; do
    min="${range%%:*}"
    max="${range##*:}"
    if [[ "${min}" == "${range}" || -z "${min}" || -z "${max}" ]]; then
      echo "Invalid PROMPT_RANGES entry: ${range}. Use MIN:MAX, e.g. 5:10." >&2
      exit 2
    fi
    run_repeated_range "${min}" "${max}"
  done
elif [[ -n "${START_WORD_COUNT_MIN:-}" || -n "${START_WORD_COUNT_MAX:-}" ]]; then
  if [[ -z "${START_WORD_COUNT_MIN:-}" || -z "${START_WORD_COUNT_MAX:-}" ]]; then
    echo "START_WORD_COUNT_MIN and START_WORD_COUNT_MAX must be set together." >&2
    exit 2
  fi
  run_repeated_range "${START_WORD_COUNT_MIN}" "${START_WORD_COUNT_MAX}"
else
  run_one_range "${OUT_ROOT}"
fi
