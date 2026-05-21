#!/usr/bin/env bash
set -euo pipefail

CONFIG="${CONFIG-config_openrouter.yaml}"
OUT_ROOT="${OUT_ROOT-results/eq_validation}"
DATASETS="${DATASETS-audiocaps_val clotho_validation macs_test_remaining macs_val mecat_train_top160}"
PROMPT_RANGES="${PROMPT_RANGES-3:15}"
RUN_REPEATS="${RUN_REPEATS-1}"
START_INDEX="${START_INDEX-0}"
END_INDEX="${END_INDEX-}"
NUM_QUERIES="${NUM_QUERIES-}"
QUERY_TYPES="${QUERY_TYPES-}"
AUDIOCAPS_VAL_QUERY_TYPES="${AUDIOCAPS_VAL_QUERY_TYPES-}"
DRY_RUN="${DRY_RUN-0}"
UV_CACHE_DIR="${UV_CACHE_DIR-.uv-cache}"

usage() {
  cat <<'EOF'
Run EQ generation for the checked-in validation inputs.

Environment variables:
  CONFIG        Config YAML. Default: config_openrouter.yaml
  OUT_ROOT      Output root. Default: results/eq_validation
  DATASETS      Space-separated dataset keys to run.
                Supported: audiocaps_val clotho_validation macs_full macs_test
                           macs_test_balanced500 macs_test_remaining
                           macs_val mecat_train_top160
                Default: audiocaps_val clotho_validation macs_test_remaining
                         macs_val mecat_train_top160
  PROMPT_RANGES Space-separated MIN:MAX ranges for prompt start-word counts.
                Default: 3:15. Set to empty to disable.
  RUN_REPEATS   Repeats per prompt range. Default: 1
  START_INDEX   Inclusive start index after grouping. Default: 0
  END_INDEX     Inclusive end index after grouping. Default: unset, no end limit
  NUM_QUERIES   Optional max clips after indexing. Default: unset
  QUERY_TYPES   Optional space-separated query types, e.g. "key_phrase statement question"
  AUDIOCAPS_VAL_QUERY_TYPES
                Optional query types for audiocaps_val only. Overrides QUERY_TYPES
                for audiocaps_val when set, e.g. "command indirect full_caption".
  DRY_RUN       Print commands without running them. Default: 0

Examples:
  scripts/run_eq_validation.sh
  DATASETS="macs_val" NUM_QUERIES=10 scripts/run_eq_validation.sh
  DRY_RUN=1 DATASETS="macs_val" NUM_QUERIES=10 scripts/run_eq_validation.sh
  CONFIG=config.yaml PROMPT_RANGES="" QUERY_TYPES="key_phrase statement" scripts/run_eq_validation.sh
  AUDIOCAPS_VAL_QUERY_TYPES="command indirect full_caption" scripts/run_eq_validation.sh
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

index_label() {
  if [[ -n "${END_INDEX}" ]]; then
    printf 'idx%s_%s' "${START_INDEX}" "${END_INDEX}"
  else
    printf 'idx%s_end' "${START_INDEX}"
  fi
}

run_dataset() {
  local key="$1"
  local dataset="$2"
  local captions_path="$3"
  local split="$4"
  local out_root="$5"
  shift 5

  if [[ ! -f "${captions_path}" ]]; then
    echo "Missing input file for ${key}: ${captions_path}" >&2
    exit 1
  fi

  local limit_label
  limit_label="$(index_label)"

  local index_args=(--start-index "${START_INDEX}")
  if [[ -n "${END_INDEX}" ]]; then
    index_args+=(--end-index "${END_INDEX}")
  fi
  if [[ -n "${NUM_QUERIES}" ]]; then
    index_args+=(--num-queries "${NUM_QUERIES}")
    limit_label="${limit_label}_n${NUM_QUERIES}"
  fi

  local query_args=()
  if [[ -n "${QUERY_TYPES}" ]]; then
    # shellcheck disable=SC2206
    query_args=(--query-types ${QUERY_TYPES})
  fi

  local output_dir="${out_root}/${key}_${limit_label}"

  echo "[INFO] ${key}: ${captions_path} -> ${output_dir}"

  local cmd=(
    uv run python scripts/makeeq.py
    --dataset "${dataset}" \
    --captions-path "${captions_path}" \
    --output-dir "${output_dir}" \
    --split "${split}" \
    --config "${CONFIG}" \
    "${index_args[@]}" \
    "${query_args[@]}" \
    "$@"
  )

  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[DRY_RUN] UV_CACHE_DIR=%q' "${UV_CACHE_DIR}"
    printf ' %q' "${cmd[@]}"
    printf '\n'
    return 0
  fi

  UV_CACHE_DIR="${UV_CACHE_DIR}" "${cmd[@]}"
}

run_dataset_key() {
  local key="$1"
  local out_root="$2"
  shift 2
  local dataset_query_args=()

  if [[ "${key}" == "audiocaps_val" && -n "${AUDIOCAPS_VAL_QUERY_TYPES}" ]]; then
    # shellcheck disable=SC2206
    dataset_query_args=(--query-types ${AUDIOCAPS_VAL_QUERY_TYPES})
  fi

  case "${key}" in
    audiocaps_val)
      run_dataset "${key}" audiocaps input/audiocaps/val.csv val "${out_root}" "${dataset_query_args[@]}" "$@"
      ;;
    clotho_validation)
      run_dataset "${key}" clotho input/clotho/clotho_captions_validation.csv validation "${out_root}" "$@"
      ;;
    macs_full)
      run_dataset "${key}" macs input/MACS/MACS.yaml test "${out_root}" "$@"
      ;;
    macs_test)
      run_dataset "${key}" macs input/MACS/MACS_test.yaml test "${out_root}" "$@"
      ;;
    macs_test_balanced500)
      run_dataset "${key}" macs input/MACS/MACS_test_balanced500.yaml test "${out_root}" "$@"
      ;;
    macs_test_remaining)
      run_dataset \
        "${key}" \
        macs \
        input/MACS/MACS_test.yaml \
        test \
        "${out_root}" \
        --exclude-captions-path input/MACS/MACS_test_balanced500.yaml \
        "$@"
      ;;
    macs_val)
      run_dataset "${key}" macs input/MACS/MACS_val.yaml validation "${out_root}" "$@"
      ;;
    mecat_train_top160)
      run_dataset "${key}" mecat input/mecat/json_train/00A_train_top160.jsonl train "${out_root}" "$@"
      ;;
    *)
      echo "Unknown DATASETS entry: ${key}" >&2
      echo "Run with --help to see supported keys." >&2
      exit 2
      ;;
  esac
}

run_selected_datasets() {
  local out_root="$1"
  shift

  for dataset_key in ${DATASETS}; do
    run_dataset_key "${dataset_key}" "${out_root}" "$@"
  done
}

run_repeated_range() {
  local min="$1"
  local max="$2"

  for run_idx in $(seq 1 "${RUN_REPEATS}"); do
    run_label=$(printf "run_%02d" "${run_idx}")
    run_selected_datasets \
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
  run_selected_datasets "${OUT_ROOT}"
fi
