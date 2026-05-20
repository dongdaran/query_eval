# EQ Generation Pipeline
[fix] for wjm test

LLM-based toolkit for generating `EQ` (Extended Query) data for audio retrieval from an **AudioCaps-style caption CSV** or **MeCAT JSON captions**. Outputs are validated and saved as JSONL.

## Installation

Requirements:

- Python `>=3.9`
- [uv](https://docs.astral.sh/uv/)
- OpenAI API key for generation

Clone and install:

```bash
git clone <your-fork-or-upstream-url>
cd UIQ
uv sync --frozen
```

To use the Hugging Face helper script below, also install the optional `datasets` extra:

```bash
uv sync --frozen --extra hf
```

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your_api_key_here
```

For OpenRouter, use its OpenAI-compatible endpoint:

```env
OPENROUTER_API_KEY=your_openrouter_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
```

Then run with the OpenRouter config:

```bash
uv run python scripts/makeeq.py --config config_openrouter.yaml --dataset macs ...
```

`scripts/makeeq.py` loads `.env` automatically. `OPENROUTER_BASE_URL` is optional when only `OPENROUTER_API_KEY` is set.

Default model settings live in [`config.yaml`](config.yaml):

```yaml
model:
  source_model: gpt-5.4-mini
  regen_model: gpt-5.4-mini
  backend: gpt
  temperature: 0.7
  batch_size: 10
  max_tokens: 100
```

## Data Preparation

Use [`scripts/prepare_data.py`](scripts/prepare_data.py) to create input directories and download supported metadata:
- `clotho`: downloads offical `*_evaluation`(3 datasets) from [clotho](https://zenodo.org/records/3490684)

- `audiocaps`: downloads official `train.csv`, `val.csv`, and `test.csv` into `input/audiocaps/` from [cdjkim/audiocaps](https://github.com/cdjkim/audiocaps/tree/master/dataset)
- `mecat`: creates `input/mecat/json_files/` for manual placement

### MeCAT 파일 준비

MeCAT은 자동 다운로드를 지원하지 않습니다. GitHub에 올려둔 ZIP 파일을 받아 `input/mecat/` 아래에 두고 압축을 해제하면 됩니다.

예시:

```bash
mkdir -p input/mecat

# GitHub에서 받은 ZIP들을 input/mecat/ 아래에 둔 뒤
mv ~/Downloads/json_files.zip input/mecat/json_files.zip
mv ~/Downloads/flac_files.zip input/mecat/flac_files.zip

# unzip이 있으면
unzip input/mecat/json_files.zip -d input/mecat/
unzip input/mecat/flac_files.zip -d input/mecat/

# unzip이 없는 환경이면 python으로
python3 -c "import zipfile; zipfile.ZipFile('input/mecat/json_files.zip').extractall('input/mecat/')"
python3 -c "import zipfile; zipfile.ZipFile('input/mecat/flac_files.zip').extractall('input/mecat/')"
```

압축을 `input/mecat/` 바로 아래에 풀었다면, `.json`과 `.flac`를 각각 폴더로 분리합니다:

```bash
mkdir -p input/mecat/json_files input/mecat/flac_files
find input/mecat -maxdepth 1 -type f -name '*.json' -exec mv -t input/mecat/json_files {} +
find input/mecat -maxdepth 1 -type f -name '*.flac' -exec mv -t input/mecat/flac_files {} +
```

정리 후에는 아래 구조가 되면 됩니다:

```text
input/mecat/
├── flac_files.zip
├── flac_files/
│   ├── <audio_id>.flac
│   ├── <audio_id>.flac
│   └── ...
├── json_files.zip
└── json_files/
    ├── <audio_id>.json
    ├── <audio_id>.json
    └── ...
```

`makeeq.py`는 `input/mecat/json_files/` 안의 각 JSON에서 `short` 필드를 읽어 사용합니다.

### AudioCaps via Hugging Face (`datasets`)

For **official** train/val/test caption CSVs from the paper repo, use `prepare_data.py --dataset audiocaps` (no `datasets` install required).

The snippet below is **[Hugging Face Hub](https://huggingface.co/datasets)** + the [`datasets`](https://huggingface.co/docs/datasets) library. The repo id **`d0rj/audiocaps`** is a mirror of **AudioCaps**, not Clotho. (Clotho is a different dataset; use `prepare_data.py --dataset clotho` or another Hub id for Clotho.)

After `uv sync --extra hf`, export a split to the same CSV columns `makeeq` expects (`youtube_id`, `start_time`, `caption`):

```bash
./scripts/download_audiocaps_hf.py --split test --output input/audiocaps/test.csv
```

Splits on the Hub are named `train`, `validation`, and `test`.

### One row per clip (merged captions)

Official CSVs have several caption lines per `(youtube_id, start_time)`. To collapse them into a single row (first `audiocap_id`, captions joined with ` | ` by default):

```bash
# Defaults: input/audiocaps/test.csv -> input/audiocaps/test_merged.csv
./scripts/merge_audiocaps_captions.py

./scripts/merge_audiocaps_captions.py \
  --input input/audiocaps/val.csv \
  --output input/audiocaps/val_merged.csv
```

Use `--separator` to change the join string. Note: `makeeq` already groups by clip when reading multi-row CSVs; merging is for tools that need one row per clip.

## EQ Query Types

The pipeline generates six variants per clip:

- `key_phrase`
- `statement`
- `question`
- `command`
- `indirect`
- `full_caption`

- **`full_caption`**: prompted with the **full caption list** for the clip; `original_captions` in JSONL is that full list.
- **`key_phrase`, `statement`, `question`, `command`, `indirect`**: prompted with the **middle caption only** (index `len // 2`); `original_captions` is a one-element list with that string. Metadata adds `eq_reference`, `middle_caption_index`, and `full_caption_count`.
- **MeCAT**: the loader uses the JSON `short` field as the caption list. That means `full_caption` uses all `short` captions, and every other query type uses only the middle `short` caption.

## Generate EQ

Use [`scripts/makeeq.py`](scripts/makeeq.py):

```bash
./scripts/makeeq.py \
  --dataset audiocaps \
  --captions-csv input/audiocaps/test.csv \
  --output-dir results/eq/audiocaps_test \
  --split test
```

MeCAT example:

```bash
./scripts/makeeq.py \
  --dataset mecat \
  --captions-path input/mecat/json_files \
  --output-dir results/eq/mecat_default \
  --split default
```

Optional arguments:

- `--dataset`: original dataset source to convert
- `--split`: split label stored in metadata, default `test`
- `--num-queries`: cap number of clips (order after grouping)
- `--config`: custom config file

Generated files:

- `eq_key_phrase.jsonl`
- `eq_statement.jsonl`
- `eq_question.jsonl`
- `eq_command.jsonl`
- `eq_indirect.jsonl`
- `eq_full_caption.jsonl`

Validation logs are written to `eq_validation.log`.

### Merge EQ types per clip (one file)

After generation, to combine the six `eq_*.jsonl` files into **one record per `audio_id`** with `original_captions` and all six strings under `generated_queries`:

```bash
./scripts/merge_eq_by_clip.py --input-dir results/eq/test_sample5
# writes results/eq/test_sample5/eq_by_clip.jsonl (pretty-printed blocks; use --compact for strict one-line JSONL)
```

## Output Schema

Each JSONL line follows this shape:

```json
{
  "audio_id": "...",
  "dataset": "...",
  "dataset_slug": "...",
  "query_type": "...",
  "generated_query": "...",
  "original_captions": ["..."],
  "metadata": {},
  "source_model": "gpt-5.4-mini",
  "regen_model": "gpt-5.4-mini"
}
```

## Text-to-Audio Retrieval Evaluation

This evaluation uses a dataset caption JSONL and audio files to build audio/text embeddings for each model, then computes text-to-audio retrieval performance. The scripts default to `clotho`, and the same workflow can be reused for another dataset such as `mecats` by passing `--dataset mecats`.

This workflow reads the project `config*.yaml` files. Model paths, model IDs, enabled models, device, and default batch size come from `--config` (default: `config.yaml`). Dataset paths can come from the config when present, or from `--dataset` and explicit path arguments.

### 1. Prepare inputs

Default input paths for a dataset named `{dataset}`:

- Caption JSONL: `input/captions/{dataset}/eq_by_clip.jsonl`
- Audio directory: `input/{dataset}/audio`

The caption JSONL should contain `audio_id`, `file_name`, the original caption, and generated query text. If `file_name` is empty, rows are merged by `audio_id`.

### 2. Generate and merge embeddings

Generate source text embeddings, generated text embeddings, and audio embeddings, then merge them into one `merged.jsonl` per model.

```bash
uv run python scripts/eval_pipeline.py
```

To run with one of the per-model configs:

```bash
uv run python scripts/eval_pipeline.py --config config_mga.yaml
```

For MECATS, pass the dataset name:

```bash
uv run python scripts/eval_pipeline.py --config config.yaml --dataset mecats
```

For a quick check on a subset of models, use `--models` and `--limit`:

```bash
uv run python scripts/eval_pipeline.py --config config.yaml --dataset mecats --models msclap laion --limit 100
```

Main output paths:

- Audio embedding: `results/audioEmb/{dataset}/{model}/emb.jsonl`
- Text embedding: `results/testEmb/{dataset}/{model}/source_emb.jsonl`, `results/testEmb/{dataset}/{model}/generated_emb.jsonl`
- Merged embedding: `results/mergedEmb/{dataset}/{model}/merged.jsonl`

### 3. Compute retrieval metrics

Compute text-to-audio Recall@K from the merged embeddings.

```bash
uv run python scripts/eval_text_to_audio_retrieval.py
```

By default, the script evaluates all query pools: `original`, `full_caption`, `statement`, `command`, `key_phrase`, `indirect`, and `question`. It saves Recall@1/5/10 for each pool.

Main output paths:

- Metric summary: `results/retrieval/{dataset}/text_to_audio_recall.json`
- Per-query retrieval results: `results/retrieval/{dataset}/{model}_{pool}_retrieval.jsonl`

To evaluate only selected models or query pools:

```bash
uv run python scripts/eval_text_to_audio_retrieval.py \
  --config config.yaml \
  --dataset mecats \
  --models msclap laion mga m2d \
  --pools full_caption statement command \
  --ks 1 5 10
```

### 4. Generate a recall bar plot

Create a model-wise Recall@K bar plot from the metric JSON.

```bash
uv run python scripts/eval_plot_text_to_audio_recall.py
```

Default output:

```text
results/retrieval/{dataset}/text_to_audio_recall_bar.png
```

To plot another query pool or change the dataset label in the chart title:

```bash
uv run python scripts/eval_plot_text_to_audio_recall.py \
  --config config.yaml \
  --dataset mecats \
  --pool statement \
  --dataset-label MECATS
```

### 5. Interpret results

- `R@1`: the fraction of queries where the correct audio is ranked first
- `R@5`: the fraction of queries where the correct audio appears in the top 5
- `R@10`: the fraction of queries where the correct audio appears in the top 10

The `original` pool uses the original caption as the query. The other pools use different generated query text types. Comparing Recall across generated query pools helps identify which text formats are easier or harder for each model in audio retrieval.

## Repository Layout

```text
repo/
├── config.yaml
├── eq_generation/
├── scripts/
│   ├── prepare_data.py
│   ├── download_audiocaps_hf.py
│   ├── merge_audiocaps_captions.py
│   ├── merge_eq_by_clip.py
│   └── makeeq.py
└── results/
```
