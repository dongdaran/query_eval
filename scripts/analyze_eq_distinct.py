#!/usr/bin/env -S uv run python
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator


QUERY_TYPE_ORDER = [
    "key_phrase",
    "statement",
    "question",
    "command",
    "indirect",
    "full_caption",
]
WORD_RE = re.compile(r"[A-Za-z0-9]+(?:['’][A-Za-z0-9]+)?")


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


def tokens(text: str) -> list[str]:
    return WORD_RE.findall(text.lower())


def ngrams(words: list[str], n: int) -> list[tuple[str, ...]]:
    if n <= 0 or len(words) < n:
        return []
    return [tuple(words[idx : idx + n]) for idx in range(len(words) - n + 1)]


def query_type_key(query_type: str) -> tuple[int, str]:
    try:
        return (QUERY_TYPE_ORDER.index(query_type), query_type)
    except ValueError:
        return (len(QUERY_TYPE_ORDER), query_type)


def distinct_ratio(items: list[tuple[str, ...]]) -> float | None:
    if not items:
        return None
    return len(set(items)) / len(items)


def parse_range_run(path: Path, input_root: Path) -> tuple[str, str, str]:
    rel = path.relative_to(input_root)
    if len(rel.parts) < 4:
        raise ValueError(f"Expected <range>/<run>/<dataset>/eq_*.jsonl under {input_root}: {path}")
    return rel.parts[0], rel.parts[1], rel.parts[2]


def load_queries(root: Path) -> dict[tuple[str, str], list[str]]:
    grouped: dict[tuple[str, str], list[str]] = defaultdict(list)
    for path in sorted(root.glob("*/eq_*.jsonl")):
        fallback_query_type = path.stem.removeprefix("eq_")
        for obj in iter_json_objects(path):
            query = str(obj.get("generated_query", "")).strip()
            if not query:
                continue
            dataset_slug = str(obj.get("dataset_slug") or path.parent.name)
            query_type = str(obj.get("query_type") or fallback_query_type)
            grouped[(dataset_slug, query_type)].append(query)
            grouped[("ALL", query_type)].append(query)
    return grouped


def load_prompt_range_queries(root: Path) -> dict[tuple[str, str, str, str], list[str]]:
    grouped: dict[tuple[str, str, str, str], list[str]] = defaultdict(list)
    for path in sorted(root.glob("range_*_*/run_*/**/eq_*.jsonl")):
        range_name, run_name, dataset_dir = parse_range_run(path, root)
        fallback_query_type = path.stem.removeprefix("eq_")
        for obj in iter_json_objects(path):
            query = str(obj.get("generated_query", "")).strip()
            if not query:
                continue
            dataset_slug = str(obj.get("dataset_slug") or dataset_dir)
            query_type = str(obj.get("query_type") or fallback_query_type)
            grouped[(range_name, run_name, dataset_slug, query_type)].append(query)
            grouped[(range_name, run_name, "ALL", query_type)].append(query)
            grouped[(range_name, run_name, dataset_slug, "ALL_QUERY_TYPES")].append(query)
            grouped[(range_name, run_name, "ALL", "ALL_QUERY_TYPES")].append(query)
    return grouped


def mean(values: list[float]) -> float:
    return sum(values) / len(values)


def sample_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    avg = mean(values)
    variance = sum((value - avg) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def format_ratio(value: float | None) -> str:
    if value is None:
        return "NA"
    return f"{value:.4f}"


def format_float(value: float) -> str:
    return f"{value:.6f}"


def range_key(range_name: str) -> tuple[int, int, str]:
    match = re.fullmatch(r"range_(\d+)_(\d+)", range_name)
    if not match:
        return (0, 0, range_name)
    return (int(match.group(1)), int(match.group(2)), range_name)


def compute_distinct_rows(
    grouped: dict[tuple[str, str, str, str], list[str]],
    ns: list[int],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for (range_name, run_name, dataset_slug, query_type), queries in grouped.items():
        tokenized = [tokens(query) for query in queries]
        for n in ns:
            prefix_items = [tuple(words[:n]) for words in tokenized if len(words) >= n]
            global_items = [gram for words in tokenized for gram in ngrams(words, n)]
            rows.append(
                {
                    "range": range_name,
                    "run": run_name,
                    "dataset": dataset_slug,
                    "query_type": query_type,
                    "records": len(queries),
                    "n": n,
                    "prefix_unique": len(set(prefix_items)),
                    "prefix_total": len(prefix_items),
                    "prefix_distinct": distinct_ratio(prefix_items),
                    "global_unique": len(set(global_items)),
                    "global_total": len(global_items),
                    "global_distinct": distinct_ratio(global_items),
                }
            )
    rows.sort(
        key=lambda row: (
            range_key(str(row["range"])),
            str(row["run"]),
            str(row["dataset"]),
            query_type_key(str(row["query_type"])),
            int(row["n"]),
        )
    )
    return rows


def summarize_runs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[
            (
                str(row["range"]),
                str(row["dataset"]),
                str(row["query_type"]),
                int(row["n"]),
            )
        ].append(row)

    summary_rows: list[dict[str, Any]] = []
    for (range_name, dataset, query_type, n), run_rows in grouped.items():
        prefix_values = [float(row["prefix_distinct"]) for row in run_rows if row["prefix_distinct"] is not None]
        global_values = [float(row["global_distinct"]) for row in run_rows if row["global_distinct"] is not None]
        records = [int(row["records"]) for row in run_rows]
        summary_rows.append(
            {
                "range": range_name,
                "dataset": dataset,
                "query_type": query_type,
                "n": n,
                "runs": len(run_rows),
                "records_per_run_min": min(records),
                "records_per_run_max": max(records),
                "prefix_distinct_mean": mean(prefix_values) if prefix_values else None,
                "prefix_distinct_std": sample_std(prefix_values) if prefix_values else None,
                "global_distinct_mean": mean(global_values) if global_values else None,
                "global_distinct_std": sample_std(global_values) if global_values else None,
            }
        )
    summary_rows.sort(
        key=lambda row: (
            range_key(str(row["range"])),
            str(row["dataset"]),
            query_type_key(str(row["query_type"])),
            int(row["n"]),
        )
    )
    return summary_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            formatted = dict(row)
            for key, value in formatted.items():
                if isinstance(value, float):
                    formatted[key] = format_float(value)
                elif value is None:
                    formatted[key] = "NA"
            writer.writerow(formatted)


def write_top_prefixes(
    output_dir: Path,
    grouped: dict[tuple[str, str, str, str], list[str]],
    ns: list[int],
    top_k: int,
) -> Path:
    path = output_dir / "top_prefixes.csv"
    fields = [
        "range",
        "run",
        "dataset",
        "query_type",
        "n",
        "rank",
        "count",
        "share",
        "prefix",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (range_name, run_name, dataset, query_type), queries in sorted(
            grouped.items(),
            key=lambda item: (
                range_key(item[0][0]),
                item[0][1],
                item[0][2],
                query_type_key(item[0][3]),
            ),
        ):
            tokenized = [tokens(query) for query in queries]
            for n in ns:
                prefix_items = [tuple(words[:n]) for words in tokenized if len(words) >= n]
                total = len(prefix_items)
                if not total:
                    continue
                counts: dict[tuple[str, ...], int] = defaultdict(int)
                for prefix in prefix_items:
                    counts[prefix] += 1
                for rank, (prefix, count) in enumerate(
                    sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:top_k],
                    start=1,
                ):
                    writer.writerow(
                        {
                            "range": range_name,
                            "run": run_name,
                            "dataset": dataset,
                            "query_type": query_type,
                            "n": n,
                            "rank": rank,
                            "count": count,
                            "share": format_float(count / total),
                            "prefix": " ".join(prefix),
                        }
                    )
    return path


def plot_summary(
    rows: list[dict[str, Any]],
    output_path: Path,
    dataset: str,
    query_type: str,
    metric: str,
    show_std: bool,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(output_path.parent / ".matplotlib"))

    import matplotlib.pyplot as plt

    filtered = [
        row
        for row in rows
        if row["dataset"] == dataset and row["query_type"] == query_type and row[metric] is not None
    ]
    if not filtered:
        raise SystemExit(f"No rows to plot for dataset={dataset}, query_type={query_type}, metric={metric}")

    by_range: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in filtered:
        by_range[str(row["range"])].append(row)

    plt.figure(figsize=(9, 5.5))
    std_metric = metric.replace("_mean", "_std")
    for range_name in sorted(by_range, key=range_key):
        range_rows = sorted(by_range[range_name], key=lambda row: int(row["n"]))
        xs = [int(row["n"]) for row in range_rows]
        ys = [float(row[metric]) for row in range_rows]
        line = plt.plot(
            xs,
            ys,
            marker="o",
            linewidth=2,
            label=range_name,
        )[0]
        if show_std:
            stds = [float(row[std_metric]) if row[std_metric] is not None else 0.0 for row in range_rows]
            plt.fill_between(
                xs,
                [max(0.0, y - std) for y, std in zip(ys, stds)],
                [min(1.0, y + std) for y, std in zip(ys, stds)],
                color=line.get_color(),
                alpha=0.14,
                linewidth=0,
            )

    plt.xlabel("distinct-n n")
    plt.ylabel(metric + (" (+/- 1 std)" if show_std else ""))
    plt.title(f"{metric} by prompt range ({dataset}, {query_type})")
    plt.grid(True, alpha=0.25)
    plt.legend(title="range")
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def run_prompt_range_analysis(args: argparse.Namespace, ns: list[int]) -> None:
    grouped = load_prompt_range_queries(args.input_root)
    if not grouped:
        raise SystemExit(f"No EQ prompt-range JSONL records found under {args.input_root}")

    output_dir = args.output_dir or (args.input_root / "distinct_n")
    output_dir.mkdir(parents=True, exist_ok=True)

    by_run_rows = compute_distinct_rows(grouped, ns)
    summary_rows = summarize_runs(by_run_rows)

    by_run_path = output_dir / "distinct_n_by_run.csv"
    summary_path = output_dir / "distinct_n_summary.csv"
    plot_path = output_dir / f"{args.plot_metric}_{args.plot_dataset}_{args.plot_query_type}.png"
    top_prefixes_path = write_top_prefixes(output_dir, grouped, ns, args.top_prefixes)

    write_csv(
        by_run_path,
        by_run_rows,
        [
            "range",
            "run",
            "dataset",
            "query_type",
            "records",
            "n",
            "prefix_unique",
            "prefix_total",
            "prefix_distinct",
            "global_unique",
            "global_total",
            "global_distinct",
        ],
    )
    write_csv(
        summary_path,
        summary_rows,
        [
            "range",
            "dataset",
            "query_type",
            "n",
            "runs",
            "records_per_run_min",
            "records_per_run_max",
            "prefix_distinct_mean",
            "prefix_distinct_std",
            "global_distinct_mean",
            "global_distinct_std",
        ],
    )
    plot_summary(
        summary_rows,
        plot_path,
        args.plot_dataset,
        args.plot_query_type,
        args.plot_metric,
        args.plot_std,
    )
    query_type_plot_paths = []
    if args.plot_each_query_type:
        for query_type in QUERY_TYPE_ORDER:
            query_type_plot_path = output_dir / f"{args.plot_metric}_{args.plot_dataset}_{query_type}.png"
            plot_summary(
                summary_rows,
                query_type_plot_path,
                args.plot_dataset,
                query_type,
                args.plot_metric,
                args.plot_std,
            )
            query_type_plot_paths.append(query_type_plot_path)

    print(f"Wrote {by_run_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {top_prefixes_path}")
    print(f"Wrote {plot_path}")
    for query_type_plot_path in query_type_plot_paths:
        print(f"Wrote {query_type_plot_path}")
    print(f"\n# {args.plot_dataset} / {args.plot_query_type} / {args.plot_metric}")
    print("range\tn\truns\tmean\tstd")
    std_metric = args.plot_metric.replace("_mean", "_std")
    for row in summary_rows:
        if row["dataset"] != args.plot_dataset or row["query_type"] != args.plot_query_type:
            continue
        print(
            f"{row['range']}\t{row['n']}\t{row['runs']}\t"
            f"{format_ratio(row[args.plot_metric])}\t{format_ratio(row[std_metric])}"
        )


def run_single_root_analysis(args: argparse.Namespace, ns: list[int]) -> None:
    grouped = load_queries(args.input_root)
    if not grouped:
        raise SystemExit(f"No EQ JSONL records found under {args.input_root}")

    print(f"# distinct-n: {args.input_root}")
    print(
        "scope\tdataset\tquery_type\trecords\tn\t"
        "prefix_unique\tprefix_total\tprefix_distinct\t"
        "global_unique\tglobal_total\tglobal_distinct"
    )

    rows = []
    for (dataset_slug, query_type), queries in grouped.items():
        scope = "all" if dataset_slug == "ALL" else "per_dataset"
        if args.scope == "all" and scope != "all":
            continue
        if args.scope == "per-dataset" and scope != "per_dataset":
            continue

        tokenized = [tokens(query) for query in queries]
        for n in ns:
            prefix_items = [tuple(words[:n]) for words in tokenized if len(words) >= n]
            global_items = [gram for words in tokenized for gram in ngrams(words, n)]
            rows.append(
                (
                    scope,
                    dataset_slug,
                    query_type,
                    len(queries),
                    n,
                    len(set(prefix_items)),
                    len(prefix_items),
                    distinct_ratio(prefix_items),
                    len(set(global_items)),
                    len(global_items),
                    distinct_ratio(global_items),
                )
            )

    for row in sorted(
        rows,
        key=lambda item: (
            0 if item[0] == "all" else 1,
            item[1],
            query_type_key(item[2]),
            item[4],
        ),
    ):
        print(
            f"{row[0]}\t{row[1]}\t{row[2]}\t{row[3]}\t{row[4]}\t"
            f"{row[5]}\t{row[6]}\t{format_ratio(row[7])}\t"
            f"{row[8]}\t{row[9]}\t{format_ratio(row[10])}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute prefix/global distinct-n for EQ JSONL outputs.")
    parser.add_argument("--input-root", type=Path, default=Path("results/eq_top30_prompt_ranges"))
    parser.add_argument(
        "--mode",
        choices=["auto", "single-root", "prompt-ranges"],
        default="auto",
        help="single-root prints the legacy TSV; prompt-ranges writes run averages and a plot.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for prompt-ranges mode. Default: <input-root>/distinct_n",
    )
    parser.add_argument("--n", default="1,3,5,7,10", help="Comma-separated n values.")
    parser.add_argument(
        "--scope",
        choices=["all", "per-dataset", "both"],
        default="both",
        help="Report all-dataset aggregate, per-dataset values, or both.",
    )
    parser.add_argument("--plot-dataset", default="ALL")
    parser.add_argument("--plot-query-type", default="ALL_QUERY_TYPES")
    parser.add_argument(
        "--plot-metric",
        default="global_distinct_mean",
        choices=["prefix_distinct_mean", "global_distinct_mean"],
    )
    parser.add_argument(
        "--plot-each-query-type",
        action="store_true",
        help="Also write one plot per EQ query type for prompt-ranges mode.",
    )
    parser.add_argument(
        "--no-plot-std",
        dest="plot_std",
        action="store_false",
        help="Do not draw +/- 1 std shading across runs on plots.",
    )
    parser.add_argument(
        "--top-prefixes",
        type=int,
        default=20,
        help="Number of most frequent prefixes to write per group and n.",
    )
    parser.set_defaults(plot_std=True)
    args = parser.parse_args()

    ns = [int(item.strip()) for item in args.n.split(",") if item.strip()]
    if not ns:
        raise SystemExit("--n must include at least one integer")

    mode = args.mode
    if mode == "auto":
        mode = "prompt-ranges" if any(args.input_root.glob("range_*_*/run_*")) else "single-root"

    if mode == "prompt-ranges":
        run_prompt_range_analysis(args, ns)
    else:
        run_single_root_analysis(args, ns)


if __name__ == "__main__":
    main()
