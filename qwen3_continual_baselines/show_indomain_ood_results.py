"""Print one compact comparison row per completed independent-domain arm."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from run_continual_baselines import MODELS


TASKS = (
    "math500_holdout", "amc23", "aime25",
    "mmlu_holdout", "arc_challenge", "gpqa_diamond",
    "medqa_test", "medmcqa_validation", "pubmedqa_labeled",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-key",
        choices=("all",) + tuple(MODELS),
        default="all",
        help="show only one backbone size",
    )
    parser.add_argument(
        "--complete-only",
        action="store_true",
        help="omit arms that do not yet contain all nine benchmark results",
    )
    parser.add_argument(
        "--show-arm",
        action="store_true",
        help="include the unique arm/run-tag identifier",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    grouped = {}
    for path in sorted(Path.cwd().glob("results_idood_*.json")):
        try:
            rows = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        for row in rows:
            if (
                args.model_key != "all"
                and row.get("model") != MODELS[args.model_key]
            ):
                continue
            grouped.setdefault(row["arm"], {})[row["task"]] = row

    if not grouped:
        raise SystemExit("No results_idood_*.json files found")

    labels = {
        "math500_holdout": "math500",
        "amc23": "amc23",
        "aime25": "aime25",
        "mmlu_holdout": "mmlu",
        "arc_challenge": "arc",
        "gpqa_diamond": "gpqa",
        "medqa_test": "medqa",
        "medmcqa_validation": "medmcqa",
        "pubmedqa_labeled": "pubmedqa",
    }
    if args.complete_only:
        grouped = {
            arm: rows for arm, rows in grouped.items()
            if all(task in rows for task in TASKS)
        }
    if not grouped:
        raise SystemExit("No matching complete result arms found")

    header = ["method", "trained_on", "in_avg", "out_avg"] + [labels[t] for t in TASKS]
    if args.show_arm:
        header.insert(0, "arm")
    print(" ".join(f"{value:>12}" for value in header))
    print("-" * (13 * len(header)))
    for arm, task_rows in sorted(grouped.items()):
        sample = next(iter(task_rows.values()))
        source = sample["source_domain"]
        in_values = [row["accuracy"] for row in task_rows.values()
                     if row["scope"] == "in_domain"]
        out_values = [row["accuracy"] for row in task_rows.values()
                      if row["scope"] == "out_of_domain"]
        cells = [
            sample["method"],
            source,
            f"{mean(in_values):.1%}" if in_values else "-",
            f"{mean(out_values):.1%}" if out_values else "-",
        ]
        if args.show_arm:
            cells.insert(0, arm)
        cells.extend(
            f"{task_rows[task]['accuracy']:.1%}" if task in task_rows else "-"
            for task in TASKS
        )
        print(" ".join(f"{value:>12}" for value in cells))
    print("\nin_avg/out_avg are unweighted macro averages across benchmarks.")


if __name__ == "__main__":
    main()
