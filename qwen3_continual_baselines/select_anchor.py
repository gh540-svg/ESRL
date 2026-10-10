"""Rank ES-GRPO anchor sweeps without consulting the transfer test suite.

Primary criterion: macro accuracy over the three core validation tasks at the
final continual checkpoint. Tie-breaker: macro accuracy over those tasks across
all post-base checkpoints (core AUC). Related transfer benchmarks are deliberately
excluded so they remain available for reporting after the anchor is selected.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path


CORE_TASKS = ("math500_holdout", "mmlu_holdout", "medqa_test")


def percent(value):
    return "-" if value is None else f"{100 * value:.2f}%"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "files",
        nargs="*",
        default=["results_es_grpo_ff_qwen3_4b_n16_anchor_kl*.json"],
    )
    args = parser.parse_args()

    paths = []
    for pattern in args.files:
        paths.extend(Path(match) for match in glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("no anchor-sweep result files found")

    summaries = []
    for path in paths:
        rows = json.loads(path.read_text(encoding="utf-8"))
        arms = {row["arm"] for row in rows}
        if len(arms) != 1:
            raise SystemExit(f"{path}: expected exactly one arm, found {sorted(arms)}")
        arm = next(iter(arms))
        final_visit = max(int(row["visit"]) for row in rows)
        lookup = {
            (int(row["visit"]), row["task"]): float(row["accuracy"])
            for row in rows
        }
        final_values = [lookup.get((final_visit, task)) for task in CORE_TASKS]
        final_macro = (
            sum(final_values) / len(final_values)
            if all(value is not None for value in final_values) else None
        )
        post_base = [
            float(row["accuracy"])
            for row in rows
            if int(row["visit"]) > 0 and row["task"] in CORE_TASKS
        ]
        core_auc = sum(post_base) / len(post_base) if post_base else None

        manifest_path = path.with_name(
            path.name.replace("results_", "manifest_", 1)
        )
        anchor = {}
        if manifest_path.is_file():
            anchor = json.loads(manifest_path.read_text(encoding="utf-8")).get(
                "anchor", {}
            )
        summaries.append(
            {
                "arm": arm,
                "visit": final_visit,
                "lambda": anchor.get("kl_lambda"),
                "refresh": anchor.get("teacher_reanchor_every_prompts"),
                "sigma": anchor.get("es_perturbation_sigma"),
                "final": final_macro,
                "auc": core_auc,
                "tasks": final_values,
            }
        )

    summaries.sort(
        key=lambda row: (
            row["final"] if row["final"] is not None else -1.0,
            row["auc"] if row["auc"] is not None else -1.0,
        ),
        reverse=True,
    )
    header = (
        "rank", "lambda", "refresh", "sigma", "visit", "final_core",
        "core_auc", *CORE_TASKS, "arm",
    )
    table = [header]
    for rank, row in enumerate(summaries, 1):
        table.append(
            (
                str(rank), str(row["lambda"]), str(row["refresh"]),
                str(row["sigma"]), str(row["visit"]), percent(row["final"]),
                percent(row["auc"]), *(percent(value) for value in row["tasks"]),
                row["arm"],
            )
        )
    widths = [max(len(str(row[i])) for row in table) for i in range(len(header))]
    for index, row in enumerate(table):
        print("  ".join(str(value).ljust(widths[i]) for i, value in enumerate(row)))
        if index == 0:
            print("  ".join("-" * width for width in widths))
    print("\nSelection rule: highest final_core; core_auc breaks ties.")
    print("Transfer benchmarks were not used for selection.")


if __name__ == "__main__":
    main()
