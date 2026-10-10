"""Print a compact comparison table from completed result JSON files."""
from __future__ import annotations

import argparse
from pathlib import Path
import glob
import json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("files", nargs="*", default=["results_*.json"])
    parser.add_argument("--visit", type=int, default=None)
    args = parser.parse_args()

    paths = []
    for pattern in args.files:
        matches = glob.glob(pattern)
        paths.extend(Path(match) for match in matches)
    paths = sorted(set(paths))
    if not paths:
        raise SystemExit("no result files found")

    rows = []
    for path in paths:
        rows.extend(json.loads(path.read_text(encoding="utf-8")))
    arms = sorted({row["arm"] for row in rows})
    chosen_visit = {}
    for arm in arms:
        visits = [row["visit"] for row in rows if row["arm"] == arm]
        chosen_visit[arm] = args.visit if args.visit is not None else max(visits)
    tasks = sorted(
        {
            row["task"]
            for row in rows
            if row["visit"] == chosen_visit[row["arm"]]
        }
    )
    values = {
        (row["arm"], row["task"]): row["accuracy"]
        for row in rows
        if row["visit"] == chosen_visit[row["arm"]]
    }

    header = ["arm", "visit"] + tasks
    table = [header]
    for arm in arms:
        table.append(
            [arm, str(chosen_visit[arm])]
            + [
                f"{100 * values[(arm, task)]:.1f}%"
                if (arm, task) in values
                else "-"
                for task in tasks
            ]
        )
    widths = [max(len(row[column]) for row in table) for column in range(len(header))]
    for index, row in enumerate(table):
        print("  ".join(value.ljust(widths[column]) for column, value in enumerate(row)))
        if index == 0:
            print("  ".join("-" * width for width in widths))


if __name__ == "__main__":
    main()
