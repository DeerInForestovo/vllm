#!/usr/bin/env python3
"""Write per-layer expert ratio_all_topk values sorted descending.

The input CSV is expected to contain these columns:
layer_id, expert_id, count_all_topk, count_top1, ratio_all_topk, ratio_top1

The output is a plain-text file with one line per layer, for example:
Layer 1: 0.015440, 0.003720, ...
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sort ratio_all_topk values for every layer and write them to a text file."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("expert_stats.csv"),
        help="Path to the expert statistics CSV.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("layer_ratio_all_topk_sorted.txt"),
        help="Path to the output text file.",
    )
    return parser.parse_args()


def load_layer_ratios(path: Path) -> dict[int, List[float]]:
    layers: DefaultDict[int, List[float]] = defaultdict(list)
    with path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                layer_id = int(row["layer_id"])
                ratio_all_topk = float(row["ratio_all_topk"])
            except (KeyError, ValueError):
                continue
            layers[layer_id].append(ratio_all_topk)
    return dict(layers)


def format_layer_line(layer_id: int, ratios: List[float]) -> str:
    sorted_ratios = sorted(ratios, reverse=True)
    ratio_text = ", ".join(f"{ratio:.6f}" for ratio in sorted_ratios)
    return f"Layer {layer_id}: {ratio_text}"


def main() -> None:
    args = parse_args()
    layer_ratios = load_layer_ratios(args.input)

    output_lines = [
        format_layer_line(layer_id, ratios)
        for layer_id, ratios in sorted(layer_ratios.items())
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output_lines) + ("\n" if output_lines else ""), encoding="utf-8")


if __name__ == "__main__":
    main()