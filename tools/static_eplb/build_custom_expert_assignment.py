#!/usr/bin/env python3
"""Build per-layer custom expert placement for vLLM EP.

This script reads expert routing statistics from a CSV file and builds a
balanced 8-way (or user-provided ep-size) assignment for each layer.

Default behavior matches the user's request:
- Use layers 3..77 (inclusive)
- Use `count_top1` as expert weight
- Partition experts into 8 groups with balanced total weights

Output JSON schema:
{
  "version": 1,
  "ep_size": 8,
  "weight_column": "count_top1",
  "layer_start": 3,
  "layer_end": 77,
  "layers": {
    "3": {
      "rank_to_experts": [[...], ...],
      "rank_weights": [123, ...],
      "total_weight": 1234
    }
  }
}
"""

from __future__ import annotations

import argparse
import csv
import heapq
import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from collections import deque


@dataclass(order=True)
class _RankBucket:
    total_weight: float
    rank: int
    experts: list[int] = field(compare=False, default_factory=list)


def _build_layer_assignment(
    expert_weights: dict[int, float],
    ep_size: int,
) -> tuple[list[list[int]], list[float], float]:
    num_experts = len(expert_weights)
    if num_experts % ep_size != 0:
        raise ValueError(f"Expert count {num_experts} is not divisible by EP size {ep_size}")

    sorted_experts = sorted(expert_weights.items(), key=lambda x: (-x[1], x[0]))
    expert_pool = deque(sorted_experts)

    buckets = [
        {"rank": i, "total_weight": 0.0, "experts": []} 
        for i in range(ep_size)
    ]

    for i in range(ep_size):
        expert_id, weight = expert_pool.popleft()
        buckets[i]["experts"].append(expert_id)
        buckets[i]["total_weight"] += weight

    num_rounds = num_experts // ep_size
    for _ in range(1, num_rounds):
        current_total = sum(b["total_weight"] for b in buckets)
        avg_weight = current_total / ep_size

        above_avg = [b for b in buckets if b["total_weight"] >= avg_weight]
        below_avg = [b for b in buckets if b["total_weight"] < avg_weight]

        above_avg.sort(key=lambda x: x["total_weight"], reverse=True)
        below_avg.sort(key=lambda x: x["total_weight"])

        for bucket in below_avg:
            expert_id, weight = expert_pool.popleft()
            bucket["experts"].append(expert_id)
            bucket["total_weight"] += weight

        for bucket in above_avg:
            expert_id, weight = expert_pool.pop()
            bucket["experts"].append(expert_id)
            bucket["total_weight"] += weight

    buckets.sort(key=lambda x: x["rank"])
    
    rank_to_experts = [sorted(b["experts"]) for b in buckets]
    rank_weights = [float(b["total_weight"]) for b in buckets]
    total_weight = float(sum(rank_weights))
    
    return rank_to_experts, rank_weights, total_weight


def _load_csv_weights(
    csv_path: Path,
    weight_column: str,
    layer_start: int,
    layer_end: int,
) -> dict[int, dict[int, float]]:
    weights_by_layer: dict[int, dict[int, float]] = defaultdict(dict)

    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"layer_id", "expert_id", weight_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        for row in reader:
            layer_id = int(row["layer_id"])
            if layer_id < layer_start or layer_id > layer_end:
                continue
            expert_id = int(row["expert_id"])
            weight = float(row[weight_column])
            weights_by_layer[layer_id][expert_id] = weight

    return dict(weights_by_layer)


def _write_summary_csv(
    summary_path: Path,
    layers_payload: dict[str, dict[str, object]],
    ep_size: int,
) -> None:
    header = ["layer_id"] + [f"gpu{i}_load_pct" for i in range(ep_size)] + [
        "total_weight",
        "imbalance_ratio",
    ]

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for layer_id_str in sorted(layers_payload, key=lambda x: int(x)):
            layer_info = layers_payload[layer_id_str]
            rank_weights = [float(x) for x in layer_info["rank_weights"]]
            total_weight = float(layer_info["total_weight"])
            
            pct_weights = [
                f"{(w / total_weight * 100):.2f}%" if total_weight > 0 else "0.00%" 
                for w in rank_weights
            ]
            
            max_w = max(rank_weights) if rank_weights else 0.0
            min_w = min(rank_weights) if rank_weights else 0.0
            imbalance = (max_w / min_w) if min_w > 0 else 1.0

            writer.writerow(
                [int(layer_id_str)]
                + pct_weights
                + [f"{total_weight:.2f}", f"{imbalance:.3f}"]
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build custom EP placement JSON from CSV stats")
    parser.add_argument("--csv", required=True, help="Input expert stats CSV path")
    parser.add_argument("--output", required=True, help="Output JSON path")
    parser.add_argument("--ep-size", type=int, default=8, help="EP world size")
    parser.add_argument("--layer-start", type=int, default=3, help="Inclusive start layer")
    parser.add_argument("--layer-end", type=int, default=77, help="Inclusive end layer")
    parser.add_argument(
        "--summary-output",
        default=None,
        help="Optional summary CSV path. Default: <output_stem>_summary.csv",
    )
    parser.add_argument(
        "--weight-column",
        default="count_all_topk",
        choices=["count_top1", "count_all_topk", "ratio_top1", "ratio_all_topk"],
        help="CSV column used as balancing weight",
    )

    args = parser.parse_args()

    csv_path = Path(args.csv)
    output_path = Path(args.output)

    if args.ep_size <= 0:
        raise ValueError("--ep-size must be > 0")

    weights_by_layer = _load_csv_weights(
        csv_path=csv_path,
        weight_column=args.weight_column,
        layer_start=args.layer_start,
        layer_end=args.layer_end,
    )

    if not weights_by_layer:
        raise ValueError("No layer data found for the requested range")

    layers_payload: dict[str, dict[str, object]] = {}

    for layer_id in sorted(weights_by_layer):
        rank_to_experts, rank_weights, total_weight = _build_layer_assignment(
            expert_weights=weights_by_layer[layer_id],
            ep_size=args.ep_size,
        )
        layers_payload[str(layer_id)] = {
            "rank_to_experts": rank_to_experts,
            "rank_weights": rank_weights,
            "total_weight": total_weight,
        }

    payload = {
        "version": 1,
        "ep_size": args.ep_size,
        "weight_column": args.weight_column,
        "layer_start": args.layer_start,
        "layer_end": args.layer_end,
        "layers": layers_payload,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=True, indent=2), encoding="utf-8")

    summary_path = (
        Path(args.summary_output)
        if args.summary_output
        else output_path.with_name(output_path.stem + "_summary.csv")
    )
    _write_summary_csv(
        summary_path=summary_path,
        layers_payload=layers_payload,
        ep_size=args.ep_size,
    )

    print(f"Wrote assignment file: {output_path}")
    print(f"Wrote summary file: {summary_path}")
    print(f"Layers exported: {len(layers_payload)}")


if __name__ == "__main__":
    main()
