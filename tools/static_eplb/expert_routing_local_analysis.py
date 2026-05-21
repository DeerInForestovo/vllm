#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Expert routing frequency analysis using local AsyncLLM.

This is the offline variant of expert routing collection. It follows the
same capture style as examples/rl/routed_experts_e2e.py: it creates a local
AsyncLLM engine with enable_return_routed_experts=True and reads
completion.routed_experts directly, so it does not depend on the HTTP serve
path returning routed_experts.

Usage:
    python expert_routing_local_analysis.py \
        --dataset-file /mnt/data/0/kuangliang/glm5.1-test/32b-emotion-20260304.jsonl \
        --model /mnt/data/0/kuangliang/glm5.1-model \
        --tensor-parallel-size 8 \
        --enable-expert-parallel \
        --max-model-len 100000 \
        --gpu-memory-utilization 0.9 \
        --served-model-name glm5.1-data-analysis \
        --trust-remote-code \
        --kv-cache-dtype fp8 \
        --attention-backend FLASHMLA_SPARSE \
        --language-model-only \
        --max-num-seqs 16 \
        --enable-return-routed-experts \
        --num-prompt 300 \
        --concurrent-list 16

Notes:
    This local AsyncLLM path supports the engine arguments that map to
    AsyncEngineArgs. Flags such as --aggregate-engine-logging belong to the
    online serve process and are not used here.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import chardet
import numpy as np
from tqdm import tqdm

from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.sampling_params import SamplingParams
from vllm.v1.engine.async_llm import AsyncLLM

DEFAULT_MODEL = "/mnt/data/0/kuangliang/glm5.1-model"


@dataclass
class PromptResult:
    request_id: str
    routed_experts: np.ndarray
    num_completion_tokens: int


class ExpertRoutingStatistics:
    """Collects and aggregates expert routing statistics."""

    def __init__(self, num_layers: int, num_experts: int):
        self.num_layers = num_layers
        self.num_experts = num_experts
        self.expert_count_all_topk = np.zeros((num_layers, num_experts), dtype=np.int64)
        self.expert_count_top1 = np.zeros((num_layers, num_experts), dtype=np.int64)
        self.request_stats: List[Dict[str, Any]] = []

    def add_routed_experts(
        self,
        routed_experts: np.ndarray,
        request_id: str,
        num_completion_tokens: int,
    ) -> None:
        if routed_experts is None:
            return

        routed_experts = np.asarray(routed_experts)
        if routed_experts.ndim != 3:
            raise ValueError(
                f"Expected routed_experts to have 3 dims, got shape {routed_experts.shape}"
            )

        num_tokens, num_layers, topk = routed_experts.shape

        for layer_idx in range(num_layers):
            for topk_rank in range(topk):
                expert_ids = routed_experts[:, layer_idx, topk_rank]
                valid_ids = expert_ids[expert_ids >= 0]
                for expert_id in valid_ids:
                    if 0 <= expert_id < self.num_experts:
                        self.expert_count_all_topk[layer_idx, expert_id] += 1

        for layer_idx in range(num_layers):
            expert_ids = routed_experts[:, layer_idx, 0]
            valid_ids = expert_ids[expert_ids >= 0]
            for expert_id in valid_ids:
                if 0 <= expert_id < self.num_experts:
                    self.expert_count_top1[layer_idx, expert_id] += 1

        self.request_stats.append(
            {
                "request_id": request_id,
                "num_tokens": num_tokens,
                "num_completion_tokens": num_completion_tokens,
                "timestamp": time.time(),
            }
        )

    def to_records(self) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        for layer_idx in range(self.num_layers):
            total_all = int(self.expert_count_all_topk[layer_idx, :].sum())
            total_top1 = int(self.expert_count_top1[layer_idx, :].sum())
            for expert_idx in range(self.num_experts):
                count_all = int(self.expert_count_all_topk[layer_idx, expert_idx])
                count_top1 = int(self.expert_count_top1[layer_idx, expert_idx])
                if count_all > 0 or count_top1 > 0:
                    records.append(
                        {
                            "layer_id": layer_idx,
                            "expert_id": expert_idx,
                            "count_all_topk": count_all,
                            "count_top1": count_top1,
                            "ratio_all_topk": count_all / total_all if total_all else 0.0,
                            "ratio_top1": count_top1 / total_top1 if total_top1 else 0.0,
                        }
                    )
        return records


def detect_encoding(file_path: str) -> str:
    with open(file_path, "rb") as f:
        raw_data = f.read(10000)
    result = chardet.detect(raw_data)
    return result["encoding"] or "utf-8"


def get_input(file_path: str, num_prompt: int) -> List[str]:
    encoding = detect_encoding(file_path)
    print(f"检测到文件编码: {encoding}")

    query_list: List[str] = []
    with open(file_path, "r", encoding=encoding, errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue

            prompt = None
            if isinstance(data, dict):
                for msg_field in ("input_messages", "messages"):
                    if msg_field in data:
                        messages = data[msg_field]
                        if isinstance(messages, str):
                            try:
                                messages = json.loads(messages)
                            except json.JSONDecodeError:
                                continue
                        if isinstance(messages, list):
                            for msg in messages:
                                if msg.get("role") == "user":
                                    content = msg.get("content")
                                    if isinstance(content, list):
                                        for item in content:
                                            if item.get("type") == "text":
                                                prompt = item.get("text", "")
                                                break
                                    else:
                                        prompt = content
                                    break
                        if prompt:
                            break

                if prompt is None:
                    for field in ("prompt", "instruction", "input", "query", "text"):
                        if field in data:
                            prompt = data[field]
                            break

            elif isinstance(data, list):
                for msg in data:
                    if msg.get("role") == "user":
                        prompt = msg.get("content")
                        break

            if prompt:
                query_list.append(prompt)

    query_list = query_list[:num_prompt]
    print(f"问句总数: {len(query_list)}")
    return query_list


async def _collect_for_concurrency(
    engine: AsyncLLM,
    prompts: List[str],
    max_new_tokens: int,
    concurrency: int,
    stats: ExpertRoutingStatistics,
) -> int:
    sampling_params = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    semaphore = asyncio.Semaphore(concurrency)

    async def _one_prompt(prompt: str, idx: int) -> bool:
        async with semaphore:
            request_id = str(uuid.uuid4())
            final_output = None
            async for output in engine.generate(prompt, sampling_params, request_id):
                final_output = output

            if final_output is None:
                print(f"Prompt {idx}: no final output")
                return False

            completion = final_output.outputs[0]
            routed = completion.routed_experts
            if routed is None:
                print(f"Prompt {idx}: routed_experts is None")
                return False

            routed_array = np.asarray(routed)
            stats.add_routed_experts(
                routed_array,
                request_id,
                len(completion.token_ids),
            )
            return True

    tasks = [asyncio.create_task(_one_prompt(prompt, idx)) for idx, prompt in enumerate(prompts)]
    successful = 0
    for result in tqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc=f"并发 {concurrency}",
    ):
        if await result:
            successful += 1
    return successful


def write_expert_stats_csv(stats: ExpertRoutingStatistics, output_file: str) -> None:
    records = stats.to_records()
    if not records:
        print("No expert statistics to write")
        return

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(
            "layer_id,expert_id,count_all_topk,count_top1,ratio_all_topk,ratio_top1\n"
        )
        for rec in records:
            f.write(
                f"{rec['layer_id']},{rec['expert_id']},"
                f"{rec['count_all_topk']},{rec['count_top1']},"
                f"{rec['ratio_all_topk']:.6f},{rec['ratio_top1']:.6f}\n"
            )

    print(f"Expert statistics written to {output_file}")


async def main_async() -> None:
    parser = argparse.ArgumentParser(
        description="Expert routing frequency analysis using local AsyncLLM"
    )
    parser.add_argument("--dataset-file", type=str, required=True)
    parser.add_argument("--output-expert-stats", type=str, default="expert_stats.csv")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL)
    parser.add_argument("--served-model-name", type=str, default=None)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--aggregate-engine-logging", action="store_true")
    parser.add_argument("--enable-expert-parallel", action="store_true")
    parser.add_argument("--max-model-len", type=int, default=100000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--kv-cache-dtype", type=str, default="auto")
    parser.add_argument("--attention-backend", type=str, default=None)
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--language-model-only", action="store_true")
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--num-prompt", type=int, default=50)
    parser.add_argument("--concurrent-list", nargs="*", type=int, default=[1])
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--num-experts", type=int, default=None)
    parser.add_argument(
        "--enforce-eager",
        action="store_true",
        help="Use eager execution for faster startup/debugging",
    )
    parser.add_argument(
        "--enable-return-routed-experts",
        action="store_true",
        help="Return routed experts from the engine.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.dataset_file):
        print(f"Error: Dataset file not found: {args.dataset_file}")
        return

    prompts = get_input(args.dataset_file, args.num_prompt)
    if not prompts:
        print("Error: No prompts loaded")
        return

    print("Initializing local AsyncLLM engine...")
    engine_args = AsyncEngineArgs(
        model=args.model,
        served_model_name=args.served_model_name,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        aggregate_engine_logging=args.aggregate_engine_logging,
        enable_expert_parallel=args.enable_expert_parallel,
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        kv_cache_dtype=args.kv_cache_dtype,
        attention_backend=args.attention_backend,
        language_model_only=args.language_model_only,
        max_num_seqs=args.max_num_seqs,
        enable_return_routed_experts=args.enable_return_routed_experts,
        disable_log_stats=True,
        enforce_eager=args.enforce_eager,
    )
    engine = AsyncLLM.from_engine_args(engine_args)

    try:
        hf_config = engine.model_config.hf_text_config
        num_layers = args.num_layers or getattr(hf_config, "num_hidden_layers", None)
        num_experts = args.num_experts or getattr(hf_config, "num_experts", None) or getattr(
            hf_config, "n_routed_experts", None
        ) or getattr(hf_config, "num_local_experts", None)

        if num_layers is None:
            raise RuntimeError("Could not determine num_layers from model config")
        if num_experts is None:
            raise RuntimeError("Could not determine num_experts from model config")

        stats = ExpertRoutingStatistics(int(num_layers), int(num_experts))

        print(
            f"Starting local expert routing analysis with {len(prompts)} prompts, "
            f"max_tokens={args.max_new_tokens}"
        )
        print(f"Model: {args.model}")
        print(f"TP: {args.tensor_parallel_size}")
        print(f"num_layers: {num_layers}, num_experts: {num_experts}")

        for concurrency in sorted(args.concurrent_list):
            print(f"\n{'=' * 60}")
            print(f"Testing concurrency level: {concurrency}")
            print(f"{'=' * 60}")
            successful = await _collect_for_concurrency(
                engine=engine,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                concurrency=concurrency,
                stats=stats,
            )
            print(f"Successful requests: {successful}/{len(prompts)}")
            time.sleep(1)

        write_expert_stats_csv(stats, args.output_expert_stats)

        print(f"\n{'=' * 60}")
        print("Summary Statistics")
        print(f"{'=' * 60}")
        total_all_topk = int(np.sum(stats.expert_count_all_topk))
        total_top1 = int(np.sum(stats.expert_count_top1))
        print(f"Total token-expert routings (all topk): {total_all_topk}")
        print(f"Total token-expert routings (top1): {total_top1}")
        print(
            f"Average load per expert (all topk): "
            f"{total_all_topk / stats.num_experts / stats.num_layers:.2f}"
        )
        print(
            f"Average load per expert (top1): "
            f"{total_top1 / stats.num_experts / stats.num_layers:.2f}"
        )
    finally:
        engine.shutdown()


if __name__ == "__main__":
    asyncio.run(main_async())
