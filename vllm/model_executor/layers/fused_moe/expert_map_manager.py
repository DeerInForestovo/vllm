# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Expert Map Manager for MoE layers.

This module contains the ExpertMapManager class which manages expert ID
mappings and placement strategies for Expert Parallelism in MoE models.
"""

import json
import re
from typing import Any

import torch

from vllm.config.parallel import ExpertPlacementStrategy
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEParallelConfig
from vllm.model_executor.layers.fused_moe.experts.rocm_aiter_moe import (
    init_aiter_topK_meta_data,
)

logger = init_logger(__name__)

_CUSTOM_EXPERT_MAP_CACHE: dict[str, dict[str, Any]] = {}


def _extract_layer_id(layer_name: str | None) -> int | None:
    if not layer_name:
        return None

    match = re.search(r"(?:^|\.)layers\.(\d+)(?:\.|$)", layer_name)
    if match:
        return int(match.group(1))

    fallback = re.search(r"(\d+)", layer_name)
    return int(fallback.group(1)) if fallback else None


def _load_custom_expert_layers(config_file: str) -> dict[str, Any]:
    if config_file not in _CUSTOM_EXPERT_MAP_CACHE:
        with open(config_file, encoding="utf-8") as f:
            payload = json.load(f)
        _CUSTOM_EXPERT_MAP_CACHE[config_file] = payload
    return _CUSTOM_EXPERT_MAP_CACHE[config_file]


def _get_custom_rank_to_experts(
    config_file: str,
    layer_id: int,
    ep_size: int,
    global_num_experts: int,
) -> list[list[int]] | None:
    payload = _load_custom_expert_layers(config_file)
    layers = payload.get("layers")
    if not isinstance(layers, dict):
        raise ValueError(
            "Custom expert placement file must contain a dict field 'layers'."
        )

    layer_entry = layers.get(str(layer_id))
    if layer_entry is None:
        return None

    if isinstance(layer_entry, dict):
        rank_to_experts = layer_entry.get("rank_to_experts")
    else:
        rank_to_experts = layer_entry

    if not isinstance(rank_to_experts, list):
        raise ValueError(
            f"Invalid custom mapping for layer {layer_id}: expected list of ranks."
        )
    if len(rank_to_experts) != ep_size:
        raise ValueError(
            f"Custom mapping for layer {layer_id} has {len(rank_to_experts)} ranks, "
            f"but ep_size is {ep_size}."
        )

    normalized: list[list[int]] = []
    seen = set()
    for rank, experts in enumerate(rank_to_experts):
        if not isinstance(experts, list):
            raise ValueError(
                f"Custom mapping for layer {layer_id}, rank {rank} must be a list."
            )
        parsed = [int(expert_id) for expert_id in experts]
        for expert_id in parsed:
            if not 0 <= expert_id < global_num_experts:
                raise ValueError(
                    f"Custom mapping layer {layer_id} has out-of-range expert "
                    f"id {expert_id}, expected [0, {global_num_experts})."
                )
            if expert_id in seen:
                raise ValueError(
                    f"Custom mapping layer {layer_id} contains duplicate "
                    f"expert id {expert_id}."
                )
            seen.add(expert_id)
        normalized.append(parsed)

    expected = set(range(global_num_experts))
    if seen != expected:
        missing = sorted(expected - seen)
        raise ValueError(
            f"Custom mapping layer {layer_id} must cover all experts exactly once. "
            f"Missing experts: {missing[:10]}"
            f"{'...' if len(missing) > 10 else ''}."
        )

    return normalized


def determine_expert_map(
    ep_size: int,
    ep_rank: int,
    global_num_experts: int,
    expert_placement_strategy: ExpertPlacementStrategy = "linear",
    custom_rank_to_experts: list[list[int]] | None = None,
    num_fused_shared_experts: int = 0,
    return_expert_mask: bool = False,
) -> tuple[int, torch.Tensor | None, torch.Tensor | None]:
    """
    Calculates how many experts should be assigned to each rank for EP and
    creates a mapping from global to local expert index. Experts are
    distributed evenly across ranks. Any remaining are assigned to the
    last rank.

    Args:
        ep_size: The size of the expert parallel group
        ep_rank: The rank of the current process in the expert parallel
            group
        global_num_experts: The total number of experts in the model.
        expert_placement_strategy: The expert placement strategy.
        custom_rank_to_experts: Custom rank-to-experts assignment for this
            layer. Used only when strategy is 'custom'.
        num_fused_shared_experts: Number of fused shared experts (for AITER)
        return_expert_mask: Whether to return expert mask for AITER

    Returns:
        tuple[int, Optional[torch.Tensor], Optional[torch.Tensor]]: A tuple containing:
            - local_num_experts (int): The number of experts assigned
                to the current rank.
            - expert_map (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts,) mapping from global to local index.
                Contains -1 for experts not assigned to the current rank.
                Returns None if ep_size is 1.
            - expert_mask (Optional[torch.Tensor]): A tensor of shape
                (global_num_experts + num_fused_shared_experts + 1,)
                containing 1 for experts assigned to the current rank
                and 0 for sentinel.
                Returns None if ep_size is 1.
                Used only when AITER MOE is enabled.
    """
    from typing import get_args

    assert ep_size > 0
    if ep_size == 1:
        return (global_num_experts, None, None)

    # Create a tensor of size num_experts filled with -1
    expert_map = torch.full((global_num_experts,), -1, dtype=torch.int32)

    # Create an expert map for the local experts
    if expert_placement_strategy == "linear":
        base_experts = global_num_experts // ep_size
        remainder = global_num_experts % ep_size
        local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts
        start_idx = ep_rank * base_experts + min(ep_rank, remainder)
        expert_map[start_idx : start_idx + local_num_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    elif expert_placement_strategy == "round_robin":
        base_experts = global_num_experts // ep_size
        remainder = global_num_experts % ep_size
        local_num_experts = base_experts + 1 if ep_rank < remainder else base_experts
        local_log_experts = torch.arange(
            ep_rank, global_num_experts, ep_size, dtype=torch.int32
        )

        expert_map[local_log_experts] = torch.arange(
            0, local_num_experts, dtype=torch.int32
        )
    elif expert_placement_strategy == "custom":
        if custom_rank_to_experts is None:
            raise ValueError(
                "Custom expert placement strategy requires custom_rank_to_experts."
            )
        local_experts = custom_rank_to_experts[ep_rank]
        local_num_experts = len(local_experts)
        if local_num_experts > 0:
            expert_map[torch.tensor(local_experts, dtype=torch.int64)] = torch.arange(
                0, local_num_experts, dtype=torch.int32
            )
    else:
        raise ValueError(
            "Unsupported expert placement strategy "
            f"'{expert_placement_strategy}', expected one of "
            f"{get_args(ExpertPlacementStrategy)}"
        )

    expert_mask = None
    if return_expert_mask:
        expert_mask = torch.ones(
            (global_num_experts + num_fused_shared_experts + 1,), dtype=torch.int32
        )
        expert_mask[-1] = 0
        expert_mask[:global_num_experts] = expert_map > -1
        expert_map = torch.cat(
            (
                expert_map,
                torch.tensor(
                    [local_num_experts + i for i in range(num_fused_shared_experts)],
                    dtype=torch.int32,
                ),
            ),
            dim=0,
        )

    return (local_num_experts, expert_map, expert_mask)


def determine_expert_placement_strategy(
    expert_placement_strategy: ExpertPlacementStrategy,
    moe_parallel_config: FusedMoEParallelConfig,
    num_expert_group: int | None,
    num_redundant_experts: int,
    enable_eplb: bool,
) -> ExpertPlacementStrategy:
    if expert_placement_strategy == "round_robin":
        round_robin_supported = (
            (num_expert_group is not None and num_expert_group > 1)
            and num_redundant_experts == 0
            and not enable_eplb
        )

        if not round_robin_supported:
            logger.warning(
                "Round-robin expert placement is only supported for "
                "models with multiple expert groups and no redundant "
                "experts. Falling back to linear expert placement."
            )
            return "linear"
        if (
            moe_parallel_config.use_all2all_kernels
            and not moe_parallel_config.needs_round_robin_routing_tables
        ):
            logger.warning(
                "Round-robin expert placement currently only supports "
                "the DeepEP low-latency or NIXL EP backend, but '%s' was configured. "
                "Falling back to linear expert placement.",
                moe_parallel_config.all2all_backend,
            )
            return "linear"

    return expert_placement_strategy


class ExpertMapManager:
    """
    Manages expert ID mappings and placement for Expert Parallelism.

    Responsibilities:
    - Calculate local vs global expert counts
    - Map between global, local, and physical expert IDs
    - Manage placement strategies (linear, round_robin, custom)
    - Maintain routing tables for round-robin placement
    - Support dynamic reconfiguration of EP topology

    When expert_map is required:
    - Expert Parallelism (EP) is enabled, i.e., when ep_size > 1
    - EP disabled (ep_size == 1): expert_map is None
      * All experts are local to the current rank
      * No mapping is needed
    - EP enabled (ep_size > 1): expert_map is created
      * Maps global expert IDs to local expert IDs
      * Shape: (global_num_experts,)
      * Contains the local expert index for experts on this rank, -1 for experts
         on other ranks
      * Used by kernels to handle distributed expert execution
    - Kernel support varies:
      * Supports expert_map: fused_moe, fused_marlin_moe, fused_humming_moe,
        rocm_aiter_fused_moe, deep_gemm_moe, xpu_moe, gpt_oss_triton_kernels_moe
      * Does not support: flashinfer_cutlass_moe, fused_batched_moe, most cutlass_moe
        variants, trtllm_* kernels
      * When kernel doesn't support expert_map: The modular kernel method sets
        expert_map=None even if EP is enabled
    """

    def __init__(
        self,
        max_num_batched_tokens: int,
        top_k: int,
        global_num_experts: int,
        num_redundant_experts: int,
        num_expert_group: int | None,
        moe_parallel_config: FusedMoEParallelConfig,
        placement_strategy: ExpertPlacementStrategy,
        enable_eplb: bool,
        layer_name: str = "",
        expert_placement_config_file: str | None = None,
        num_fused_shared_experts: int = 0,
        rocm_aiter_enabled: bool = False,
    ):
        """
        Initialize expert map manager.

        Args:
            global_num_experts: Total number of experts across all ranks
            moe_parallel_config: MoE parallel configuration (contains ep_size,
                                 ep_rank, backend flags)
            placement_strategy: Strategy for placing experts ('linear',
                'round_robin', or 'custom')
            layer_name: Name/prefix of this layer, used to resolve layer id
                for custom placement.
            expert_placement_config_file: JSON file path for custom placement.
            num_fused_shared_experts: Number of fused shared experts (for AITER)
            rocm_aiter_enabled: Whether ROCm AITER fusion is enabled
        """
        self.global_num_experts = global_num_experts
        self.moe_parallel_config = moe_parallel_config
        self.num_fused_shared_experts = num_fused_shared_experts
        self.rocm_aiter_enabled = rocm_aiter_enabled
        self.top_k = top_k
        self.max_num_batched_tokens = max_num_batched_tokens
        self.layer_name = layer_name
        self.layer_id = _extract_layer_id(layer_name)
        self.expert_placement_config_file = expert_placement_config_file
        self._custom_rank_to_experts: list[list[int]] | None = None

        if moe_parallel_config.use_ep:
            # Determine expert placement strategy before creating manager
            placement_strategy = determine_expert_placement_strategy(
                expert_placement_strategy=placement_strategy,
                moe_parallel_config=moe_parallel_config,
                num_expert_group=num_expert_group,
                num_redundant_experts=num_redundant_experts,
                enable_eplb=enable_eplb,
            )

        # Determine effective placement strategy
        self._placement_strategy = self._determine_placement_strategy(
            placement_strategy
        )

        if self._placement_strategy == "custom":
            self._custom_rank_to_experts = self._load_custom_rank_to_experts_for_layer()
            if self._custom_rank_to_experts is None:
                self._placement_strategy = "linear"

        # Calculate expert mappings
        self._calculate_expert_maps()

        # Initialize routing tables if needed
        self._routing_tables = self._init_routing_tables()

        self._init_aiter_shared_experts_topK_buffer()

        if self.use_ep and self.rocm_aiter_enabled:
            expert_mask = self.expert_mask
            assert expert_mask is None or torch.all(
                (expert_mask == 0) | (expert_mask == 1)
            ), "Aiter Fused MoE kernel only supports expert_map with 0 and 1s."

        # Log EP configuration
        if self.use_ep:
            logger.info_once(
                "[EP Rank %s/%s] Expert parallelism is enabled. Expert "
                "placement strategy: %s. Local/global"
                " number of experts: %s/%s. Experts local to global index map:"
                " %s.",
                self.ep_rank,
                self.ep_size,
                self.placement_strategy,
                self.local_num_experts,
                self.global_num_experts,
                self.get_compressed_map_string(),
            )

    def _init_aiter_shared_experts_topK_buffer(self):
        if self.num_fused_shared_experts > 0:
            dp_size = self.moe_parallel_config.dp_size
            init_aiter_topK_meta_data(
                n_routed_experts=self.global_num_experts,
                n_shared_experts=self.num_fused_shared_experts,
                top_k=self.top_k,
                tp_rank=self.ep_rank if self.use_ep else self.tp_rank,
                tp_size=self.ep_size if self.use_ep else self.tp_size,
                shared_experts_score=1.0,
                max_num_tokens=self.max_num_batched_tokens * dp_size,
                is_EP=self.use_ep,
            )

    @property
    def use_ep(self) -> int:
        return self.moe_parallel_config.use_ep

    @property
    def ep_size(self) -> int:
        return self.moe_parallel_config.ep_size

    @property
    def ep_rank(self) -> int:
        return self.moe_parallel_config.ep_rank

    @property
    def tp_size(self) -> int:
        return self.moe_parallel_config.tp_size

    @property
    def tp_rank(self) -> int:
        return self.moe_parallel_config.tp_rank

    @property
    def local_num_experts(self) -> int:
        return self._local_num_experts

    @property
    def expert_map(self) -> torch.Tensor | None:
        """
        Mapping from global expert ID to local expert ID.

        Returns tensor of shape (global_num_experts,) where:
        - expert_map[global_id] = local_id if expert is on this rank
        - expert_map[global_id] = -1 if expert is not on this rank

        Returns None if EP is not enabled (ep_size == 1).
        """
        return self._expert_map

    @property
    def expert_mask(self) -> torch.Tensor | None:
        """
        Expert mask for AITER fusion (ROCm-specific).

        Returns tensor of shape (global_num_experts + num_fused_shared + 1,)
        where 1 indicates expert is on this rank, 0 otherwise.
        """
        return self._expert_mask

    @property
    def placement_strategy(self) -> ExpertPlacementStrategy:
        """Expert placement strategy ('linear', 'round_robin', or 'custom')."""
        return self._placement_strategy

    @property
    def routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """
        Routing tables for round-robin placement.

        Returns (global_to_physical, physical_to_global, local_to_global)
        or None if not using round-robin or tables not needed.
        """
        return self._routing_tables

    def map_global_to_local(self, global_id: int) -> int:
        """
        Map global expert ID to local expert ID.

        Args:
            global_id: Global expert ID (0 to global_num_experts - 1)

        Returns:
            Local expert ID (0 to local_num_experts - 1)

        Raises:
            ValueError: If expert is not on this rank
        """
        if self._expert_map is None:
            return global_id

        return self._expert_map[global_id].item()

    def is_local_expert(self, global_id: int) -> bool:
        """Check if expert is assigned to this rank."""
        if self._expert_map is None:
            return True
        return self._expert_map[global_id] != -1

    def get_local_expert_ids(self) -> list[int]:
        """Get list of global IDs for experts on this rank."""
        if self._expert_map is None:
            return list(range(self.global_num_experts))

        return torch.where(self._expert_map != -1)[0].tolist()

    def update(
        self,
        moe_parallel_config: FusedMoEParallelConfig,
        global_num_experts: int,
    ) -> None:
        """
        Update expert mappings for new EP configuration.

        Used during dynamic reconfiguration (e.g., elastic scaling).

        Args:
            global_num_experts: New total number of experts across all ranks
            moe_parallel_config: New MoE parallel configuration (contains ep_size,
                                 ep_rank, backend flags)
        """
        self.moe_parallel_config = moe_parallel_config
        self.global_num_experts = global_num_experts

        if self._expert_map is not None:
            device = self._expert_map.device
        elif self._expert_mask is not None:
            device = self._expert_mask.device
        else:
            raise AssertionError("_expert_map or _expert_mask must be present.")

        with device:
            self._calculate_expert_maps()
            self._routing_tables = self._init_routing_tables()

            # Reinitialize AITER buffer if needed and parameters provided
            self._init_aiter_shared_experts_topK_buffer()

    def get_compressed_map_string(self) -> str:
        """
        Get compressed string representation of expert map for logging.

        Returns string mapping local to global expert IDs.
        """
        if self._expert_map is None:
            return f"[0..{self.global_num_experts - 1}]"

        global_indices = torch.where(self._expert_map != -1)[0]
        local_indices = self._expert_map[global_indices]
        return ", ".join(
            f"{local_index.item()}->{global_index.item()}"
            for local_index, global_index in zip(local_indices, global_indices)
        )

    # Private methods

    def _determine_placement_strategy(
        self, requested_strategy: ExpertPlacementStrategy
    ) -> ExpertPlacementStrategy:
        """Determine effective placement strategy based on config."""
        if requested_strategy == "custom":
            if self.ep_size == 1:
                return "linear"
            return "custom"

        if requested_strategy != "round_robin":
            return requested_strategy

        # Round-robin requires specific conditions
        if self.ep_size == 1:
            return "linear"

        if (
            self.moe_parallel_config.use_all2all_kernels
            and not self.moe_parallel_config.needs_round_robin_routing_tables
        ):
            logger.warning(
                "Round-robin placement requires DeepEP-ll or NIXL backend. "
                "Falling back to linear."
            )
            return "linear"

        return "round_robin"

    def _load_custom_rank_to_experts_for_layer(self) -> list[list[int]] | None:
        if not self.expert_placement_config_file:
            raise ValueError(
                "expert_placement_strategy='custom' requires setting "
                "expert_placement_config_file."
            )
        if self.layer_id is None:
            raise ValueError(
                "Unable to infer layer id from layer name for custom expert "
                f"placement: '{self.layer_name}'."
            )

        mapping = _get_custom_rank_to_experts(
            config_file=self.expert_placement_config_file,
            layer_id=self.layer_id,
            ep_size=self.ep_size,
            global_num_experts=self.global_num_experts,
        )
        if mapping is None:
            logger.warning_once(
                "Layer %s is not present in custom placement file '%s'; "
                "falling back to linear placement for this layer.",
                self.layer_id,
                self.expert_placement_config_file,
            )
            return None

        return mapping

    def _calculate_expert_maps(self) -> None:
        """Calculate expert mappings based on placement strategy."""
        (
            self._local_num_experts,
            self._expert_map,
            self._expert_mask,
        ) = determine_expert_map(
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            global_num_experts=self.global_num_experts,
            expert_placement_strategy=self._placement_strategy,
            custom_rank_to_experts=self._custom_rank_to_experts,
            num_fused_shared_experts=self.num_fused_shared_experts,
            return_expert_mask=self.rocm_aiter_enabled,
        )

        self._local_num_experts += self.num_fused_shared_experts

    def _init_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """
        Ensure routing tables are initialized if needed for round-robin.

        This is a public method that can be called to explicitly initialize
        routing tables. It's safe to call multiple times (idempotent).
        """
        if self._placement_strategy != "round_robin":
            return None

        if not self.moe_parallel_config.needs_round_robin_routing_tables:
            return None

        if self._expert_map is None:
            return None

        return self._init_round_robin_expert_routing_tables()

    def _init_round_robin_expert_routing_tables(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build routing tables for round-robin placement."""
        assert self.num_fused_shared_experts == 0, (
            "Round robin not supported for AITER."
        )

        global_indices = torch.arange(
            self.global_num_experts,
            dtype=torch.long,
        )
        owner = torch.remainder(global_indices, self.ep_size)
        local_index = torch.div(global_indices, self.ep_size, rounding_mode="floor")

        base = self.global_num_experts // self.ep_size
        remainder = self.global_num_experts % self.ep_size
        physical_offset = owner * base

        if remainder > 0:
            remainder_tensor = torch.tensor(
                remainder,
                dtype=torch.long,
            )
            physical_offset = physical_offset + torch.minimum(owner, remainder_tensor)

        global_to_physical = physical_offset + local_index
        physical_to_global = torch.empty_like(global_to_physical)
        physical_to_global[global_to_physical] = global_indices

        local_global = torch.arange(
            self.ep_rank,
            self.global_num_experts,
            self.ep_size,
            dtype=torch.long,
        )
        if local_global.numel() != self._local_num_experts:
            local_global = local_global[: self._local_num_experts]

        return (global_to_physical, physical_to_global, local_global)
