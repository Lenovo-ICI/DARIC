import copy
from enum import IntEnum
import torch
from dataclasses import dataclass, fields
from typing import List, Any, Optional
from vllm.config import VllmConfig
from vllm.sampling_params import SamplingParams

from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput, CachedRequestData
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec, SlidingWindowSpec, UniformTypeKVCacheSpecs
from vllm.distributed.parallel_state import get_world_group

from ...configs import ProcessLocalConfig, CacheTopo, LayerCacheTopo, CacheEngineConfig
from ...cache_engine import LookUpResult
from ...logging import get_logger

logger = get_logger(__name__)


@dataclass
class RequestTracker:
    req_id: str
    all_token_ids: list[int]
    prompt_len: int
    current_len: int
    block_ids: tuple[list[int], ...]
    sampling_params: SamplingParams
    is_cached: bool = False
    write_cache_ids: Optional[set] = None

    @staticmethod
    def from_new_request(
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
    ) -> "RequestTracker":
        return RequestTracker(
            req_id=new_request.req_id,
            all_token_ids=new_request.prompt_token_ids,
            current_len=num_tokens_to_compute,
            prompt_len=len(new_request.prompt_token_ids),
            block_ids=tuple([block_ids.copy() for block_ids in new_request.block_ids]),
            sampling_params=new_request.sampling_params
        )
    
    @staticmethod
    def from_req_meta(
        req_meta: "ReqMeta",
    ) -> "RequestTracker":
        return RequestTracker(
            req_id=req_meta.req_id,
            all_token_ids=None,
            current_len=req_meta.current_len,
            prompt_len=req_meta.prompt_len,
            block_ids=None,
            sampling_params=None
        )
    
    def update(self,
        new_len: int,
        new_block_ids: Optional[tuple[list[int], ...]],
        preemption: Optional[list[int]] = None,
    ) -> None:
        self.current_len = new_len

        if new_block_ids is None or len(new_block_ids) == 0:
            new_block_ids = ()
        elif not isinstance(new_block_ids, tuple):
            raise ValueError(f"Unsupported new_block_ids type {type(new_block_ids)}")
        if not preemption:
            for i in range(len(new_block_ids)):
                self.block_ids[i].extend(new_block_ids[i])
        else:
            self.block_ids = tuple([block_ids.copy() for block_ids in new_block_ids])

    def mark_cached(self) -> None:
        self.is_cached = True


class PICMode(IntEnum):
    NONE = 0
    STRICT = 1
    FLEXIBLE = 2


class RequestStatus(IntEnum):
    PREFIX = 0
    GROUP = 1


class WorkerStage(IntEnum):
    PRE_PROCESS = 0
    RECOMPUTE = 1
    POST_PROCESS = 2
    NONE = 3


@dataclass
class ReqMeta:
    req_id: str
    current_len: int
    prompt_len: int
    status: RequestStatus
    external_meta: LookUpResult
    can_save: bool = False
    can_load: bool = False

    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        status: RequestStatus,
        external_meta: LookUpResult,
        can_save: bool = False,
        can_load: bool = False,
    ) -> "ReqMeta":
        
        return ReqMeta(
            req_id=tracker.req_id,
            current_len=tracker.current_len,
            prompt_len=tracker.prompt_len,
            status=status,
            external_meta=external_meta.for_worker(),
            can_save=can_save,
            can_load=can_load,
        )


@dataclass
class DARICConnectorMetadata(KVConnectorMetadata):
    stage: WorkerStage
    requests: dict[str, ReqMeta]
    async_loads: dict[str, LookUpResult]
    preempted_req_ids: set[str]
    report_finished: bool
    step_pic: bool
 
    def __init__(self):
        self.stage = WorkerStage.NONE
        self.requests = {}
        self.async_loads = {}
        self.preempted_req_ids = set()
        self.report_finished = True
        self.step_pic = False


def lazy_register_process_local_config(
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ) -> None:
    model_key = vllm_config.model_config.compute_hash()
    if vllm_config.speculative_config is not None:
        if vllm_config.speculative_config.draft_model_config is not None:
            draft_model_key = vllm_config.speculative_config.draft_model_config.compute_hash()
            if draft_model_key != model_key:
                raise NotImplementedError(f"Speculative decoding with draft model is not supported yet.")
            
    is_deepseek_mla = vllm_config.model_config.is_deepseek_mla
    block_size = vllm_config.cache_config.block_size

    tp_size = vllm_config.parallel_config.tensor_parallel_size
    pp_size = vllm_config.parallel_config.pipeline_parallel_size
    world_size = vllm_config.parallel_config.world_size
    if tp_size * pp_size != world_size:
        raise ValueError(
            f"tp_size * pp_size ({tp_size} * {pp_size}) != world_size ({world_size}), currently DP and other parallelism are not supported."
        )
    world_rank = vllm_config.parallel_config.rank
    tp_rank = world_rank % tp_size
    pp_rank = world_rank // tp_size

    kv_role = vllm_config.kv_transfer_config.kv_role
    engine_id = vllm_config.kv_transfer_config.engine_id

    kv_cache_groups = kv_cache_config.kv_cache_groups
    cache_topo = CacheTopo(layers={}, groups=[])
    for g_idx, g in enumerate(kv_cache_groups):
        cache_topo.groups.append(g.layer_names)
        for layer_name in g.layer_names:
            cache_topo.layers[layer_name] = LayerCacheTopo(
                group_idx=g_idx,
                logical_block_size = g.kv_cache_spec.block_size,
                block_byte_size = None,
                page_first = None,
                is_mamba=isinstance(g.kv_cache_spec, MambaSpec),
            )
    ProcessLocalConfig.cache_topo = cache_topo
    
    ProcessLocalConfig.engine_id = engine_id
    ProcessLocalConfig.model_key = model_key
    ProcessLocalConfig.is_deepseek_mla = is_deepseek_mla

    ProcessLocalConfig.tp_size = tp_size
    ProcessLocalConfig.pp_size = pp_size
    ProcessLocalConfig.world_size = world_size
    ProcessLocalConfig.tp_rank = tp_rank
    ProcessLocalConfig.pp_rank = pp_rank
    ProcessLocalConfig.rank = world_rank

    local_world_size = vllm_config.parallel_config.local_world_size
    node_rank = vllm_config.parallel_config.node_rank
    nnodes = vllm_config.parallel_config.nnodes
    local_rank = world_rank % local_world_size
    ProcessLocalConfig.local_rank = local_rank
    ProcessLocalConfig.local_world_size = local_world_size
    ProcessLocalConfig.node_rank = node_rank
    ProcessLocalConfig.nnodes = nnodes

    ProcessLocalConfig.kv_role = kv_role
    ProcessLocalConfig.block_size = block_size


def finish_register_process_local_config(
        vllm_config: "VllmConfig", 
        kv_cache_config: KVCacheConfig,
        kv_caches: dict[str, torch.Tensor]
    ) -> None:
    for layer_name, layer_cache in kv_caches.items():
        page_first = layer_cache.shape[0] == kv_cache_config.num_blocks
        ProcessLocalConfig.cache_topo.layers[layer_name].block_byte_size = int(layer_cache.nbytes // kv_cache_config.num_blocks)
        ProcessLocalConfig.cache_topo.layers[layer_name].page_first = page_first


def check_pic_available(kv_cache_config: KVCacheConfig) -> bool:
    # Check if all kv_cache_groups are either MambaSpec or SlidingWindowSpec, which support position-independent caching.
    pic_available = True
    for g in kv_cache_config.kv_cache_groups:
        if isinstance(g.kv_cache_spec, (MambaSpec, SlidingWindowSpec)):
            pic_available = False
            break
        if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs):
            if any(isinstance(spec, (MambaSpec, SlidingWindowSpec)) for _, spec in g.kv_cache_spec.kv_cache_specs.items()):
                pic_available = False
                break

    return pic_available


def validate_mamba_support(
    vllm_config: VllmConfig,
    cache_engine_config: CacheEngineConfig,
) -> None:
    mamba_group_idxs = ProcessLocalConfig.cache_topo.get_mamba_group_idxs()
    if not mamba_group_idxs:
        return
    if cache_engine_config.reuse_mode != "prefix":
        raise ValueError("Mamba cache offloading only supports prefix reuse.")
    if ProcessLocalConfig.kv_role != "kv_both":
        raise ValueError(
            "Mamba cache offloading currently only supports kv_role='kv_both'."
        )
    if cache_engine_config.backend_config.storage_backend != "local_store":
        raise ValueError(
            "Mamba cache offloading currently only supports local_store."
        )
    if cache_engine_config.backend_config.storage_layerwise:
        raise ValueError(
            "Mamba cache offloading currently requires layer_wise=false."
        )
    if vllm_config.cache_config.mamba_cache_mode != "align":
        raise ValueError(
            f"Mamba cache offloading requires mamba_cache_mode='align', got {vllm_config.cache_config.mamba_cache_mode}."
        )

    chunk_len = cache_engine_config.max_chunk_len
    mamba_block_sizes = {
        ProcessLocalConfig.cache_topo.layers[
            ProcessLocalConfig.cache_topo.groups[group_idx][0]
        ].logical_block_size
        for group_idx in mamba_group_idxs
    }
    if mamba_block_sizes != {chunk_len}:
        raise ValueError(
            "DARIC max_chunk_len must equal every Mamba checkpoint block "
            f"size; got max_chunk_len={chunk_len}, "
            f"mamba_block_sizes={sorted(mamba_block_sizes)}."
        )

    incompatible_group_sizes = [
        block_size
        for block_size in ProcessLocalConfig.cache_topo.get_all_group_block_size()
        if chunk_len % block_size != 0
    ]
    if incompatible_group_sizes:
        raise ValueError(
            f"DARIC max_chunk_len={chunk_len} must be divisible by every "
            f"cache group block size; got {incompatible_group_sizes}."
        )


def get_engine_load(
        request_trackers: dict[str, RequestTracker], 
        vllm_config: "VllmConfig") -> dict[str, Any]:
    active_request_count = len(request_trackers)
    active_token_count = sum(tracker.current_len for tracker in request_trackers.values())
    total_blocks = vllm_config.cache_config.num_gpu_blocks
    used_blocks = 0
    for tracker in request_trackers.values():
        for block_ids in tracker.block_ids:
            used_blocks += sum(1 for bid in block_ids if bid > 0)
    cache_utilization = used_blocks / total_blocks if total_blocks > 0 else 0.0
    return active_request_count, active_token_count, cache_utilization


def build_empty_scheduler_output(
    scheduler_output: SchedulerOutput,
    skip_cached_reqs: bool = False,
) -> SchedulerOutput:
    new_output = copy.copy(scheduler_output)
    for attr in [
        "scheduled_new_reqs",
        "finished_req_ids",
        "num_scheduled_tokens",
        "total_num_scheduled_tokens",
        "scheduled_spec_decode_tokens",
    ]:
        setattr(
            new_output, attr, copy.copy(getattr(scheduler_output, attr))
        )
    if skip_cached_reqs:
        new_output.scheduled_cached_reqs = CachedRequestData.make_empty()
    else:
        new_output.scheduled_cached_reqs = copy.deepcopy(
            scheduler_output.scheduled_cached_reqs
        )

    original_meta = scheduler_output.kv_connector_metadata
    new_meta = DARICConnectorMetadata()
    new_meta.requests = {}
    new_meta.preempted_req_ids = original_meta.preempted_req_ids.copy()
    new_meta.report_finished = original_meta.report_finished
    for k, old_req in original_meta.requests.items():
        new_meta.requests[k] = copy.copy(old_req)
    new_output.kv_connector_metadata = new_meta
    return new_output


def remove_from_cached_reqs(request_id: str, cached_reqs: CachedRequestData):
    remove_idx = cached_reqs.req_ids.index(request_id)
    cached_reqs.req_ids.pop(remove_idx)
    if hasattr(cached_reqs, "resumed_req_ids"): # vllm>=0.11.1
        if request_id in cached_reqs.resumed_req_ids:
            if isinstance(cached_reqs.resumed_req_ids, set):
                cached_reqs.resumed_req_ids.remove(request_id)
            else:
                cached_reqs.resumed_req_ids.pop(request_id)
    else:
        cached_reqs.resumed_from_preemption.pop(remove_idx)
    cached_reqs.num_computed_tokens.pop(remove_idx)
    if len(cached_reqs.new_token_ids) > 0:  # Bug of vllm<0.10.0
        cached_reqs.new_token_ids.pop(remove_idx)
    cached_reqs.new_block_ids.pop(remove_idx)
    if hasattr(cached_reqs, "num_output_tokens") and len(cached_reqs.num_output_tokens) > remove_idx:
        cached_reqs.num_output_tokens.pop(remove_idx)
    if hasattr(cached_reqs, "all_token_ids") and request_id in cached_reqs.all_token_ids:
        cached_reqs.all_token_ids.pop(request_id)


def rebuild_cached_reqs_subset(
    cached_reqs: CachedRequestData,
    keep_req_ids: set[str],
) -> CachedRequestData:
    """Build a new CachedRequestData keeping only ``keep_req_ids`` in one pass.

    This avoids the O(n^2) cost of repeatedly calling ``remove_from_cached_reqs``
    when filtering down to a subset (e.g. inside ``split_post_output_by_max_tokens``).
    """
    new_obj = copy.copy(cached_reqs)
    req_ids = cached_reqs.req_ids
    # Pre-compute keep indices in a single O(n) pass.
    keep_idx = [i for i, rid in enumerate(req_ids) if rid in keep_req_ids]

    new_obj.req_ids = [req_ids[i] for i in keep_idx]

    if hasattr(cached_reqs, "resumed_req_ids"):  # vllm>=0.11.1
        if isinstance(cached_reqs.resumed_req_ids, set):
            new_obj.resumed_req_ids = {
                rid for rid in cached_reqs.resumed_req_ids if rid in keep_req_ids
            }
        else:
            new_obj.resumed_req_ids = [
                rid for rid in cached_reqs.resumed_req_ids if rid in keep_req_ids
            ]
    else:
        new_obj.resumed_from_preemption = [
            cached_reqs.resumed_from_preemption[i] for i in keep_idx
        ]

    new_obj.num_computed_tokens = [cached_reqs.num_computed_tokens[i] for i in keep_idx]
    if len(cached_reqs.new_token_ids) > 0:  # Bug of vllm<0.10.0
        new_obj.new_token_ids = [cached_reqs.new_token_ids[i] for i in keep_idx]
    else:
        new_obj.new_token_ids = []
    new_obj.new_block_ids = [cached_reqs.new_block_ids[i] for i in keep_idx]

    if hasattr(cached_reqs, "num_output_tokens"):
        new_obj.num_output_tokens = [cached_reqs.num_output_tokens[i] for i in keep_idx]
    if hasattr(cached_reqs, "all_token_ids"):
        new_obj.all_token_ids = {
            rid: cached_reqs.all_token_ids[rid]
            for rid in new_obj.req_ids
            if rid in cached_reqs.all_token_ids
        }
    return new_obj


def rebuild_new_sub_request(
    request: RequestTracker,
    group_name: Optional[str],
    start_ptr: int,
    end_ptr: int,
    block_size: int
):
    """Rebuild a new request for the missed group."""
    block_end_idx = (
        end_ptr // block_size
        if end_ptr % block_size == 0
        else end_ptr // block_size + 1
    )

    sub_sampling_params = request.sampling_params
    sub_block_ids = [[]]
    if block_end_idx > len(request.block_ids[0]):
        sub_block_ids[0] = request.block_ids[0][start_ptr // block_size :]
    else:
        sub_block_ids[0] = request.block_ids[0][
            start_ptr // block_size : block_end_idx
        ]
    if group_name is not None:
        new_request_id = f"{request.req_id}_{group_name}"
    else:
        new_request_id = request.req_id
    if not "mm_features" in {f.name for f in fields(NewRequestData)}:
        new_request = NewRequestData(
            req_id=new_request_id,
            prompt_token_ids=request.all_token_ids[start_ptr:end_ptr],
            mm_inputs=[],
            mm_hashes=[],
            mm_positions=[],
            sampling_params=sub_sampling_params,
            pooling_params=None,
            block_ids=tuple(sub_block_ids),
            num_computed_tokens=0,
            lora_request=None,
        )
    else:   # vllm=0.11.1
        new_request = NewRequestData(
            req_id=new_request_id,
            prompt_token_ids=request.all_token_ids[start_ptr:end_ptr],
            mm_features=[],
            sampling_params=sub_sampling_params,
            pooling_params=None,
            block_ids=tuple(sub_block_ids),
            num_computed_tokens=0,
            lora_request=None,
            prompt_embeds=None
        )
    if hasattr(new_request, "prefill_token_ids"):
        new_request.prefill_token_ids = new_request.prompt_token_ids
    
    # Inject start_ptr for visualization
    try:
        setattr(new_request, "_daric_start_ptr", start_ptr)
    except Exception:
        pass
    return new_request


def split_post_output_by_max_tokens(
    output: SchedulerOutput,
    max_tokens: int,
    original_req_ids: Optional[set[str]] = None,
) -> List[SchedulerOutput]:
    if max_tokens <= 0 or output.total_num_scheduled_tokens <= max_tokens:
        return [output]

    ordered_req_ids: List[str] = []
    seen_req_ids = set()

    for req in output.scheduled_new_reqs:
        req_id = req.req_id
        if req_id in output.num_scheduled_tokens and req_id not in seen_req_ids:
            ordered_req_ids.append(req_id)
            seen_req_ids.add(req_id)

    if not isinstance(output.scheduled_cached_reqs, list):
        for req_id in output.scheduled_cached_reqs.req_ids:
            if req_id in output.num_scheduled_tokens and req_id not in seen_req_ids:
                ordered_req_ids.append(req_id)
                seen_req_ids.add(req_id)

    for req_id in output.num_scheduled_tokens:
        if req_id not in seen_req_ids:
            ordered_req_ids.append(req_id)
            seen_req_ids.add(req_id)

    ordered_recompute_req_ids = [
        req_id for req_id in ordered_req_ids if req_id.endswith("_r")
    ]
    ordered_normal_req_ids = [
        req_id for req_id in ordered_req_ids if not req_id.endswith("_r")
    ]
    ordered_req_ids = ordered_recompute_req_ids + ordered_normal_req_ids

    if original_req_ids is not None:
        last_chunk = [
            req_id for req_id in ordered_req_ids if req_id in original_req_ids
        ]
        sub_req_ids = [
            req_id for req_id in ordered_req_ids if req_id not in original_req_ids
        ]
    else:
        last_chunk = []
        sub_req_ids = ordered_req_ids

    split_req_id_chunks: List[List[str]] = []
    cur_chunk: List[str] = []
    cur_chunk_tokens = 0

    for req_id in sub_req_ids:
        req_tokens = output.num_scheduled_tokens.get(req_id, 0)
        if req_tokens <= 0:
            continue

        if cur_chunk and cur_chunk_tokens + req_tokens > max_tokens:
            split_req_id_chunks.append(cur_chunk)
            cur_chunk = []
            cur_chunk_tokens = 0

        cur_chunk.append(req_id)
        cur_chunk_tokens += req_tokens

    if cur_chunk:
        split_req_id_chunks.append(cur_chunk)

    if last_chunk:
        split_req_id_chunks.append(last_chunk)

    if len(split_req_id_chunks) <= 1:
        return [output]

    original_meta = output.kv_connector_metadata
    dyn_meta = next((m for m in original_meta.metadata if isinstance(m, DARICConnectorMetadata)), None) if hasattr(original_meta, "metadata") else original_meta
    
    split_outputs: List[SchedulerOutput] = []
    unassigned_finished_ids = set(output.finished_req_ids)

    for req_id_chunk in split_req_id_chunks:
        req_id_set = set(req_id_chunk)
        split_output = build_empty_scheduler_output(output)
        split_meta = split_output.kv_connector_metadata
        split_dyn_meta = next((m for m in split_meta.metadata if isinstance(m, DARICConnectorMetadata)), None) if hasattr(split_meta, "metadata") else split_meta

        split_output.scheduled_new_reqs = [
            req for req in output.scheduled_new_reqs if req.req_id in req_id_set
        ]
        split_output.num_scheduled_tokens = {
            req_id: output.num_scheduled_tokens[req_id]
            for req_id in req_id_chunk
            if req_id in output.num_scheduled_tokens
        }
        split_output.total_num_scheduled_tokens = sum(
            split_output.num_scheduled_tokens.values()
        )

        if isinstance(output.scheduled_cached_reqs, list):
            split_output.scheduled_cached_reqs = []
        else:
            # Build only the requested subset in a single O(n) pass instead of
            # deep-copying the whole structure and removing items one-by-one
            # (which was O(n^2) per chunk).
            split_output.scheduled_cached_reqs = rebuild_cached_reqs_subset(
                output.scheduled_cached_reqs,
                req_id_set,
            )

        split_dyn_meta.requests = {
            req_id: copy.copy(req_meta)
            for req_id, req_meta in dyn_meta.requests.items()
            if req_id in req_id_set
        }

        split_output.finished_req_ids = set()
        for finished_req_id in list(unassigned_finished_ids):
            if finished_req_id in req_id_set:
                split_output.finished_req_ids.add(finished_req_id)
                unassigned_finished_ids.remove(finished_req_id)

        split_outputs.append(split_output)

    if split_outputs:
        split_outputs[-1].finished_req_ids.update(unassigned_finished_ids)

    return split_outputs

