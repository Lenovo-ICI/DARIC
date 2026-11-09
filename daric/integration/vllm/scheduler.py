import copy
from typing import Any, Optional, TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata
)
from vllm.v1.core.sched.output import SchedulerOutput

from .utils import (
    lazy_register_process_local_config,
    check_pic_available,
    validate_mamba_support,
    RequestTracker,
    RequestStatus,
    DARICConnectorMetadata,
    ReqMeta
)
from ...cache_engine import CacheEngineScheduler, ChunkStatus, LookUpResult
from ...configs import get_engine_config, ProcessLocalConfig
from ...logging import get_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVCacheBlocks,
        Request,
        KVCacheConfig
    )


logger = get_logger(__name__)
 
 
class DARICConnectorScheduler:
    def __init__(self, vllm_config: "VllmConfig", kv_cache_config: "KVCacheConfig"):
        self.vllm_config = vllm_config
        self.kv_cache_config = kv_cache_config
        assert vllm_config.kv_transfer_config is not None
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        kv_connector_extra_config = (
            vllm_config.kv_transfer_config.kv_connector_extra_config
        )
        assert "config_file" in kv_connector_extra_config, "config_file not in kv_connector_extra_config"
        if not ProcessLocalConfig.check_registered():
            lazy_register_process_local_config(vllm_config, kv_cache_config)
        cache_engine_config = get_engine_config(
            kv_connector_extra_config["config_file"],
            self.kv_role,
            check_pic_available(kv_cache_config),
            max(ProcessLocalConfig.cache_topo.get_all_group_block_size())
        )
        validate_mamba_support(vllm_config, cache_engine_config)
        
        self.cache_engine = CacheEngineScheduler(cache_engine_config)
        self._cached_reqs_list: dict[str, Request] = {}
        self._request_trackers: dict[str, RequestTracker] = {}
        self._async_loads: dict[str, LookUpResult] = {}
        self.async_mamba_load = bool(ProcessLocalConfig.cache_topo.get_mamba_group_idxs())

    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> Optional[int]:
        transfer_params = request.kv_transfer_params
        if transfer_params is None:
            transfer_params = {}
        if "chunk_ptr" not in transfer_params:
            transfer_params["chunk_ptr"] = [0, len(request.prompt_token_ids)]

        if len(request.all_token_ids) > len(request.prompt_token_ids):
            transfer_params["chunk_ptr"][-1] = len(request.all_token_ids)
        lookup_result = self.cache_engine.submit_lookup(
            request.request_id,
            request.all_token_ids,
            transfer_params["chunk_ptr"],
            internal_hit_length=num_computed_tokens
        )
        if lookup_result is None:
            # Lookup in flight; ask the scheduler to re-query later.
            return None
        need_to_allocate = lookup_result.hit_length - num_computed_tokens
        return need_to_allocate
 
    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        self._cached_reqs_list[request.request_id] = request
        if self.async_mamba_load and num_external_tokens > 0:
            request_id = request.request_id
            self.cache_engine.update_metas(
                request_id, request.all_token_ids, blocks.get_block_ids(), hist_length=0
            )
            external_meta = self.cache_engine.get_external_meta(request_id)
            selected = external_meta.for_worker()
            load_meta = copy.deepcopy(selected)
            if not any(status == ChunkStatus.READ for status in load_meta.chunk_status):
                raise RuntimeError(f"No allocated external cache chunks for request {request_id}")
            self._async_loads[request_id] = load_meta
            for chunk, status in zip(selected.chunk_chain, selected.chunk_status):
                if status == ChunkStatus.READ:
                    chunk.require_io = False

    def build_connector_meta(
        self, scheduler_output: "SchedulerOutput"
    ) -> KVConnectorMetadata:
        meta = DARICConnectorMetadata()
        meta.async_loads = self._async_loads
        self._async_loads = {}
        meta.preempted_req_ids = set(
            getattr(scheduler_output, "preempted_req_ids", ()) or ()
        )
        for finished_req_id in scheduler_output.finished_req_ids:
            self.cache_engine.cancel_pending_lookup(finished_req_id)
            if finished_req_id in self._request_trackers.keys():
                self.cache_engine.pop_external_meta(finished_req_id)
                request_tracker = self._request_trackers[finished_req_id]
                self._request_trackers.pop(finished_req_id, None)
                self._cached_reqs_list.pop(finished_req_id, None)
            elif finished_req_id in self._cached_reqs_list:
                self.cache_engine.pop_external_meta(finished_req_id)
                self._cached_reqs_list.pop(finished_req_id, None)
        step_pic = False
        for request in scheduler_output.scheduled_new_reqs:
            num_tokens_cur_step = (
                request.num_computed_tokens
                + scheduler_output.num_scheduled_tokens[request.req_id]
            )   # Total tokens can be computed after this step
            can_save = num_tokens_cur_step == len(request.prompt_token_ids)
            request_tracker = RequestTracker.from_new_request(request, num_tokens_cur_step)
            self.cache_engine.update_metas(
                request_id=request.req_id, 
                all_token_ids=self._cached_reqs_list[request.req_id].all_token_ids,
                block_ids=request_tracker.block_ids, 
                hist_length=0
            )
            if can_save:
                request_tracker.mark_cached()

            self._request_trackers[request.req_id] = request_tracker
            external_meta = self.cache_engine.get_external_meta(request.req_id)
            step_pic = step_pic or external_meta.is_groupped
            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                status=RequestStatus.GROUP if external_meta.is_groupped else RequestStatus.PREFIX,
                external_meta=external_meta,
                can_load=True,  # New request always can load
                can_save=can_save,
            )
            meta.requests[request.req_id] = req_meta
        cached_reqs = scheduler_output.scheduled_cached_reqs
        if isinstance(cached_reqs, list):
            raise RuntimeError("Please install vllm > 0.8.5")
        for idx, request_id in enumerate(cached_reqs.req_ids):
            request_tracker = self._request_trackers[request_id]
            external_meta = self.cache_engine.get_external_meta(request_id)
            step_pic = step_pic or external_meta.is_groupped
            num_new_tokens = scheduler_output.num_scheduled_tokens[request_id]
            num_tokens_cur_step = cached_reqs.num_computed_tokens[idx] + num_new_tokens
            if hasattr(cached_reqs, "resumed_req_ids"):
                preemption = request_id in cached_reqs.resumed_req_ids
            else:
                preemption = cached_reqs.resumed_from_preemption[idx]
            num_current_tokens = request_tracker.current_len
            new_block_ids = cached_reqs.new_block_ids[idx]
            can_save = request_tracker.prompt_len == num_tokens_cur_step
            if request_tracker.is_cached:
                can_save = False    # Already saved before
            can_load = True if external_meta.hit_length > num_current_tokens else False
            request_tracker.update(num_tokens_cur_step, new_block_ids, preemption)
            self.cache_engine.update_metas(
                request_id,
                self._cached_reqs_list[request_id].all_token_ids,
                request_tracker.block_ids,
                hist_length=0 if preemption else cached_reqs.num_computed_tokens[idx]
            )
            if can_save:
                request_tracker.mark_cached()

            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                status=RequestStatus.GROUP if external_meta.is_groupped else RequestStatus.PREFIX,
                external_meta=external_meta,
                can_load=can_load, # Cached request do not need to load
                can_save=can_save,
            )
            meta.requests[request_id] = req_meta
        meta.step_pic = step_pic
        return meta

    def request_finished(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        self.cache_engine.cancel_pending_lookup(request.request_id)
        never_scheduled = request.request_id not in self._request_trackers
        if never_scheduled:
            return False, None
        return True, None

    def shutdown(self) -> None:
        self.cache_engine.close()