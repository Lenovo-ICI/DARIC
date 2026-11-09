import torch
from typing import Any, List, Optional, Tuple, TYPE_CHECKING

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorMetadata
)
from vllm.v1.core.sched.output import SchedulerOutput, NewRequestData

from .utils import (
    lazy_register_process_local_config, 
    finish_register_process_local_config, 
    remove_from_cached_reqs,
    build_empty_scheduler_output,
    split_post_output_by_max_tokens, 
    rebuild_new_sub_request,
    check_pic_available,
    validate_mamba_support,
    RequestTracker,
    PICMode,
    WorkerStage,
    RequestStatus,
    DARICConnectorMetadata,
    ReqMeta,
)
from ...cache_engine import CacheEngineWorker, ChunkStatus, KVCache
from ...configs import get_engine_config, ProcessLocalConfig
from ...logging import get_logger

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        AttentionMetadata,
        ForwardContext,
        KVCacheConfig
    )


logger = get_logger(__name__)
 
 
class DARICConnectorWorker:
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
        self.pic_mode = PICMode.NONE
        if cache_engine_config.reuse_mode == "independent":
            self.pic_mode = PICMode.STRICT if cache_engine_config.strict_pic else PICMode.FLEXIBLE

        self.cache_engine = CacheEngineWorker(cache_engine_config)
        
        self.model_runner = None    # Worker-side model runner
        self._block_size = vllm_config.cache_config.block_size
        self.layer_wise = cache_engine_config.backend_config.layer_wise
        self.recompute = cache_engine_config.adaption_config.recompute
        self.pending = False
        self._finished_async_loads: set[str] = set()
 
        self._request_trackers: dict[str, RequestTracker] = {}
        self.read_metas, self.write_metas = [], []
        self.pending_finished_requests = set()
        self._finished_reqs = set()
        self._pre_process_write_cache_ids: set[str | int] = set()

    @staticmethod
    def _select_read_groups(metas: list[KVCache]) -> None:
        mamba_group_idxs = ProcessLocalConfig.cache_topo.get_mamba_group_idxs()
        if not metas or not mamba_group_idxs:
            return
        attention_group_idxs = tuple(
            ProcessLocalConfig.cache_topo.get_attention_group_idxs()
        )
        last_meta = max(metas, key=lambda meta: meta.ptr[1])
        for meta in metas:
            if meta is not last_meta:
                meta.io_group_idxs = attention_group_idxs
 
    def _register_rotary_embed(self):
        layer_names = list(self.cache_engine.get_kvcaches().keys())
        if self.pic_mode != PICMode.NONE and not self.cache_engine.adapter.register_done():
            for layer_name in layer_names:
                rotary_embed_layer = (
                    self.model_runner.get_model()
                    .model.layers[int(layer_name.split(".")[2])]
                    .self_attn.rotary_emb
                )
                self.cache_engine.adapter.register_rotary_embed(layer_name, rotary_embed_layer)
            assert self.cache_engine.adapter.register_done()
    
    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        finish_register_process_local_config(self.vllm_config, self.kv_cache_config, kv_caches)
        self.cache_engine.register_kvcaches(kv_caches)

    def bind_connector_metadata(
            self, connector_metadata: KVConnectorMetadata) -> None:
        def _apply_saved_pos(metas: list[KVCache], group_ptr: list[int]):
            if self.pic_mode != PICMode.STRICT:
                for meta in metas:
                    meta.saved_pos = int(meta.ptr[0])
            else:
                for meta in metas:
                    _start = int(meta.ptr[0])
                    if _start >= group_ptr[-2]:
                        meta.saved_pos = _start
                    else:
                        relative_pos = min([_start - gp for gp in group_ptr if _start >= gp])
                        meta.saved_pos = relative_pos

        assert isinstance(connector_metadata, DARICConnectorMetadata)
        self._connector_metadata = connector_metadata

        write_cid_to_tracker: dict[str, RequestTracker] = {}

        def _track_write_ids(tracker, metas: list[KVCache]):
            for m in metas:
                write_cid_to_tracker.setdefault(m.cache_id, tracker)

        if self.layer_wise:
            if len(self.read_metas) > 0 and len(self.write_metas) == 0:
                self.pending = True
                return
       
        read_metas, write_metas = [], []
        for external_meta in connector_metadata.async_loads.values():
            cur_read_metas = [
                meta for meta, status in zip(external_meta.chunk_chain, external_meta.chunk_status)
                if status == ChunkStatus.READ and meta.require_io and meta.block_ids is not None
            ]
            self._select_read_groups(cur_read_metas)
            read_metas.append(cur_read_metas)
        for req_meta in connector_metadata.requests.values():
            request_tracker = self._request_trackers[req_meta.req_id]
            external_meta = req_meta.external_meta
            if not request_tracker.is_cached:
                if request_tracker.prompt_len <= req_meta.current_len:
                    request_tracker.mark_cached()

            status = req_meta.status
            if connector_metadata.stage == WorkerStage.RECOMPUTE:
                continue
            if connector_metadata.stage == WorkerStage.PRE_PROCESS and status == RequestStatus.PREFIX:
                continue
            if connector_metadata.stage in (WorkerStage.POST_PROCESS, WorkerStage.NONE):
                if status == RequestStatus.GROUP:
                    cur_write_metas = [
                        meta for meta, st in zip(external_meta.chunk_chain, external_meta.chunk_status)
                        if st == ChunkStatus.WRITE and meta.require_io and meta.block_ids is not None
                        and (
                            connector_metadata.stage != WorkerStage.POST_PROCESS
                            or meta.cache_id not in self._pre_process_write_cache_ids
                        )
                    ]
                    _apply_saved_pos(cur_write_metas, external_meta.adjusted_group_ptr)
                    _track_write_ids(request_tracker, cur_write_metas)
                    write_metas.append(cur_write_metas)
                    continue

            cur_read_metas = [
                meta for meta, st in zip(external_meta.chunk_chain, external_meta.chunk_status)
                if st == ChunkStatus.READ and meta.require_io and meta.block_ids is not None
            ]
            self._select_read_groups(cur_read_metas)
            read_metas.append(cur_read_metas)

            cur_write_metas = [
                meta for meta, st in zip(external_meta.chunk_chain, external_meta.chunk_status)
                if st == ChunkStatus.WRITE and meta.require_io and meta.block_ids is not None
            ]
            _apply_saved_pos(cur_write_metas, external_meta.adjusted_group_ptr)
            _track_write_ids(request_tracker, cur_write_metas)
            write_metas.append(cur_write_metas)

        self.write_metas, self.read_metas = self.cache_engine.prepare_io_metas(write_metas, read_metas)

        if self.write_metas:
            for i in ProcessLocalConfig.mla_write_keep_indices(len(self.write_metas)):
                tracker = write_cid_to_tracker.get(self.write_metas[i].cache_id)
                if tracker is None:
                    continue
                if tracker.write_cache_ids is None:
                    tracker.write_cache_ids = set()
                tracker.write_cache_ids.add(self.write_metas[i].cache_id)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        assert isinstance(kv_connector_metadata, DARICConnectorMetadata)
        write_cache_ids = set()
        for request_id in kv_connector_metadata.preempted_req_ids:
            tracker = self._request_trackers.get(request_id)
            if tracker is not None and tracker.write_cache_ids:
                write_cache_ids.update(tracker.write_cache_ids)
        if write_cache_ids:
            self.cache_engine.wait_write_acks(write_cache_ids)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        if len(self.read_metas) == 0:
            return

        if not self.layer_wise:
            torch.cuda.synchronize()
            self.cache_engine.read(self.read_metas)
            self._finished_async_loads.update(self._connector_metadata.async_loads)
            if self.pic_mode == PICMode.FLEXIBLE:
                self.cache_engine.adapter.apply_rerotate_load(self.read_metas)
        else:
            if not self.pending:
                layer_names = ProcessLocalConfig.cache_topo.get_all_layer_names()
                for layer_name in layer_names:
                    self.cache_engine.submit_layer_read(self.read_metas, layer_name)

    def wait_for_layer_load(self, layer_name: str) -> None:
        if len(self.read_metas) == 0:
            return
        if self.layer_wise:
            self.cache_engine.layer_read_finish(layer_name)
            if self.pic_mode == PICMode.FLEXIBLE and not self.pending:
                self.cache_engine.adapter.apply_rerotate_load(
                    self.read_metas, layer_name=layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        if len(self.write_metas) == 0:
            return
        if self.layer_wise:
            self.cache_engine.submit_layer_write(
                self.write_metas,
                layer_name
            )

    def wait_for_save(self):
        """Blocking until the KV cache is saved to the connector buffer."""
        if len(self.write_metas) != 0:
            if not self.layer_wise:
                self.cache_engine.write(self.write_metas)
            else:
                layer_names = ProcessLocalConfig.cache_topo.get_all_layer_names()
                for layer_name in layer_names:
                    self.cache_engine.layer_write_finish(layer_name)
                self.read_metas = []
                self.pending = False
            if self._connector_metadata.stage == WorkerStage.PRE_PROCESS:
                self._pre_process_write_cache_ids.update(
                    meta.cache_id for meta in self.write_metas
                )
            if ProcessLocalConfig.cache_topo.get_mamba_group_idxs():
                self.cache_engine.wait_write_acks({
                    meta.cache_id for meta in self.write_metas
                })
        
        if self.pic_mode == PICMode.STRICT:
            ptrs, blocks = [], []
            for req_meta in self._connector_metadata.requests.values():
                if not req_meta.can_save:
                    continue
                status = req_meta.status
                external_meta = req_meta.external_meta
                if self._connector_metadata.stage == WorkerStage.PRE_PROCESS and status == RequestStatus.GROUP:
                    ptrs.append(external_meta.adjusted_group_ptr)
                    blocks.append(self._request_trackers[req_meta.req_id].block_ids)
            if ptrs and self.cache_engine.adapter.rerotate:
                self.cache_engine.wait_write_acks()
            self.cache_engine.adapter.apply_rerotate_multi(ptrs, blocks)

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        finish_request_info = {}
        for request_id in finished_req_ids:
            tracker = self._request_trackers.pop(request_id, None)
            if tracker is None:
                continue
            if request_id in self._finished_async_loads:
                if tracker.write_cache_ids:
                    self.cache_engine.wait_write_acks(tracker.write_cache_ids)
                continue
            has_blocks = bool(tracker.block_ids and any(tracker.block_ids))
            if not has_blocks:
                continue
            finish_request_info[request_id] = tracker.write_cache_ids or set()
        finished_requests = self.cache_engine.get_finished_requests(
            finish_request_info,
            self._connector_metadata.report_finished,
        )
        finished_loads = self._finished_async_loads
        self._finished_async_loads = set()
        return finished_requests or None, finished_loads or None
   
    def shutdown(self) -> None:
        self.cache_engine.close()

    def rebuild_scheduler_output(self, scheduler_output: SchedulerOutput) -> Tuple[SchedulerOutput, List[SchedulerOutput]]:
        def add_recompute_request(
            request: RequestTracker,
            group_ptr: List[int],
            adjusted_group_ptr: List[int],
            post_output: SchedulerOutput,
        ) -> int:
            """build the recompute scheduler output based on the group order"""
            total_groups = len(group_ptr) - 1
            if total_groups <= 1 or not self.recompute:
                return 0
            request_id = request.req_id
            recompute_tokens_list = self.cache_engine.adapter.get_recompute_length(group_ptr)
            assert len(recompute_tokens_list) > 0, f"recompute tokens list is empty for request {request_id}"
            # Only recompute the tokens that are beyond the GPU cached length
            no_recompute_length = adjusted_group_ptr[1]
            recompute_tokens_list = [[start, end] for start, end in recompute_tokens_list if end > no_recompute_length]
            logger.debug(f"Recompute tokens list for request {request_id}: {recompute_tokens_list}")
            last_nums = 0
            cur_group_num = len(recompute_tokens_list)

            for i in range(cur_group_num):
                start_idx, end_idx = recompute_tokens_list[i]
                recompute_tokens = end_idx - start_idx
                if recompute_tokens <= 0:
                    continue

                new_recompute_request = rebuild_new_sub_request(
                    request, f"{i+1}_r", group_ptr[0], end_idx, self._block_size
                )
                new_recompute_request.num_computed_tokens = start_idx
                post_output.scheduled_new_reqs.append(new_recompute_request)
                post_output.num_scheduled_tokens[
                    new_recompute_request.req_id
                ] = recompute_tokens
                post_output.total_num_scheduled_tokens += recompute_tokens
                self.pending_finished_requests.add(new_recompute_request.req_id)
            return last_nums

        last_pending_reqs = self.pending_finished_requests.copy()
        self.pending_finished_requests.clear()
        assert isinstance(scheduler_output.kv_connector_metadata, DARICConnectorMetadata)

        step_pic = scheduler_output.kv_connector_metadata.step_pic
        if step_pic:
            # Split the scheduler output into pre-process and post-process outputs for position-independent generation
            pre_output = build_empty_scheduler_output(scheduler_output, skip_cached_reqs=True)
            post_output = build_empty_scheduler_output(scheduler_output)
            pre_meta: DARICConnectorMetadata = pre_output.kv_connector_metadata
            post_meta: DARICConnectorMetadata = post_output.kv_connector_metadata
            
            pre_meta.stage = WorkerStage.PRE_PROCESS
            post_meta.stage = WorkerStage.POST_PROCESS
            post_meta.preempted_req_ids.clear()
            pre_meta.report_finished = False
            post_meta.report_finished = False
            # Rebuild sub-requests for new scheduled requests
            pre_output.total_num_scheduled_tokens = 0

        for req_num, request in enumerate(scheduler_output.scheduled_new_reqs):
            scheduled_tokens = scheduler_output.num_scheduled_tokens.get(request.req_id, 0)
            hit_length = request.num_computed_tokens
            request_tracker = RequestTracker.from_new_request(request, scheduled_tokens + hit_length)
            self._request_trackers[request.req_id] = request_tracker
            if not step_pic:
                continue

            external_meta = scheduler_output.kv_connector_metadata.requests[request.req_id].external_meta
            # Remove the original new request from the pre_output
            pre_output.scheduled_new_reqs.remove(request)
            pre_output.num_scheduled_tokens.pop(request.req_id, None)
            
            if not external_meta.is_groupped:
                # For prefix hit request, do not need to perform pre-process
                continue
            all_inconsistent_s_ptrs = set(external_meta.inconsistent_s_ptrs or ())
            for group_idx in range(len(external_meta.adjusted_group_ptr)-1):
                start_ptr, end_ptr = external_meta.adjusted_group_ptr[group_idx], external_meta.adjusted_group_ptr[group_idx+1]
                if hit_length >= end_ptr or start_ptr in all_inconsistent_s_ptrs or start_ptr >= hit_length + scheduled_tokens:
                    # Skip full hit sub-requests & inconsistent hit sub-requests & sub-requests that not scheduled in this round
                    continue
                valid_end_ptr = min(end_ptr, hit_length + scheduled_tokens)
                # Generate the last group in a prefix mode for all PIC methods
                if self.pic_mode == PICMode.STRICT and group_idx != len(external_meta.adjusted_group_ptr)-2:
                    new_sub_request = rebuild_new_sub_request(
                        request_tracker, f"{group_idx}_s-{len(external_meta.adjusted_group_ptr)}", start_ptr, end_ptr, self._block_size
                    )
                    new_sub_request.num_computed_tokens = max(0, hit_length - start_ptr)
                    sub_scheduled_tokens = valid_end_ptr - start_ptr - new_sub_request.num_computed_tokens
                else:
                    new_sub_request = rebuild_new_sub_request(
                        request_tracker, f"{group_idx}_s-{len(external_meta.adjusted_group_ptr)}", 0, end_ptr, self._block_size
                    )
                    new_sub_request.num_computed_tokens = max(start_ptr, hit_length)
                    sub_scheduled_tokens = valid_end_ptr - new_sub_request.num_computed_tokens

                pre_output.num_scheduled_tokens[new_sub_request.req_id] = sub_scheduled_tokens
                pre_output.total_num_scheduled_tokens += sub_scheduled_tokens
                pre_output.scheduled_new_reqs.append(new_sub_request)
                post_output.finished_req_ids.add(new_sub_request.req_id)
 
            if scheduled_tokens + hit_length >= len(request.prompt_token_ids):
                # The request can be completed in a single round
                last_num = add_recompute_request(
                    request_tracker,
                    external_meta.group_ptr,
                    external_meta.adjusted_group_ptr,
                    post_output,
                )

                req_idx = [idx for idx, r in enumerate(post_output.scheduled_new_reqs) if r.req_id == request.req_id][0]
                post_output.scheduled_new_reqs[req_idx].num_computed_tokens = (
                    scheduled_tokens + hit_length - 1 - last_num
                )
                post_output.num_scheduled_tokens[request.req_id] = 1 + last_num
                post_output.total_num_scheduled_tokens -= scheduled_tokens - 1 - last_num
            else:
                # For chunked prefill request that not completed in a single round, we need to remove the original request from the post_output
                post_output.scheduled_new_reqs.remove(request)
                post_output.total_num_scheduled_tokens -= scheduled_tokens
                post_output.num_scheduled_tokens.pop(request.req_id, None)
 
        """ need to rebuild the cached requests """
        original_cached_reqs = scheduler_output.scheduled_cached_reqs
        if isinstance(original_cached_reqs, list):
            raise RuntimeError(f"Please install vllm > 0.8.5")
        else:
            for idx, request_id in enumerate(original_cached_reqs.req_ids):
                req_meta = scheduler_output.kv_connector_metadata.requests[request_id]
                external_meta = req_meta.external_meta
                scheduled_tokens = scheduler_output.num_scheduled_tokens.get(request_id, 0)
                num_computed_tokens = original_cached_reqs.num_computed_tokens[idx]
                num_tokens_cur_step = num_computed_tokens + scheduled_tokens
                request_tracker = self._request_trackers[request_id]
                if hasattr(original_cached_reqs, "resumed_req_ids"):
                    preemption = request_id in original_cached_reqs.resumed_req_ids
                else:
                    preemption = original_cached_reqs.resumed_from_preemption[idx]
                request_tracker.update(num_tokens_cur_step, original_cached_reqs.new_block_ids[idx], preemption)
                if not step_pic:
                    continue

                pre_output.num_scheduled_tokens.pop(request_id, None)
                if num_computed_tokens >= req_meta.prompt_len:
                    continue
                if not external_meta.is_groupped:
                    continue
               
                all_inconsistent_s_ptrs = {
                    meta.ptr[0]
                    for meta, st in zip(external_meta.chunk_chain, external_meta.chunk_status)
                    if st == ChunkStatus.READ and meta.ptr[0] >= external_meta.hit_length
                }
                for group_idx in range(len(external_meta.adjusted_group_ptr)-1):
                    start_ptr, end_ptr = external_meta.adjusted_group_ptr[group_idx], external_meta.adjusted_group_ptr[group_idx+1]
                    if num_computed_tokens >= end_ptr or start_ptr in all_inconsistent_s_ptrs or start_ptr >= num_computed_tokens + scheduled_tokens:
                        # Skip full computed sub-requests & inconsistent hit sub-requests & sub-requests that not scheduled in this round
                        continue
                    valid_end_ptr = min(end_ptr, num_computed_tokens + scheduled_tokens)
                    # Generate the last group in a prefix mode for all PIC methods
                    if self.pic_mode == PICMode.STRICT and group_idx != len(external_meta.adjusted_group_ptr)-2:
                        new_sub_request = rebuild_new_sub_request(
                            request_tracker, f"{group_idx}_s-{len(external_meta.adjusted_group_ptr)}", start_ptr, end_ptr, self._block_size
                        )
                        new_sub_request.num_computed_tokens = max(0, num_computed_tokens - start_ptr)
                        sub_scheduled_tokens = valid_end_ptr - start_ptr - new_sub_request.num_computed_tokens
                    else:
                        new_sub_request = rebuild_new_sub_request(
                            request_tracker, f"{group_idx}_s-{len(external_meta.adjusted_group_ptr)}", 0, end_ptr, self._block_size
                        )
                        new_sub_request.num_computed_tokens = max(start_ptr, num_computed_tokens)
                        sub_scheduled_tokens = valid_end_ptr - new_sub_request.num_computed_tokens
                    pre_output.num_scheduled_tokens[new_sub_request.req_id] = sub_scheduled_tokens
                    pre_output.total_num_scheduled_tokens += sub_scheduled_tokens

                    pre_output.scheduled_new_reqs.append(new_sub_request)
                    post_output.finished_req_ids.add(new_sub_request.req_id)
 
                need_recompute = num_computed_tokens < req_meta.prompt_len and num_tokens_cur_step >= req_meta.prompt_len
                if need_recompute:
                    last_num = add_recompute_request(
                        request_tracker,
                        external_meta.group_ptr,
                        external_meta.adjusted_group_ptr,
                        post_output,
                    )
                    if not request_tracker.is_cached:
                        remove_from_cached_reqs(request_id, post_output.scheduled_cached_reqs)
                        post_new_request = rebuild_new_sub_request(
                            request_tracker,
                            None,
                            0,
                            num_tokens_cur_step,    ###
                            self._block_size
                        )
                        post_new_request.num_computed_tokens = (
                            req_meta.prompt_len - 1 - last_num
                        )
                        post_output.scheduled_new_reqs.append(post_new_request)
                        post_output.num_scheduled_tokens[request_id] = num_tokens_cur_step - post_new_request.num_computed_tokens
                        post_output.total_num_scheduled_tokens -= scheduled_tokens - post_output.num_scheduled_tokens[request_id]
                    else:
                        post_output.scheduled_cached_reqs.num_computed_tokens[idx] = req_meta.prompt_len - 1 - last_num
                        post_output.num_scheduled_tokens[request_id] = num_tokens_cur_step - post_output.scheduled_cached_reqs.num_computed_tokens[idx]
                        post_output.total_num_scheduled_tokens -= scheduled_tokens - post_output.num_scheduled_tokens[request_id]
                else:
                    remove_from_cached_reqs(request_id, post_output.scheduled_cached_reqs)
                    post_output.num_scheduled_tokens.pop(request_id, None)
                    post_output.total_num_scheduled_tokens -= scheduled_tokens

        # Handle pending finished requests
        if len(last_pending_reqs) > 0:
            if not step_pic:
                scheduler_output.finished_req_ids.update(last_pending_reqs)
            else:
                if pre_output.total_num_scheduled_tokens > 0:
                    pre_output.finished_req_ids.update(last_pending_reqs)
                elif post_output.total_num_scheduled_tokens > 0:
                    post_output.finished_req_ids.update(last_pending_reqs)
                else:
                    pre_output.finished_req_ids.update(last_pending_reqs)
                    # raise RuntimeError(f"Scheduled tokens are zero for both pre and post outputs!")
        if not step_pic:
            return None, None

        if post_output.total_num_scheduled_tokens == 0:
            self.pending_finished_requests.update(post_output.finished_req_ids)
            pre_meta.report_finished = True
            return pre_output, []

        max_batched_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        post_output_list = split_post_output_by_max_tokens(
            post_output, max_batched_tokens, original_req_ids=set(scheduler_output.num_scheduled_tokens.keys())
        )
        
        for i, post_output_ in enumerate(post_output_list):
            out_meta: DARICConnectorMetadata = post_output_.kv_connector_metadata
            if i != len(post_output_list) - 1:
                out_meta.stage = WorkerStage.RECOMPUTE
            else:
                out_meta.stage = WorkerStage.POST_PROCESS
                out_meta.report_finished = True

        spec_reqs = list(scheduler_output.scheduled_spec_decode_tokens.keys())
        for key in spec_reqs:
            if key not in pre_output.num_scheduled_tokens.keys():
                pre_output.scheduled_spec_decode_tokens.pop(key)
        for post_output_ in post_output_list:
            for key in spec_reqs:
                if key not in post_output_.scheduled_spec_decode_tokens.keys():
                    post_output_.scheduled_spec_decode_tokens.pop(key)
        return pre_output, post_output_list

    def register_model_runner(self, model_runner: Any):
        self.model_runner = model_runner
        self._register_rotary_embed()

    def finish_runner_register(self):
        return self.model_runner is not None

    def bind_scheduler_output(self, scheduler_output: SchedulerOutput):
        self.scheduler_output_step = scheduler_output
        self._pre_process_write_cache_ids.clear()

    def step_pic_available(self):
        return self.pic_mode != PICMode.NONE and self.scheduler_output_step.kv_connector_metadata.step_pic