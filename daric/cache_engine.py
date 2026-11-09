import os
import threading
import time
import torch
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Optional, List, TYPE_CHECKING

from .logging import get_logger
from .reuse_controller import set_searcher_planner, LookUpResult, ChunkStatus
from .reuse_controller.searcher import SearchResult
from .reuse_controller.kvcache import KVCache
from .adaption import Adapter
from .configs import ProcessLocalConfig
from .management.backend import StorageBackend

if TYPE_CHECKING:
    from .configs import CacheEngineConfig

logger = get_logger(__name__)


class CacheEngineScheduler:
    def __init__(self, config: "CacheEngineConfig"):
        self.config = config
        self.engine_id = ProcessLocalConfig.engine_id
        self.kv_role = ProcessLocalConfig.kv_role
        self.group_block_sizes = ProcessLocalConfig.cache_topo.get_all_group_block_size()
        self.searcher, self.planner = set_searcher_planner(
            reuse_mode=self.config.reuse_mode,
            config=self.config.backend_config,
            max_chunk_len=self.config.max_chunk_len,
            strict_pic=self.config.strict_pic
        )
        self.external_metas: dict[str, LookUpResult] = {}

        self._lookup_executor: Optional[ThreadPoolExecutor] = None
        self._lookup_tls = threading.local()
        self._lookup_thread_searchers: list = []
        self._pending_lookups: dict[str, Future] = {}

    def _get_group_block_range(
        self,
        start: int,
        end: int,
        group_idx: int,
    ) -> tuple[int, int]:
        block_size = self.group_block_sizes[group_idx]
        group = ProcessLocalConfig.cache_topo.groups[group_idx]
        is_mamba = ProcessLocalConfig.cache_topo.layers[group[0]].is_mamba
        if is_mamba:
            if end - start != block_size or end % block_size != 0:
                raise ValueError(
                    f"Mamba cache chunk [{start}, {end}) must end on and "
                    f"span exactly one state block of size {block_size}."
                )
            return end // block_size - 1, end // block_size
        return start // block_size, end // block_size + int(end % block_size != 0)
    
    def get_external_meta(self, request_id: str) -> LookUpResult:
        if not request_id in self.external_metas.keys():
            raise ValueError(f"Request id {request_id} not found in external metas.")
        return self.external_metas[request_id]
    
    def pop_external_meta(self, request_id: str):
        if not request_id in self.external_metas.keys():
            logger.warning(f"Request id {request_id} not found in external metas when popping.")
            return None
        return self.external_metas.pop(request_id)

    def _submit_lookup(self,
            prompt_token_ids: List[int],
            chunk_ptr: List[int],
            internal_hit_length: int
        ) -> Future:
        if self._lookup_executor is None:
            self._lookup_executor = ThreadPoolExecutor(
                max_workers=4,
                thread_name_prefix="daric-lookup"
            )
        chunk_ptr = list(chunk_ptr)
        prompt_len = len(prompt_token_ids)

        def _job() -> tuple[SearchResult, LookUpResult]:
            pair = getattr(self._lookup_tls, "searcher_planner", None)
            if pair is None:
                pair = set_searcher_planner(
                    reuse_mode=self.config.reuse_mode,
                    config=self.config.backend_config,
                    max_chunk_len=self.config.max_chunk_len,
                    strict_pic=self.config.strict_pic
                )
                self._lookup_tls.searcher_planner = pair
                self._lookup_thread_searchers.append(pair[0])
            searcher, planner = pair
            search_result = searcher.search(prompt_token_ids, chunk_ptr)
            lookup_result = planner.plan(
                internal_hit_length, search_result.hit_length, search_result)
            if lookup_result.hit_length == prompt_len:
                # Perfect external hit, reduce one token to avoid full hit case
                lookup_result.hit_length -= 1
            return search_result, lookup_result

        return self._lookup_executor.submit(_job)

    def submit_lookup(self,
            request_id: str,
            prompt_token_ids: List[int],
            chunk_ptr: List[int],
            internal_hit_length: int = 0
        ) -> Optional[LookUpResult]:
        entry = self._pending_lookups.get(request_id)
        if entry is None:
            if internal_hit_length > len(prompt_token_ids):
                raise RuntimeError(f"Internal hit length {internal_hit_length} is greater than prompt length {len(prompt_token_ids)} for request {request_id}.")
            future = self._submit_lookup(
                prompt_token_ids, chunk_ptr, internal_hit_length)
            self._pending_lookups[request_id] = (future, internal_hit_length)
            return None
        future, used_internal_hit = entry
        if not future.done():
            return None
        self._pending_lookups.pop(request_id)
        try:
            search_result, lookup_result = future.result()
        except Exception as e:
            # On lookup failure, fall back to "no external hit" so the
            # request can still be scheduled normally.
            logger.warning(
                f"Async lookup for request {request_id} failed: {e}. "
                "Falling back to no external hit."
            )
            search_result = SearchResult(
                kvcache_metas=[], hit_status=[],
                group_ptr=[0, len(prompt_token_ids)], hit_length=0,
            )
            return self._plan_and_register(
                request_id, search_result, internal_hit_length,
                len(prompt_token_ids))
        if used_internal_hit != internal_hit_length:
            lookup_result = self.planner.plan(
                internal_hit_length, search_result.hit_length, search_result)
            if lookup_result.hit_length == len(prompt_token_ids):
                lookup_result.hit_length -= 1
        self.external_metas[request_id] = lookup_result
        return lookup_result

    def cancel_pending_lookup(self, request_id: str) -> None:
        """Best-effort cancel of an in-flight lookup (request finished/aborted)."""
        entry = self._pending_lookups.pop(request_id, None)
        if entry is not None:
            entry[0].cancel()

    def _plan_and_register(self,
            request_id: str,
            search_result: SearchResult,
            internal_hit_length: int,
            prompt_len: int
        ) -> LookUpResult:
        lookup_result = self.planner.plan(
            internal_hit_length, search_result.hit_length, search_result)
        if lookup_result.hit_length == prompt_len:
            # Perfect external hit, reduce one token to avoid full hit case
            lookup_result.hit_length -= 1
        self.external_metas[request_id] = lookup_result
        return lookup_result

    def update_metas(self, 
            request_id: str,
            all_token_ids: List[int],
            block_ids: tuple[list[int], ...], 
            hist_length: int = 0
        ) -> None:
        def _prepare_decode_cache(
            request_id: str,
            all_token_ids: List[int],
        ) -> List[KVCache]:
            if request_id not in self.external_metas:
                raise RuntimeError(f"Request id {request_id} not found in external metas when preparing decode cache.")
            external_meta = self.external_metas[request_id]
            if len(external_meta.chunk_chain) == 0:
                previous_cached_len = 0
            else:
                previous_cached_len = max([meta.ptr[1] for meta in external_meta.chunk_chain])
            assert previous_cached_len <= len(all_token_ids), f"Previous cached length {previous_cached_len} is greater than current token length {len(all_token_ids)} for request {request_id}."
            if len(all_token_ids) - previous_cached_len < self.config.max_chunk_len:
                # Not enough new tokens to form a full chunk, skip caching.
                return []
            new_chunks: set[str] = set()
            group_start = external_meta.group_ptr[-2]
            if len(external_meta.chunk_chain) == 0 or external_meta.chunk_chain[-1].ptr[1] <= group_start:
                parent = self.searcher.system_prefix
            else:
                parent = external_meta.chunk_chain[-1].cache_id
            from .utils import get_hash_id, token_ids_to_bytes
            start = previous_cached_len
            while start + self.config.max_chunk_len <= len(all_token_ids):
                end = start + self.config.max_chunk_len
                token_split = all_token_ids[start:end]
                cache_id = get_hash_id(token_ids_to_bytes(token_split), prefix=parent.encode())
                meta = KVCache(
                    cache_id=cache_id,
                    ptr=[start, end],
                )
                new_chunks.add(meta.cache_id)
                external_meta.chunk_chain.append(meta)
                external_meta.chunk_status.append(ChunkStatus.WRITE)
                parent = cache_id
                start = end
            return new_chunks
    
        def _update_meta(meta: KVCache, hist_length: int, new_chunk: bool=False):
            start_, end_ = meta.ptr
            if meta.block_ids is not None and hist_length >= end_ and not meta.require_io:
                return
            for block_size in self.group_block_sizes:
                if start_ % block_size != 0:
                    raise ValueError(f"Meta ptr {meta.ptr} is not aligned with block size {block_size}.")
            if hist_length >= end_ and meta.require_io and not new_chunk:
                meta.require_io = False
            block_ranges = [
                self._get_group_block_range(start_, end_, group_idx)
                for group_idx in range(len(self.group_block_sizes))
            ]
            start_block_idxs = [block_range[0] for block_range in block_ranges]
            end_block_idxs = [block_range[1] for block_range in block_ranges]
            if any(end_block_idxs[i] > len(block_ids[i]) for i in range(len(self.group_block_sizes))):
                # The block has not been fully assigned during the chunk prefill.
                return
            aligned_end = max([end_block_idxs[i] * self.group_block_sizes[i] for i in range(len(self.group_block_sizes))])
            meta.ptr = [start_, aligned_end]
            meta.block_ids = tuple([[bids[i] for i in range(start_block_idxs[i], end_block_idxs[i])] for i, bids in enumerate(block_ids)])
        new_chunks = set()
        if self.config.decode_cache_enable:
            new_chunks = _prepare_decode_cache(request_id, all_token_ids)
        req_meta = self.external_metas[request_id]
        for meta in req_meta.chunk_chain:
            _update_meta(meta, hist_length, new_chunk=meta.cache_id in new_chunks)

    def close(self) -> None:
        time0 = time.perf_counter()
        for future, _ in self._pending_lookups.values():
            future.cancel()
        self._pending_lookups.clear()
        if self._lookup_executor is not None:
            # Don't wait for in-flight RPCs; queued lookups are cancelled.
            self._lookup_executor.shutdown(wait=False, cancel_futures=True)
            self._lookup_executor = None
        time1 = time.perf_counter()
        logger.debug(
            f"Scheduler-side cache engine [engine_id={self.engine_id}] closed in {(time1 - time0)*1000:.2f}ms."
        )


class CacheEngineWorker:
    def __init__(self, config: "CacheEngineConfig"):
        self.config = config
        self.kv_role = ProcessLocalConfig.kv_role
        self.engine_id = ProcessLocalConfig.engine_id

        self.adapter = Adapter(config.adaption_config)
        self.cache_manager = StorageBackend(config.backend_config)
        if os.environ.get("DARIC_CLEAN_CACHE_ON_START", "0") == "1" and ProcessLocalConfig.rank == 0:
            self.cache_manager.storage.clean()

    def register_kvcaches(self, kvcaches: dict[str, torch.Tensor]) -> None:
        self.adapter.register_kvcaches(kvcaches)
        self.cache_manager.register_kvcaches(kvcaches)

    def get_kvcaches(self) -> dict[str, torch.Tensor]:
        return self.adapter.kvcaches
    
    def prepare_io_metas(self, 
            write_meta_list: list[list["KVCache"]], 
            read_meta_list: list[list["KVCache"]], 
        ) -> tuple[list["KVCache"], list["KVCache"]]:
        write_metas = self._prepare_io_batch(write_meta_list, True)
        read_metas = self._prepare_io_batch(read_meta_list, False)
        return write_metas, read_metas

    def _prepare_io_batch(self, meta_list: list[list["KVCache"]], no_duplicate=False):
        if len(meta_list) == 0:
            return []
        metas, seen_ids = [], []
        for req_metas in meta_list:
            for meta in req_metas:
                if meta.cache_id in seen_ids:
                    continue
                if no_duplicate:
                    seen_ids.append(meta.cache_id)
                if meta.block_ids is None:
                    # The blocks have not been assigned due to the chunk prefill
                    continue
                metas.append(meta)
        return metas

    def write(self, metas: List["KVCache"]) -> None:
        time0 = time.perf_counter()
        self.cache_manager.write(metas)
        time1 = time.perf_counter()
        logger.debug(f"Write {len(metas)} kvcache chunks time: {(time1 - time0)*1000:.2f}ms.")

    def wait_write_acks(self, cache_ids: Optional[set[str]] = None) -> set:
        """Block until all outstanding async (fire-and-forget) writes have
        been acked by the store, i.e. the GPU blocks are safe to mutate."""
        return self.cache_manager.wait_write_acks(cache_ids)

    def get_finished_requests(
        self,
        finish_request_info: dict[str, set[str]],
        report_finished: bool,
    ) -> set[str]:
        return self.cache_manager.get_finished_requests(
            finish_request_info, report_finished
        )
        
    def read(self, metas: List["KVCache"]) -> None:
        time0 = time.perf_counter()
        self.cache_manager.read(metas)
        time1 = time.perf_counter()
        logger.debug(f"Read {len(metas)} kvcache chunks time: {(time1 - time0)*1000:.2f}ms.")

    def submit_layer_read(self, metas: List["KVCache"], layer_idx: int) -> None:
        self.cache_manager.submit_layer_read(metas, layer_idx)
        
    def layer_read_finish(self, layer_idx: int) -> None:
        self.cache_manager.layer_read_finish(layer_idx)

    def submit_layer_write(self, metas: List["KVCache"],layer_idx: int) -> None:
        self.cache_manager.submit_layer_write(metas, layer_idx)

    def layer_write_finish(self, layer_idx: int) -> None:
        self.cache_manager.layer_write_finish(layer_idx)

    def close(self) -> None:
        time0 = time.perf_counter()
        self.cache_manager.close()
        time1 = time.perf_counter()
        logger.debug(
            f"Worker-side cache engine [engine_id={self.engine_id}, worker_rank={ProcessLocalConfig.rank}] close in {(time1 - time0)*1000:.2f}ms."
        )
