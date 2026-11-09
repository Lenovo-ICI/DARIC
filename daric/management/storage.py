import queue
import time
import torch
from collections import Counter
from typing import List, TYPE_CHECKING, Dict
from concurrent.futures import ThreadPoolExecutor

from .client import LocalStorageClient
from ..logging import get_logger
from ..configs import ProcessLocalConfig
from ..reuse_controller.kvcache import get_partition_cache_id

logger = get_logger(__name__)

if TYPE_CHECKING:
    from ..configs import LocalStoreConfig
    from ..reuse_controller.kvcache import KVCache


class LocalStorage:
    def __init__(self, 
        config: "LocalStoreConfig",
    ) -> None:
        self.client = LocalStorageClient(config.host, config.port)

        self.config = config
        self.pending_finish_requests: dict[str, set[str]] = {}
        self.read_comp_events: Dict[str, "torch.cuda.Event"] = {}
        self.pending_write_counts: Counter = Counter()
        self._write_completions = queue.Queue()
        self._write_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="daric-write-ack",
        )
        self._write_client: LocalStorageClient | None = None

    @property
    def _partitioned(self) -> bool:
        return bool(ProcessLocalConfig.cache_topo.get_mamba_group_idxs())

    def _expand_partition_payload(self, metas: List["KVCache"]):
        if not self._partitioned:
            cache_ids, block_ids_list = self._metas_to_payload(metas)
            return cache_ids, cache_ids, block_ids_list, metas

        logical_ids, storage_ids, block_ids_list, expanded_metas = [], [], [], []
        num_groups = len(ProcessLocalConfig.cache_topo.groups)
        for meta in metas:
            selected_group_idxs = (
                set(range(num_groups))
                if meta.io_group_idxs is None
                else set(meta.io_group_idxs)
            )
            for partition, partition_group_idxs in (
                ProcessLocalConfig.cache_topo.get_cache_partitions()
            ):
                included_group_idxs = selected_group_idxs.intersection(
                    partition_group_idxs
                )
                if not included_group_idxs:
                    continue
                logical_ids.append(meta.cache_id)
                storage_ids.append(
                    get_partition_cache_id(meta.cache_id, partition)
                )
                block_ids_list.append(tuple(
                    meta.block_ids[idx] if idx in included_group_idxs else []
                    for idx in range(num_groups)
                ))
                expanded_metas.append(meta)
        return logical_ids, storage_ids, block_ids_list, expanded_metas

    def _schedule_write(self, metas: List["KVCache"]) -> List["KVCache"]:
        if not ProcessLocalConfig.is_deepseek_mla:
            return metas
        keep = set(ProcessLocalConfig.mla_write_keep_indices(len(metas)))
        return [meta for i, meta in enumerate(metas) if i in keep]

    def _metas_to_payload(self, metas: List["KVCache"]):
            cache_ids = [m.cache_id for m in metas]
            block_ids_list = []
            for m in metas:
                block_ids_list.append([bl for bl in m.block_ids])
            return cache_ids, block_ids_list

    def register_kvcaches(self, kvcaches: dict[str, torch.Tensor]) -> None:
        self.kvcaches = kvcaches
        self.client.register_layer_info()
        self.client.register_kv_caches(kvcaches)
        self._write_client = LocalStorageClient(
            self.config.host,
            self.config.port,
        )

    def _submit_write(
        self,
        cache_ids: list[str],
        block_ids_list: list,
        layer_name: str | None,
        saved_positions: list[int],
        storage_cache_ids: list[str] | None = None,
    ) -> None:
        keys, producer_event_handle, producer_event = self.client.prepare_store_request(
            storage_cache_ids or cache_ids, layer_name
        )
        self.pending_write_counts.update(cache_ids)
        self._write_executor.submit(
            self._write_and_notify,
            cache_ids,
            keys,
            block_ids_list,
            layer_name,
            saved_positions,
            producer_event_handle,
            producer_event,
        )

    def _write_and_notify(
        self,
        cache_ids: list[str],
        keys: list[str],
        block_ids_list: list,
        layer_name: str | None,
        saved_positions: list[int],
        producer_event_handle: bytes,
        producer_event=None,
    ) -> None:
        error = None
        try:
            assert self._write_client is not None
            self._write_client.submit_store_prepared(
                keys,
                block_ids_list,
                layer_name,
                saved_positions,
                producer_event_handle,
            )
        except Exception as exc:
            error = exc
        finally:
            producer_event = None
        self._write_completions.put((cache_ids, error))

    def batch_read(self, metas: List["KVCache"]) -> None:
        if len(metas) == 0:
            return
        _, storage_ids, block_ids_list, expanded_metas = \
            self._expand_partition_payload(metas)
        comp, saved, producer_event = self.client.submit_load(storage_ids, block_ids_list, None)
        comp.synchronize()
        del producer_event
        for m, s in zip(expanded_metas, saved):
            m.saved_pos = s

    def batch_write(self, metas: List["KVCache"]) -> None:
        metas = self._schedule_write(metas)
        if len(metas) == 0:
            return
        cache_ids, storage_ids, block_ids_list, expanded_metas = \
            self._expand_partition_payload(metas)
        saved_positions = [m.saved_pos for m in expanded_metas]
        self._submit_write(
            cache_ids, block_ids_list, None, saved_positions, storage_ids
        )

    def poll_write_acks(self) -> set:
        newly = set()
        first_error = None
        while True:
            try:
                cache_ids, error = self._write_completions.get_nowait()
            except queue.Empty:
                break
            for cache_id in cache_ids:
                self.pending_write_counts[cache_id] -= 1
                if self.pending_write_counts[cache_id] == 0:
                    del self.pending_write_counts[cache_id]
                    newly.add(cache_id)
            if error is not None and first_error is None:
                first_error = error
        if first_error is not None:
            raise RuntimeError("Background KV cache write failed") from first_error
        return newly

    def register_write_finish_info(
        self, finish_request_info: dict[str, set[str]]
    ) -> None:
        for request_id, cache_ids in finish_request_info.items():
            self.pending_finish_requests.setdefault(request_id, set()).update(
                cache_ids
            )

    def report_write_finish(self) -> set[str]:
        self.poll_write_acks()
        finished_request_ids = {
            request_id
            for request_id, cache_ids in self.pending_finish_requests.items()
            if cache_ids.isdisjoint(self.pending_write_counts)
        }
        for request_id in finished_request_ids:
            self.pending_finish_requests.pop(request_id)
        return finished_request_ids

    def wait_write_acks(self, cache_ids: set[str] | None = None) -> set:
        target_cache_ids = None if cache_ids is None else set(cache_ids)
        newly = set()
        while self.pending_write_counts and (
            target_cache_ids is None
            or not target_cache_ids.isdisjoint(self.pending_write_counts)
        ):
            newly |= self.poll_write_acks()
            if self.pending_write_counts and (
                target_cache_ids is None
                or not target_cache_ids.isdisjoint(self.pending_write_counts)
            ):
                time.sleep(0.0005)
        return newly

    def submit_layer_write(self,
        metas: List["KVCache"],
        layer_name: str
    ) -> None:
        metas = self._schedule_write(metas)
        if len(metas) == 0:
            return
        cache_ids, block_ids_list = self._metas_to_payload(metas)
        saved_positions = [m.saved_pos for m in metas]
        self._submit_write(
            cache_ids, block_ids_list, layer_name, saved_positions
        )

    def layer_write_finish(self, layer_name: str) -> None:
        return

    def submit_layer_read(self,
        metas: List["KVCache"],
        layer_name: str
    ) -> None:
        group_idx = ProcessLocalConfig.cache_topo.get_group_idx(layer_name)
        metas = [
            meta for meta in metas
            if meta.io_group_idxs is None or group_idx in meta.io_group_idxs
        ]
        if len(metas) == 0:
            return
        cache_ids, block_ids_list = self._metas_to_payload(metas)
        comp, saved, producer_event = self.client.submit_load(cache_ids, block_ids_list, layer_name)
        self.read_comp_events[layer_name] = (comp, producer_event)
        for m, s in zip(metas, saved):
            m.saved_pos = s

    def layer_read_finish(self, layer_name: str) -> None:
        if layer_name in self.read_comp_events:
            comp, producer_event = self.read_comp_events.pop(layer_name)
            comp.synchronize()
            del producer_event

    def clean(self) -> None:
        self.client.clean()

    def close(self) -> None:
        self._write_executor.shutdown(wait=True)
        if self._write_client is not None:
            self._write_client.close(unregister_kv_cache=False)
        self.client.close()