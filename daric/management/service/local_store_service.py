import os
from typing import List, Union
import uuid

from .index import KVCacheIndexManager
from .local_hybrid_manager import HybridStorageManager
from .local_pinmem_manager import PinMemStorageManager
from ...logging import get_logger

logger = get_logger(__name__)

INDEX_PROTECTED_WINDOW = float(os.environ.get("DARIC_INDEX_PROTECTED_WINDOW", "2.0"))


class LocalStoreManager:
    def __init__(
        self,
        shm_size: int,
        root_dir: str,
        disk_size: int,
    ):
        if shm_size < 0 or disk_size < 0:
            raise ValueError("Storage sizes must be non-negative.")
        if shm_size == 0 and disk_size == 0:
            raise ValueError("Local store requires at least one of --mem-size or --disk-size to be non-zero.")
        self.kv_ipc = None
        self.index_manager = KVCacheIndexManager()
        shm_name = f"daric-local-store-rank-0-{uuid.uuid4().hex}"

        mem_index = self.index_manager.create_index("mem", shm_size, protected_window=INDEX_PROTECTED_WINDOW)
        if disk_size == 0:
            self.storage_manager = PinMemStorageManager(shm_name, shm_size, protected_window=0.0, index=mem_index)
            logger.info("Initialized memory store with size: %.2f GB", shm_size / (1024**3))
            self._maybe_init_kv_ipc()
            return

        mem_size = max(shm_size, 4 * 1024**3)
        if shm_size < mem_size:
            logger.warning(f"Adjusting effective pinned memory size to {mem_size / (1024**3):.2f} GB for buffering.")
        if disk_size < mem_size:
            raise ValueError(f"Local store requires --disk-size [{disk_size / (1024**3):.2f}] GB to be at least the effective pinned memory size [{mem_size / (1024**3):.2f}] GB.")
        if mem_size != shm_size:
            mem_index = self.index_manager.replace_index("mem", mem_size, protected_window=INDEX_PROTECTED_WINDOW)
        disk_index = self.index_manager.create_index("disk", disk_size, protected_window=INDEX_PROTECTED_WINDOW)
        self.storage_manager = HybridStorageManager(shm_name, mem_size, root_dir, disk_size, mem_index=mem_index, disk_index=disk_index)
        logger.info("Initialized local store with mem %.2f GB and disk %.2f GB", mem_size / (1024**3), disk_size / (1024**3))
        self._maybe_init_kv_ipc()

    def _maybe_init_kv_ipc(self):
        from .gpu_kv_io import KVIOManager
        self.kv_io_manager = KVIOManager(self.storage_manager)
        logger.info("KV IPC service enabled (store is now GPU-aware).")

    def register_kv_cache(
        self, rank, ipc_wrappers, layer_names, group_idxs, layer_bpbs,
        page_firsts, owner_pid=None,
    ):
        if self.kv_io_manager is None:
            raise RuntimeError("register_kv_cache called but KV IPC is not enabled on the store")
        return self.kv_io_manager.register_kv_cache(
            rank, ipc_wrappers, layer_names, group_idxs, layer_bpbs,
            page_firsts, owner_pid,
        )

    def submit_store(self, rank, keys, block_ids_list, layer_name, saved_positions, producer_event_handle):
        return self.kv_io_manager.submit_store(rank, keys, block_ids_list, layer_name, saved_positions, producer_event_handle)

    def submit_load(self, rank, keys, block_ids_list, layer_name, producer_event_handle):
        return self.kv_io_manager.submit_load(rank, keys, block_ids_list, layer_name, producer_event_handle)

    def unregister_kv_cache(self, rank):
        if self.kv_io_manager is not None:
            self.kv_io_manager.unregister_kv_cache(rank)

    def get_storage_space(self):
        return self.storage_manager.get_storage_space()

    def lookup(self, keys: List[Union[str, int]]):
        results = [None for _ in keys]
        for index_name in ("mem", "disk"):
            if index_name not in self.index_manager.indexes:
                continue
            index_results = self.index_manager.lookup(index_name, keys)
            for idx, result in enumerate(index_results):
                if results[idx] is None and result is not None:
                    results[idx] = True
        return results

    def clean(self):
        self.storage_manager.clean_local()
        for name in self.index_manager.indexes:
            self.index_manager.clean(name)
        logger.info("Local store cleaned.")

    def close(self):
        if self.kv_io_manager is not None:
            self.kv_io_manager.close()
        return self.storage_manager.close()
