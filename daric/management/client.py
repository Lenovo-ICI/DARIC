import os
import torch
from typing import Optional

from .service.zmq_rpc import ZMQRPCClient, ZMQRPCProxy
from ..configs import ProcessLocalConfig
from ..logging import get_logger
from ..utils import export_kv_tensor, make_interprocess_event, export_event, import_event

logger = get_logger(__name__)


class LocalStorageClient:
    def __init__(
            self,
            host: str = '127.0.0.1',
            port: int = 50000,
        ):
        
        self._rpc_client = ZMQRPCClient(host, port, timeout=500.0)
        self.storage_mgr = ZMQRPCProxy(self._rpc_client, "store")

        self.shm_size = self.storage_mgr.get_storage_space()
        print(f"Connected to storage (KV IPC mode) at {host}:{port}, size {self.shm_size / (1024**3):.2f} GB")

    def register_layer_info(self):
        self._layer_names = ProcessLocalConfig.cache_topo.get_all_layer_names()
        self._group_idxs = [ProcessLocalConfig.cache_topo.layers[l_name].group_idx for l_name in self._layer_names]
        self._group_idxs_map = {l_name: ProcessLocalConfig.cache_topo.layers[l_name].group_idx for l_name in self._layer_names}
        self._layer_bpbs = [ProcessLocalConfig.cache_topo.layers[l_name].block_byte_size for l_name in self._layer_names]
        if ProcessLocalConfig.is_deepseek_mla:
            self.key_rank = ProcessLocalConfig.get_local_mla_saver_rank()
        else:
            self.key_rank = ProcessLocalConfig.rank

    def register_kv_caches(self, kvcaches: dict):
        layer_names = self._layer_names
        cache_topo = ProcessLocalConfig.cache_topo
        ipc_wrappers = [export_kv_tensor(kvcaches[name]) for name in layer_names]
        group_idxs = [cache_topo.layers[name].group_idx for name in layer_names]
        layer_bpbs = [cache_topo.layers[name].block_byte_size for name in layer_names]
        page_firsts = [cache_topo.layers[name].page_first for name in layer_names]
        self.storage_mgr.register_kv_cache(
            ProcessLocalConfig.rank, ipc_wrappers, layer_names,
            group_idxs, layer_bpbs, page_firsts, os.getpid(),
        )

    def prepare_store_request(self, cache_ids, layer_name):
        dev = torch.cuda.current_device()
        ev = make_interprocess_event(dev)
        ev.record(torch.cuda.current_stream())
        keys = self._resolve_keys_list(cache_ids, layer_name)
        return keys, export_event(ev), ev

    def submit_store_prepared(
        self, keys, block_ids_list, layer_name, saved_positions,
        producer_event_handle,
    ) -> None:
        self.storage_mgr.submit_store(
            ProcessLocalConfig.rank, keys, block_ids_list, layer_name,
            saved_positions, producer_event_handle,
        )

    def submit_load(self, cache_ids, block_ids_list, layer_name):
        dev = torch.cuda.current_device()
        ev = make_interprocess_event(dev)
        ev.record(torch.cuda.current_stream())
        keys = self._resolve_keys_list(cache_ids, layer_name)
        comp_handle, saved = self.storage_mgr.submit_load(
            ProcessLocalConfig.rank, keys, block_ids_list, layer_name,
            export_event(ev),
        )
        return import_event(comp_handle, dev), saved, ev

    def unregister_kv_cache(self):
        self.storage_mgr.unregister_kv_cache(ProcessLocalConfig.rank)

    def get_storage_space(self) -> int:
        return self.storage_mgr.get_storage_space()

    def _resolve_keys_list(self, keys: list[int | str], layer_name: Optional[str] = None) -> list[str]:
        if layer_name is not None:
            layer_idx = self._group_idxs_map[layer_name]
            return [f"{key}_{self.key_rank}_{layer_idx}" for key in keys]
        return [f"{key}_{self.key_rank}" for key in keys]

    def close(self, unregister_kv_cache: bool = True):
        if unregister_kv_cache:
            try:
                self.unregister_kv_cache()
            except Exception:
                logger.exception(
                    "Failed to unregister rank %s KV cache from local storage",
                    ProcessLocalConfig.rank,
                )
        if hasattr(self, '_rpc_client'):
            self._rpc_client.close()

    def clean(self):
        self.storage_mgr.clean()