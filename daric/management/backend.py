import torch
from typing import List, Dict, TYPE_CHECKING

from .storage import LocalStorage
from ..configs import LocalStoreConfig, ProcessLocalConfig
from ..logging import get_logger

if TYPE_CHECKING:
    from ..reuse_controller.kvcache import KVCache

logger = get_logger(__name__)


class StorageBackend:
    def __init__(
        self, config: LocalStoreConfig
    ) -> None:
        self.config = config
        self.kv_role = ProcessLocalConfig.kv_role
        self.storage = LocalStorage(config)
    
    def register_kvcaches(self, kvcaches: Dict[str, torch.Tensor]) -> None:
        self.storage.register_kvcaches(kvcaches)

    def wait_write_acks(self, cache_ids: set[str] | None = None) -> set:
        return self.storage.wait_write_acks(cache_ids)

    def get_finished_requests(
        self,
        finish_request_info: dict[str, set[str]],
        report_finished: bool,
    ) -> set[str]:
        if finish_request_info:
            self.storage.register_write_finish_info(finish_request_info)

        if not report_finished:
            return set()

        return self.storage.report_write_finish()

    def submit_write_task(self) -> set:
        return self.storage.submit_write_task()

    def write(
        self,
        metas: List["KVCache"],
    ) -> None:
        self.storage.batch_write(metas)

    def read(
        self,
        metas: List["KVCache"]
    ) -> List[List[torch.Tensor]]:
        return self.storage.batch_read(metas)
    
    def submit_layer_write(
        self, metas: List["KVCache"], layer_name: str
    ) -> None:
        self.storage.submit_layer_write(metas, layer_name)

    def layer_write_finish(self, layer_name: str) -> None:
        self.storage.layer_write_finish(layer_name)

    def submit_layer_read(
        self, metas: List["KVCache"], layer_name: str
    ) -> None:
        self.storage.submit_layer_read(metas, layer_name)

    def layer_read_finish(self, layer_name: str) -> None:
        self.storage.layer_read_finish(layer_name)

    def close(self) -> None:
        self.storage.close()

