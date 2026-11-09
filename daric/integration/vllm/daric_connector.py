# SPDX-License-Identifier: Apache-2.0
from functools import wraps
from typing import TYPE_CHECKING, Any, Optional

import torch
from .scheduler import DARICConnectorScheduler
from .worker import DARICConnectorWorker

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1,
    KVConnectorMetadata,
    KVConnectorRole,
    SupportsHMA
)
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, MambaSpec

if TYPE_CHECKING:
    from vllm.distributed.kv_transfer.kv_connector.v1.base import (
        KVCacheConfig,
        AttentionMetadata,
        KVCacheBlocks,
        ForwardContext,
        Request,
    )

logger = init_logger(__name__)


class DARICConnector(KVConnectorBase_V1, SupportsHMA):
    def __init__(self, 
        vllm_config: "VllmConfig",
        role: KVConnectorRole,
        kv_cache_config: "KVCacheConfig"):
        super().__init__(vllm_config, role, kv_cache_config)
        if role == KVConnectorRole.WORKER:
            self.daric_engine = DARICConnectorWorker(vllm_config, kv_cache_config)
        else:
            self.daric_engine = DARICConnectorScheduler(vllm_config, kv_cache_config)

    # ==============================
    # Worker-side custom methods
    # ==============================
    def rebuild_scheduler_output(self, scheduler_output: "SchedulerOutput"):
        return self.daric_engine.rebuild_scheduler_output(scheduler_output)

    def register_model_runner(self, model_runner: Any):
        self.daric_engine.register_model_runner(model_runner)
    
    def finish_runner_register(self):
        return self.daric_engine.finish_runner_register()

    def bind_scheduler_output(self, scheduler_output: SchedulerOutput):
        self.daric_engine.bind_scheduler_output(scheduler_output)
    
    def pic_available(self):
        return self.daric_engine.step_pic_available()
    
    # ==============================
    # Worker-side methods
    # ==============================
    def bind_connector_metadata(
            self, connector_metadata: KVConnectorMetadata) -> None:
        self._connector_metadata = connector_metadata
        self.daric_engine.bind_connector_metadata(connector_metadata)

    def handle_preemptions(self, kv_connector_metadata: KVConnectorMetadata):
        self.daric_engine.handle_preemptions(kv_connector_metadata)

    def register_kv_caches(self, kv_caches: dict[str, torch.Tensor]):
        return self.daric_engine.register_kv_caches(kv_caches)

    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        self.daric_engine.start_load_kv(forward_context, **kwargs)

    def wait_for_layer_load(self, layer_name: str) -> None:
        self.daric_engine.wait_for_layer_load(layer_name)

    def save_kv_layer(
        self,
        layer_name: str,
        kv_layer: torch.Tensor,
        attn_metadata: "AttentionMetadata",
        **kwargs,
    ) -> None:
        self.daric_engine.save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)

    def wait_for_save(self):
        self.daric_engine.wait_for_save()

    def get_finished(
        self, finished_req_ids: set[str]
    ) -> tuple[Optional[set[str]], Optional[set[str]]]:
        return self.daric_engine.get_finished(finished_req_ids)
    
    def shutdown(self) -> None:
        return self.daric_engine.shutdown()

    # ==============================
    # Scheduler-side methods
    # ==============================
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        num_matched = self.daric_engine.get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        if num_matched is None:
            return None, False
        return num_matched, bool(num_matched and self.daric_engine.async_mamba_load)

    def update_state_after_alloc(
        self, request: "Request", blocks: "KVCacheBlocks", num_external_tokens: int
    ):
        self.daric_engine.update_state_after_alloc(
            request, blocks, num_external_tokens
        )

    def build_connector_meta(
        self, scheduler_output: SchedulerOutput
    ) -> KVConnectorMetadata:
        return self.daric_engine.build_connector_meta(scheduler_output)

    def request_finished(
        self,
        request: "Request",
        block_ids: list[int],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.daric_engine.request_finished(
            request, (block_ids,)
        )

    def request_finished_all_groups(
        self,
        request: "Request",
        block_ids: tuple[list[int], ...],
    ) -> tuple[bool, dict[str, Any] | None]:
        return self.daric_engine.request_finished(request, block_ids)
