from typing import List
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .lookup_client import LookupClient
from .kvcache import KVCache
from ..configs import ProcessLocalConfig, LocalStoreConfig
from ..logging import get_logger
from ..utils import get_hash_id, token_ids_to_bytes

logger = get_logger(__name__)


@dataclass
class SearchResult:
    kvcache_metas: List[KVCache]
    hit_status: List[bool]
    group_ptr: List[int] 
    hit_length: int


class Searcher(ABC):
    def __init__(self, 
            config: LocalStoreConfig,
            max_chunk_len: int = 256
        ):
        layerwise = config.layer_wise if hasattr(config, 'layer_wise') else False
        self.kvcache_index_client = LookupClient(
            host=config.host,
            port=config.port,
            layerwise=layerwise,
        )
        self.max_chunk_len = max_chunk_len
        self.block_size = max(ProcessLocalConfig.cache_topo.get_all_group_block_size())
        self.system_prefix = f"{ProcessLocalConfig.model_key}_{ProcessLocalConfig.tp_size}_{ProcessLocalConfig.pp_size}"
        if self.max_chunk_len % self.block_size != 0:
            raise ValueError(f"max_chunk_len ({self.max_chunk_len}) must be a multiple of block_size ({self.block_size})")

    def _split_chunk_ptr(self, ind_ptr: List[int]) -> List[List[int]]:
        sub_chunk_ptr = [0]

        for i in range(len(ind_ptr) - 1):
            chunk_start = ind_ptr[i]
            chunk_end = ind_ptr[i + 1]
            chunk_length = chunk_end - chunk_start
            if chunk_length <= 0:
                continue

            cacheable_length = chunk_length
            if i == len(ind_ptr) - 2:
                cacheable_length -= cacheable_length % self.max_chunk_len

            local_offset = 0
            while local_offset < cacheable_length:
                step = min(self.max_chunk_len, cacheable_length - local_offset)
                sub_chunk_ptr.append(sub_chunk_ptr[-1] + step)
                local_offset += step

        return sub_chunk_ptr, ind_ptr.copy()
    
    @abstractmethod
    def search(
        self,
        prompt_token_ids: List[int],
        ind_ptr: List[int],
    ) -> SearchResult:
        raise NotImplementedError
    
    def close(self):
        self.kvcache_index_client.close()


class PrefixSearcher(Searcher):
    def __init__(self, 
            config: LocalStoreConfig,
            max_chunk_len: int = 256):
        super().__init__(config, max_chunk_len)

    def search(
        self,
        prompt_token_ids: List[int],
        ind_ptr: List[int],
    ) -> SearchResult:
        sub_chunk_ptr, group_ptr = self._split_chunk_ptr(ind_ptr)
        if (
            ProcessLocalConfig.cache_topo.get_mamba_group_idxs()
            and len(sub_chunk_ptr) > 1
            and sub_chunk_ptr[-1] == len(prompt_token_ids)
        ):
            sub_chunk_ptr.pop()
        cache_ids, chunk_lengths, prefix_ids = [], [], [self.system_prefix]
        for i in range(len(sub_chunk_ptr) - 1):
            token_split = prompt_token_ids[sub_chunk_ptr[i] : sub_chunk_ptr[i + 1]]
            cache_id = get_hash_id(token_ids_to_bytes(token_split), prefix=prefix_ids[-1].encode())
            cache_ids.append(cache_id)
            chunk_lengths.append(len(token_split))
            prefix_ids.append(cache_id)
        lookup_results = self.kvcache_index_client.lookup(cache_ids)
        kvcache_metas, hit_status = [], []
        hit_length = 0
        prefix_hit = True
        length_cum = 0
        for i in range(len(cache_ids)):
            if lookup_results[i] and prefix_hit:
                hit_length += chunk_lengths[i]
                hit_status.append(True)
                kvcache_metas.append(
                    KVCache(
                        cache_id=cache_ids[i],
                        ptr=[length_cum, length_cum + chunk_lengths[i]],
                    )
                )
            else:
                prefix_hit = False
                hit_status.append(False)
                kvcache_metas.append(
                    KVCache(
                        cache_id=cache_ids[i],
                        ptr=[length_cum, length_cum + chunk_lengths[i]],
                    )
                )
            length_cum += chunk_lengths[i]
        return SearchResult(
            kvcache_metas=kvcache_metas,
            hit_status=hit_status,
            group_ptr=[0, len(prompt_token_ids)],
            hit_length=hit_length,
        )

class IndependentSearcher(Searcher):
    def __init__(self, 
            config: LocalStoreConfig,
            max_chunk_len: int = 256
        ):
        super().__init__(config, max_chunk_len)

    def search(
        self,
        prompt_token_ids: List[int],
        ind_ptr: List[int],
    ) -> SearchResult:
        assert len(prompt_token_ids) == ind_ptr[-1], f"Length mismatch between prompt_token_ids {len(prompt_token_ids)} and ind_ptr {ind_ptr[-1]}"
        sub_chunk_ptr, group_ptr = self._split_chunk_ptr(ind_ptr)
        cache_ids, chunk_lengths = [], []
        group_idx = 0
        parent = self.system_prefix
        for i in range(len(sub_chunk_ptr) - 1):
            while sub_chunk_ptr[i] >= group_ptr[group_idx + 1]:
                group_idx += 1
                parent = self.system_prefix

            token_split = prompt_token_ids[sub_chunk_ptr[i] : sub_chunk_ptr[i + 1]]
            cache_id = get_hash_id(token_ids_to_bytes(token_split), prefix=parent.encode())
            cache_ids.append(cache_id)
            chunk_lengths.append(len(token_split))
            parent = cache_id

        lookup_results = self.kvcache_index_client.lookup(cache_ids)
        kvcache_metas, hit_status = [], []
        hit_length = 0
        prefix_hit = True
        length_cum = 0
        for i in range(len(cache_ids)):
            if lookup_results[i]:
                if prefix_hit:  # Only count hits until the first miss
                    hit_length += chunk_lengths[i]
                hit_status.append(True)
                kvcache_metas.append(
                    KVCache(
                        cache_id=cache_ids[i],
                        ptr=[length_cum, length_cum + chunk_lengths[i]],
                    )
                )
            else:
                prefix_hit = False
                hit_status.append(False)
                kvcache_metas.append(
                    KVCache(
                        cache_id=cache_ids[i],
                        ptr=[length_cum, length_cum + chunk_lengths[i]],
                    )
                )
            length_cum += chunk_lengths[i]
        return SearchResult(
            kvcache_metas=kvcache_metas,
            hit_status=hit_status,
            group_ptr=group_ptr,
            hit_length=hit_length,
        )