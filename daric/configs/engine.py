import yaml
from dataclasses import dataclass

from ..utils import check_key_valid
from .backend import LocalStoreConfig, get_backend_config
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class LayerCacheTopo:
    group_idx: int
    logical_block_size: int
    block_byte_size: int
    page_first: bool
    is_mamba: bool = False


@dataclass
class CacheTopo:
    layers: dict[str, LayerCacheTopo]
    groups: list[list[str]]

    def get_all_layer_names(self) -> list[str]:
        layer_names = list(self.layers.keys())
        return sorted(layer_names)
    
    def get_layer_idx(self, layer_name: str) -> int:
        layer_names = self.get_all_layer_names()
        return layer_names.index(layer_name)
    
    def get_group_idx(self, layer_name: str) -> int:
        for g_idx, group in enumerate(self.groups):
            if layer_name in group:
                return g_idx
        raise ValueError(f"Layer {layer_name} not found in any group.")
    
    def get_all_group_block_size(self):
        group_block_sizes = []
        for group in self.groups:
            block_sizes = [self.layers[layer_name].logical_block_size for layer_name in group]
            assert all(size == block_sizes[0] for size in block_sizes), f"All layers in a group must have the same logical_block_size. Group: {group}, Sizes: {block_sizes}"
            group_block_sizes.append(block_sizes[0])
        return group_block_sizes

    def get_mamba_group_idxs(self) -> list[int]:
        return [
            group_idx
            for group_idx, group in enumerate(self.groups)
            if group and self.layers[group[0]].is_mamba
        ]

    def get_attention_group_idxs(self) -> list[int]:
        mamba_group_idxs = set(self.get_mamba_group_idxs())
        return [
            group_idx
            for group_idx in range(len(self.groups))
            if group_idx not in mamba_group_idxs
        ]

    def get_cache_partitions(self) -> list[tuple[str, list[int]]]:
        partitions = []
        attention_group_idxs = self.get_attention_group_idxs()
        mamba_group_idxs = self.get_mamba_group_idxs()
        if attention_group_idxs:
            partitions.append(("attention", attention_group_idxs))
        if mamba_group_idxs:
            partitions.append(("mamba", mamba_group_idxs))
        return partitions


class ProcessLocalConfig:
    # Model parameters
    engine_id: str
    model_key: str
    is_deepseek_mla: bool

    # Parallel states
    tp_size: int
    pp_size: int
    world_size: int
    tp_rank: int
    pp_rank: int
    rank: int

    # Node topology
    local_rank: int
    local_world_size: int
    node_rank: int
    nnodes: int

    # Engine info
    kv_role: str
    block_size: int
    cache_topo: CacheTopo

    @classmethod
    def get_node_local_tp_ranks(cls) -> list[int]:
        tp_group_start = cls.rank - cls.tp_rank
        return sorted(
            t for t in range(cls.tp_size)
            if (tp_group_start + t) // cls.local_world_size == cls.node_rank
        )

    @classmethod
    def mla_write_keep_indices(cls, n: int):
        if not cls.is_deepseek_mla:
            return range(n)
        node_tp_ranks = cls.get_node_local_tp_ranks()
        idx = node_tp_ranks.index(cls.tp_rank)
        num_on_node = len(node_tp_ranks)
        return [i for i in range(n) if i % num_on_node == idx]

    @classmethod
    def get_mla_saver_tp_rank(cls) -> int:
        return cls.get_node_local_tp_ranks()[0]

    @classmethod
    def get_local_mla_saver_rank(cls) -> int:
        tp_group_start = cls.rank - cls.tp_rank
        return tp_group_start + cls.get_mla_saver_tp_rank()

    @classmethod
    def get_mla_saver_ranks(cls) -> list[int]:
        savers: set[int] = set()
        for pp in range(cls.pp_size):
            tp_start = pp * cls.tp_size
            for node in range(cls.nnodes):
                ranks_on_node = [
                    t for t in range(cls.tp_size)
                    if (tp_start + t) // cls.local_world_size == node
                ]
                if ranks_on_node:
                    savers.add(tp_start + min(ranks_on_node))
        return sorted(savers)

    @classmethod
    def check_registered(cls):
        required_attrs = [
            "engine_id",
            "model_key",
            "is_deepseek_mla",
            "tp_size",
            "pp_size",
            "world_size",
            "tp_rank",
            "pp_rank",
            "rank",
            "local_rank",
            "local_world_size",
            "node_rank",
            "nnodes",
            "kv_role",
            "block_size",
            "cache_topo",
        ]
        for attr in required_attrs:
            if not hasattr(cls, attr):
                return False
        return True


@dataclass
class AdaptionConfig:
    rerotate: bool
    recompute: bool
    recompute_policy: None | str
    recompute_avg_num: int
    function: str = "linear"
    r: float = 0.1


"""Main config"""
@dataclass
class CacheEngineConfig:
    max_chunk_len: int
    reuse_mode: str
    strict_pic: bool
    decode_cache_enable: bool

    adaption_config: AdaptionConfig
    backend_config: LocalStoreConfig
    
    @classmethod
    def from_extra_config(cls, extra_config: dict, kv_role: str, pic_available: bool, block_size: int):
        # Set reuse config
        max_chunk_size = extra_config.get("max_chunk_len", 256)
        if block_size > max_chunk_size:
            logger.info(f"Block size {block_size} is larger than max_chunk_len {max_chunk_size}, set DARIC max_chunk_len to {block_size}.")
            max_chunk_size = block_size
        if max_chunk_size % block_size != 0:
            raise ValueError(f"max_chunk_len {max_chunk_size} should be a multiple of block_size {block_size}.")
        
        # Set backend config
        check_key_valid(extra_config, ["backend"])
        backend_info = extra_config.get("backend", None)
        backend_config = get_backend_config(backend_info)

        if pic_available:
            adaption_info = extra_config.get("adaption", None)
            if not adaption_info:
                adaption_config = AdaptionConfig(
                    rerotate=True,
                    recompute=True,
                    recompute_policy="daric",
                    recompute_avg_num=16
                )
            else:
                adaption_config = AdaptionConfig(
                    rerotate=adaption_info.get("rerotate", True),
                    recompute=adaption_info.get("recompute", True),
                    recompute_policy=adaption_info.get("recompute_policy", "daric"),
                    recompute_avg_num=adaption_info.get("recompute_avg_num", 16),
                    r=adaption_info.get("r", 0.1),
                    function=adaption_info.get("function", "linear")
                )
        else:
            logger.info(f"Non-prefix reuse is not available for current setting, fallback to prefix reuse.")
            adaption_config = AdaptionConfig(
                rerotate=False,
                recompute=False,
                recompute_policy=None,
                recompute_avg_num=0
            )
        reuse_mode = "independent" if pic_available else "prefix"

        return CacheEngineConfig(
            max_chunk_len=max_chunk_size,
            decode_cache_enable=extra_config.get("decode_cache_enable", False),
            reuse_mode=reuse_mode,
            strict_pic=extra_config.get("strict_pic", False),
            backend_config=backend_config,
            adaption_config=adaption_config,
        )


def get_engine_config(config_file: str, kv_role: str, pic_available: bool, block_size: int) -> CacheEngineConfig:
    with open(config_file, "r", encoding="utf-8") as f:
        extra_config = yaml.safe_load(f)
    engine_config = CacheEngineConfig.from_extra_config(extra_config, kv_role, pic_available, block_size)
    return engine_config