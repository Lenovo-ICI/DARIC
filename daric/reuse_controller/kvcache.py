from dataclasses import dataclass


def get_partition_cache_id(cache_id: str | int, partition: str) -> str:
    return f"{cache_id}:{partition}"

@dataclass
class KVCache:
    cache_id: str | int
    ptr: list[int, int]
    saved_pos: int = 0
    new_pos: None | int = None
    require_io: bool = True
    block_ids: None | tuple[list[int], ...] = None
    io_group_idxs: None | tuple[int, ...] = None
