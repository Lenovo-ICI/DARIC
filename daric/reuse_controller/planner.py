import os
from enum import IntEnum
from dataclasses import dataclass
from typing import List
from .searcher import SearchResult
from .kvcache import KVCache


class ChunkStatus(IntEnum):
    INTERNAL_HIT = 0
    READ = 1
    WRITE = 2
    NONE = 3


@dataclass
class LookUpResult:
    is_groupped: bool
    group_ptr: List[int]
    adjusted_group_ptr: List[int]
    hit_length: int

    chunk_chain: List[KVCache] = None
    chunk_status: List[ChunkStatus] = None
    inconsistent_s_ptrs: List[int] = None

    def for_worker(self) -> "LookUpResult":
        chunks = []
        inconsistent_s_ptrs = []
        for meta, status in zip(self.chunk_chain, self.chunk_status):
            if status == ChunkStatus.READ:
                if meta.ptr[0] >= self.hit_length:
                    inconsistent_s_ptrs.append(meta.ptr[0])
                if meta.require_io and meta.block_ids is not None:
                    chunks.append((meta, status))
            elif (
                status == ChunkStatus.WRITE
                and meta.require_io
                and meta.block_ids is not None
            ):
                chunks.append((meta, status))
        return LookUpResult(
            is_groupped=self.is_groupped,
            group_ptr=self.group_ptr,
            adjusted_group_ptr=self.adjusted_group_ptr,
            hit_length=self.hit_length,
            chunk_chain=[meta for meta, _ in chunks],
            chunk_status=[status for _, status in chunks],
            inconsistent_s_ptrs=inconsistent_s_ptrs,
        )


class Planner:
    def __init__(self):
        self.max_group = int(os.environ.get("DARIC_MAX_GROUP", "1"))

    def plan(self,
            internal_hit_length: int,
            external_hit_length: int,
            search_result: SearchResult
        ) -> LookUpResult: ...
    
    
class HybridPlanner(Planner):
    def __init__(self, strict_pic: bool = False):
        super().__init__()
        self.strict_pic = strict_pic
    
    def plan(self,
            internal_hit_length: int,
            external_hit_length: int,
            search_result: SearchResult
    ) -> LookUpResult:
        all_ptrs = [meta.ptr for meta in search_result.kvcache_metas]
        metas = search_result.kvcache_metas
        group_ptr = search_result.group_ptr
        hit_status = search_result.hit_status
        chunk_status = [ChunkStatus.NONE] * len(metas)

        for i, ptr in enumerate(all_ptrs):
            if ptr[1] <= internal_hit_length:
                chunk_status[i] = ChunkStatus.INTERNAL_HIT

        hit_length = max(internal_hit_length, external_hit_length)
        read_info = []
        inconsistent_hit_info = []

        if internal_hit_length >= external_hit_length:
            if len(group_ptr) == 2:
                adjusted_group_ptr = group_ptr
            else:
                adjusted_group_ptr = [0]
                for i in range(len(group_ptr) - 1):
                    start, end = group_ptr[i], group_ptr[i+1]
                    if end >= internal_hit_length:
                        adjusted_group_ptr.append(end)
        else:
            if len(group_ptr) == 2:
                adjusted_group_ptr = group_ptr
                for i, (ptr, meta, hit) in enumerate(zip(all_ptrs, metas, hit_status)):
                    if internal_hit_length < ptr[1] <= external_hit_length:
                        read_info.append(meta)
                        chunk_status[i] = ChunkStatus.READ
            else:
                adjusted_group_ptr = [0]
                new_start_idx = 0
                for i in range(len(group_ptr) - 1):
                    start, end = group_ptr[i], group_ptr[i+1]
                    if internal_hit_length < end:
                        if internal_hit_length >= start:
                            new_start_idx = start
                        if start > 0:
                            adjusted_group_ptr.append(start)
                adjusted_group_ptr.append(group_ptr[-1])

                for i, (ptr, meta, hit) in enumerate(zip(all_ptrs, metas, hit_status)):
                    if new_start_idx < ptr[1] <= external_hit_length:
                        read_info.append(meta)
                        chunk_status[i] = ChunkStatus.READ

        for i, hit in enumerate(hit_status):
            if not hit:
                chunk_status[i] = ChunkStatus.WRITE

        all_hit_info = read_info
        group_to_items = {i: [] for i in range(len(group_ptr) - 1)}
        for i, (meta, ptr, hit) in enumerate(zip(metas, all_ptrs, hit_status)):
            for j in range(len(group_ptr) - 1):
                if group_ptr[j] <= ptr[0] and ptr[1] <= group_ptr[j+1]:
                    group_to_items[j].append((i, meta, hit))
                    break

        for j in range(len(group_ptr) - 1):
            g_start = group_ptr[j]
            items = group_to_items[j]
            if g_start < hit_length:
                continue
            if all(hit and meta not in all_hit_info for _, meta, hit in items):
                for idx, meta, _ in items:
                    inconsistent_hit_info.append(meta)
                    chunk_status[idx] = ChunkStatus.READ

        if len(adjusted_group_ptr) - 1 > self.max_group:
            raise ValueError(
                f"Independent chunk number {len(adjusted_group_ptr) - 1} "
                f"exceeds max group {self.max_group}, "
                f"please adjust DARIC_MAX_GROUP environment variable."
            )

        if not self.strict_pic:
            all_hit_ptr = [meta.ptr[0] for meta in read_info + inconsistent_hit_info]
            adjusted_group_ptr_ = [0]
            for ptr in adjusted_group_ptr[1:-1]:
                if ptr in all_hit_ptr:
                    adjusted_group_ptr_.append(ptr)
            adjusted_group_ptr_.append(adjusted_group_ptr[-1])
            adjusted_group_ptr = adjusted_group_ptr_

        return LookUpResult(
            is_groupped=False if len(adjusted_group_ptr) <= 2 else True,
            group_ptr=group_ptr,
            adjusted_group_ptr=adjusted_group_ptr,
            hit_length=hit_length,
            chunk_chain=list(metas),
            chunk_status=chunk_status,
        )