import bisect
import _posixshmem
import threading
from typing import Callable, List, Optional, Union
from multiprocessing import shared_memory, resource_tracker

from .index import KVCacheIndex
from .rwlock import _RWLock
from ...logging import get_logger

logger = get_logger(__name__)


class PinMemStorageManager:
    def __init__(self, shm_name: str, shm_size: int, eviction_protector: Optional[Callable[[str], bool]] = None, protected_window: float = 0.0, read_lease_ttl: float = 2.0, index=None):
        self.shm_name = shm_name
        self.shm_size = shm_size
        self.index = index if index is not None else KVCacheIndex(max_size=shm_size, protected_window=protected_window)
        self.eviction_protector = eviction_protector
        self.read_lease_ttl = read_lease_ttl
        self.active_read_leases: dict = {}
        self._read_lease_lock = threading.Lock()
        self.current_offset = 0
        self.free_blocks: List[tuple] = []
        self.free_blocks_by_size: List[tuple] = []
        self.rwlock = _RWLock()
        self._monitor_cond_lock = threading.RLock()
        self.monitor_cond = threading.Condition(self._monitor_cond_lock)
        self._in_flight_keys: set = set()
        self._ops_lock = threading.Lock()
        self._ops_cond = threading.Condition(self._ops_lock)
        self._active_ops = 0
        self.pending_reclaims: list = []
        self.closed = False
        self.monitor_interval_s = 1.0

        try:
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=self.shm_size)
        except FileExistsError:
            try:
                existing_shm = shared_memory.SharedMemory(name=self.shm_name)
                existing_shm.close(); existing_shm.unlink()
            except FileNotFoundError: pass
            self.shm = shared_memory.SharedMemory(name=self.shm_name, create=True, size=self.shm_size)

        try: resource_tracker.unregister(self.shm._name, "shared_memory")
        except Exception: pass

        self.monitor_thread = threading.Thread(target=self._monitor_loop, name=f"pinmem-monitor-{self.shm_name}", daemon=True)
        self.monitor_thread.start()

    def _begin_op(self):
        with self._ops_cond:
            if self.closed: 
                raise RuntimeError(f"Storage '{self.shm_name}' is closed")
            self._active_ops += 1

    def _end_op(self):
        with self._ops_cond:
            self._active_ops -= 1
            if self._active_ops == 0: 
                self._ops_cond.notify_all()

    def get_storage_space(self): 
        return self.shm_size

    def get_used_space(self):
        with self.rwlock.read_lock():
            used = self.current_offset
            used -= sum(size for _, size in self.free_blocks)
            return used

    def get_free_space(self):
        with self.rwlock.read_lock():
            free_blocks_size = sum(size for _, size in self.free_blocks)
            remaining_space = self.shm_size - self.current_offset
            return free_blocks_size + remaining_space

    def get_shm_details(self): 
        return self.shm_name, self.shm_size
    
    def _align(self, size: int) -> int: 
        return (size + 255) // 256 * 256

    def _allocate(self, size: int) -> int:
        size = self._align(size)
        idx = bisect.bisect_left(self.free_blocks_by_size, (size, 0))
        if idx < len(self.free_blocks_by_size):
            block_size, block_offset = self.free_blocks_by_size.pop(idx)
            offset_idx = bisect.bisect_left(self.free_blocks, (block_offset, 0))
            if offset_idx < len(self.free_blocks) and self.free_blocks[offset_idx] == (block_offset, block_size):
                self.free_blocks.pop(offset_idx)
            if block_size > size: 
                self._add_free_block(block_offset + size, block_size - size)
            return block_offset

        if self.current_offset % 256 != 0:
            self.current_offset = (self.current_offset + 255) // 256 * 256
        if self.current_offset + size > self.shm_size: 
            raise MemoryError("Not enough space in shared memory")
        offset = self.current_offset
        self.current_offset += size
        return offset

    def _free(self, offset: int, size: int): 
        self._add_free_block(offset, self._align(size))

    def _get_allocatable_prefix_length_locked(self, sizes: List[int]) -> int:
        available_blocks = [size for size, _ in self.free_blocks_by_size]
        tail_space = self.shm_size - self.current_offset
        allocatable_count = 0
        for size in sizes:
            aligned_size = self._align(size)
            idx = bisect.bisect_left(available_blocks, aligned_size)
            if idx < len(available_blocks):
                block_size = available_blocks.pop(idx)
                if block_size - aligned_size > 0:
                    bisect.insort(available_blocks, block_size - aligned_size)
                allocatable_count += 1
                continue
            if tail_space < aligned_size: 
                break
            tail_space -= aligned_size; allocatable_count += 1
        return allocatable_count

    def get_allocatable_prefix_length(self, sizes: List[int]) -> int:
        with self.rwlock.read_lock():
            return self._get_allocatable_prefix_length_locked(sizes)

    def _add_free_block(self, offset: int, size: int):
        idx = bisect.bisect_left(self.free_blocks, (offset, 0))
        self.free_blocks.insert(idx, (offset, size))
        if idx > 0:
            prev_offset, prev_size = self.free_blocks[idx - 1]
            if prev_offset + prev_size == offset:
                self._remove_from_by_size(prev_size, prev_offset)
                self.free_blocks[idx - 1] = (prev_offset, prev_size + size)
                self.free_blocks.pop(idx); offset = prev_offset; size = prev_size + size
                idx -= 1
        if idx + 1 < len(self.free_blocks):
            next_offset, next_size = self.free_blocks[idx + 1]
            if offset + size == next_offset:
                self._remove_from_by_size(next_size, next_offset)
                self.free_blocks.pop(idx + 1)
                self.free_blocks[idx] = (offset, size + next_size)
                size += next_size
        bisect.insort(self.free_blocks_by_size, (size, offset))

    def _remove_from_by_size(self, size: int, offset: int):
        idx = bisect.bisect_left(self.free_blocks_by_size, (size, 0))
        while idx < len(self.free_blocks_by_size):
            s, o = self.free_blocks_by_size[idx]
            if s != size: 
                break
            if o == offset: 
                self.free_blocks_by_size.pop(idx)
                return
            idx += 1

    def _delete_index_entries_locked(self, keys: List[Union[str, int]]):
        if not keys: 
            return []
        metas = self.index.get(keys)
        self.index.delete(keys)
        return [meta for meta in metas if meta is not None]

    def _reclaim_metas_locked(self, metas: list) -> int:
        reclaimed = 0
        for meta in metas:
            if meta is None or meta.offset is None: 
                continue
            self._free(meta.offset, meta.size)
            reclaimed += 1
        return reclaimed

    def _drain_pending_reclaims_locked(self):
        metas = self.pending_reclaims
        self.pending_reclaims = []
        return metas

    def _notify_monitor(self):
        with self.monitor_cond: 
            self.monitor_cond.notify_all()

    def _protect_read_keys(self, keys: List[Union[str, int]]) -> None:
        if self.read_lease_ttl <= 0 or not keys:
            return
        with self._read_lease_lock:
            for key in keys:
                key = str(key)
                self.active_read_leases[key] = self.active_read_leases.get(key, 0) + 1

    def _is_read_lease_active(self, key: str) -> bool:
        with self._read_lease_lock:
            return self.active_read_leases.get(str(key), 0) > 0

    def release_read(self, keys: List[Union[str, int]]) -> None:
        if not keys:
            return
        with self._read_lease_lock:
            for key in keys:
                key = str(key)
                count = self.active_read_leases.get(key, 0)
                if count <= 1:
                    self.active_read_leases.pop(key, None)
                else:
                    self.active_read_leases[key] = count - 1

    def evict_to_free(self, size: int) -> bool:
        return self.evict_to_fit([size])

    def evict_to_fit(self, sizes: List[int]) -> bool:
        sizes = [size for size in sizes if size is not None]
        if not sizes:
            return True
        self._begin_op()
        try:
            bytes_to_free = sum(sizes)
            for _ in range(100):
                with self.rwlock.write_lock():
                    if self._get_allocatable_prefix_length_locked(sizes) >= len(sizes):
                        return True

                    raw = self.index.get_eviction_candidates() or []
                    candidates = [
                        k for k in raw
                        if k not in self._in_flight_keys
                        and not self._is_read_lease_active(str(k))
                        and not (self.eviction_protector is not None and self.eviction_protector(str(k)))
                    ]
                    if not candidates:
                        return False

                    evicted_metas, _ = self.index.evict_cold(candidates, bytes_to_free)
                    if not evicted_metas:
                        return False
                    self._reclaim_metas_locked(evicted_metas)
                self._notify_monitor()
            return False
        finally:
            self._end_op()

    def _monitor_loop(self):
        while True:
            if self.closed: 
                return
            # Removed proactive eviction to prevent overwriting data being read by the client
            with self.rwlock.write_lock():
                pending = self._drain_pending_reclaims_locked()
                if pending: 
                    self._reclaim_metas_locked(pending)
            with self.monitor_cond:
                if self.closed: 
                    return
                self.monitor_cond.wait(timeout=self.monitor_interval_s)

    def lookup(self, keys: List[Union[str, int]]):
        self._begin_op()
        try:
            with self.rwlock.read_lock(): 
                return self.index.lookup(keys)
        finally: self._end_op()

    def allocate(self, keys: List[Union[str, int]], sizes: List[int]):
        self._begin_op()
        try:
            offsets, add_sizes = self._try_allocate(keys, sizes)
            failed_indices = [i for i, o in enumerate(offsets) if o is None]
            if failed_indices:
                failed_keys = [keys[i] for i in failed_indices]
                failed_sizes = [sizes[i] for i in failed_indices]
                existing_metas = self.index.get(failed_keys)
                genuine_fail = [
                    (failed_indices[i], failed_keys[i], failed_sizes[i])
                    for i, m in enumerate(existing_metas)
                    if m is None
                ]

                if genuine_fail:
                    g_keys = [k for _, k, _ in genuine_fail]
                    g_sizes = [s for _, _, s in genuine_fail]
                    if self.evict_to_fit(g_sizes):
                        retry_offsets, retry_sizes = self._try_allocate(g_keys, g_sizes)
                        for i, (orig_idx, _, _) in enumerate(genuine_fail):
                            offsets[orig_idx] = retry_offsets[i]
                            add_sizes[orig_idx] = retry_sizes[i]

            return offsets, add_sizes
        finally:
            self._end_op()


    def _try_allocate(self, keys: List[Union[str, int]], sizes: List[int]):
        offsets, add_sizes = [], []
        with self.rwlock.write_lock():
            existing = self.index.get(keys)
            pending_keys, pending_sizes, pending_offsets = [], [], []
            for i, (key, size) in enumerate(zip(keys, sizes)):
                if existing[i] is not None:
                    offsets.append(None)
                    add_sizes.append(None)
                    continue
                if self._get_allocatable_prefix_length_locked([size]) == 0:
                    offsets.append(None)
                    add_sizes.append(None)
                    continue
                try:
                    offset = self._allocate(size)
                except MemoryError:
                    offsets.append(None)
                    add_sizes.append(None)
                    continue
                pending_keys.append(key)
                pending_sizes.append(size)
                pending_offsets.append(offset)
                self._in_flight_keys.add(key)
                offsets.append(offset)
                add_sizes.append(size)
            if pending_keys:
                self.index.submit_insert(pending_keys, pending_sizes, pending_offsets)
        self._notify_monitor()
        return offsets, add_sizes

    def finish_put(self, keys: List[Union[str, int]]):
        self._begin_op()
        try:
            with self.rwlock.write_lock():
                self.index.finish_insert(keys)
                for key in keys: 
                    self._in_flight_keys.discard(key)
            self._notify_monitor()
        finally: self._end_op()

    def get_meta(self, keys: List[Union[str, int]], *, touch: bool = False, require_valid: bool = True, protect_read: bool = True):
        self._begin_op()
        protected_keys = []
        try:
            if protect_read:
                self._protect_read_keys(keys)
                protected_keys = list(keys)
            lock_ctx = self.rwlock.write_lock() if touch else self.rwlock.read_lock()
            with lock_ctx:
                metas = self.index.touch(keys) if touch else self.index.get(keys)
                results = []
                for key, meta in zip(keys, metas):
                    if meta is None or (require_valid and not meta.valid):
                        raise KeyError(f"Key '{key}' not found in storage.")
                    results.append((meta.offset, meta.size))
                return results
        except BaseException:
            if protected_keys:
                self.release_read(protected_keys)
            raise
        finally: self._end_op()

    def get_meta_prefix(self, keys: List[Union[str, int]], *, touch: bool = False, require_valid: bool = True):
        return self.get_meta(keys, touch=touch, require_valid=require_valid), keys

    def get_readable_prefix_length(self, keys: List[Union[str, int]], sizes: List[int]) -> int:
        self._begin_op()
        try:
            with self.rwlock.read_lock():
                metas = self.index.get(keys)
                readable_count = 0
                for meta in metas:
                    if meta is None or not meta.valid:
                        break
                    readable_count += 1
                return readable_count
        finally: self._end_op()

    def delete(self, keys: List[Union[str, int]]):
        self._begin_op()
        try:
            with self.rwlock.write_lock():
                for key in keys: 
                    self._in_flight_keys.discard(key)
                metas = self._delete_index_entries_locked(keys)
                self._reclaim_metas_locked(metas)
            self._notify_monitor()
        finally: self._end_op()

    def clean(self):
        self._begin_op()
        try:
            with self.rwlock.write_lock():
                self.index.clean()
                self.current_offset = 0
                self.free_blocks.clear()
                self.free_blocks_by_size.clear()
                self.pending_reclaims.clear()
                self._in_flight_keys.clear()
            self._notify_monitor()
        finally: self._end_op()

    def clean_local(self):
        """Reset local physical storage without clearing the index."""
        self._begin_op()
        try:
            with self.rwlock.write_lock():
                self.current_offset = 0
                self.free_blocks.clear()
                self.free_blocks_by_size.clear()
                self.pending_reclaims.clear()
                self._in_flight_keys.clear()
            self._notify_monitor()
        finally: self._end_op()

    def close(self):
        with self._ops_cond: 
            self.closed = True
        self._notify_monitor()
        with self._ops_cond:
            while self._active_ops > 0: 
                self._ops_cond.wait()
        if hasattr(self, "monitor_thread") and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=max(1.0, self.monitor_interval_s * 2))
        try:
            self.shm.close(); _posixshmem.shm_unlink(self.shm_name)
        except Exception as e: 
            logger.error(f"Error cleaning up shared memory: {e}")