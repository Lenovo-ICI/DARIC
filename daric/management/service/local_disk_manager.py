import os
import shutil
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Union

from .index import KVCacheIndex
from .rwlock import _RWLock
from ...logging import get_logger

logger = get_logger(__name__)


class DiskStorageManager:
    def __init__(self, root_dir, max_size, before_delete_entries: Optional[Callable] = None, protected_window: float = 0.0, index=None):
        self.root_dir = os.path.abspath(root_dir)
        if not os.path.exists(self.root_dir): os.makedirs(self.root_dir)
        self.max_size = max_size
        self.index = index if index is not None else KVCacheIndex(max_size=max_size, protected_window=protected_window)
        self.before_delete_entries = before_delete_entries
        self.pending = {}
        self._in_flight_keys: set = set()
        self.rwlock = _RWLock()
        self._monitor_cond_lock = threading.RLock()
        self.monitor_cond = threading.Condition(self._monitor_cond_lock)
        self._ops_lock = threading.Lock()
        self._ops_cond = threading.Condition(self._ops_lock)
        self._active_ops = 0
        self.pending_deletes = []
        self.closed = False
        self.monitor_interval_s = 1.0
        self.eviction_low_watermark_ratio = 0.3
        self.executor = ThreadPoolExecutor()
        self.shards = 256
        for i in range(self.shards):
            shard_dir = os.path.join(self.root_dir, f"{i:02x}")
            if not os.path.exists(shard_dir): os.makedirs(shard_dir)
        self.monitor_thread = threading.Thread(target=self._monitor_loop, name="disk-storage-monitor", daemon=True)
        self.monitor_thread.start()

    def _begin_op(self):
        with self._ops_cond:
            if self.closed: raise RuntimeError(f"Disk storage '{self.root_dir}' is closed")
            self._active_ops += 1

    def _end_op(self):
        with self._ops_cond:
            self._active_ops -= 1
            if self._active_ops == 0: self._ops_cond.notify_all()

    def _notify_monitor(self):
        with self.monitor_cond: self.monitor_cond.notify_all()

    def get_root_dir(self): return self.root_dir
    def get_storage_space(self): return self.max_size

    def _get_used_space_locked(self):
        index_used = self.index.get_used_space()
        pending_keys = list(self.pending.keys())
        pending_exists = self.index.contains(pending_keys) if pending_keys else []
        pending_extra = sum(size for exists, (_, size) in zip(pending_exists, self.pending.values()) if not exists)
        return index_used + pending_extra

    def get_used_space(self):
        with self.rwlock.read_lock(): return self._get_used_space_locked()

    def _get_free_space_locked(self): return self.max_size - self._get_used_space_locked()

    def get_free_space(self):
        with self.rwlock.read_lock(): return self._get_free_space_locked()

    def _get_shard_path(self, key):
        try: shard_id = int(key) % self.shards
        except ValueError: shard_id = hash(key) % self.shards
        return os.path.join(self.root_dir, f"{shard_id:02x}", str(key))

    def _delete_index_entries_locked(self, keys: List[Union[str, int]]):
        if not keys: return []
        metas = self.index.get(keys); self.index.delete(keys)
        return [(meta.offset, meta.size) for meta in metas if meta is not None and meta.offset is not None]

    def _drain_pending_deletes_locked(self):
        entries = self.pending_deletes; self.pending_deletes = []; return entries

    def _delete_file_entry(self, entry):
        file_path, size = entry
        if file_path is None: return 0, 0
        try: os.remove(file_path)
        except FileNotFoundError: pass
        except OSError as exc: logger.error(f"Error deleting file {file_path}: {exc}"); return 0, 0
        return 1, size

    def _delete_files(self, entries):
        if not entries: 
            return 0, 0
        futures = list(self.executor.map(self._delete_file_entry, entries))
        deleted_files, deleted_bytes = 0, 0
        for files, bytes_ in futures:
            deleted_files += files; deleted_bytes += bytes_
        logger.info(f"Deleted {deleted_files} files, freed {deleted_bytes / (1024**3):.2f} GB, used space: {self.get_used_space() / (1024**3):.2f} GB")
        return deleted_files, deleted_bytes

    def _monitor_loop(self):
        while True:
            if self.closed: return
            # Removed proactive eviction to prevent data inconsistency
            with self.rwlock.write_lock(): pending = self._drain_pending_deletes_locked()
            if pending: self._delete_files(pending)
            with self.monitor_cond:
                if self.closed: return
                self.monitor_cond.wait(timeout=self.monitor_interval_s)

    def evict_to_free(self, size: int) -> bool:
        self._begin_op()
        try:
            for _ in range(100):
                files_to_delete = []
                with self.rwlock.write_lock():
                    if self._get_free_space_locked() >= size: return True
                    free_space = self._get_free_space_locked()
                    target_free_space = max(size, int(self.max_size * self.eviction_low_watermark_ratio))
                    bytes_to_free = target_free_space - free_space
                    # Ask the index for a right-sized candidate list (bounded by
                    # bytes_to_free) instead of all candidates, so we don't scan
                    # and filter the entire index on every eviction round.
                    raw = self.index.get_eviction_candidates(bytes_to_free=bytes_to_free) or []
                    candidates = [k for k in raw if k not in self._in_flight_keys]
                    if not candidates: return False
                    if self.before_delete_entries is not None:
                        try:
                            maybe_keys = self.before_delete_entries(list(candidates))
                            if maybe_keys is not None: candidates = maybe_keys
                        except Exception: return False
                    # Re-verify ts/validity and delete atomically inside the
                    # index lock (see KVCacheIndex.evict_cold): a lookup/touch
                    # landing between the candidate scan above and the delete
                    # would otherwise lose the key despite the
                    # protected_window.  Delete all selected candidates in one
                    # batch (no per-key bytes cap) to avoid shrinking each
                    # round to just a couple of files (the "Deleted 2 files"
                    # thrash).
                    evicted_metas, _ = self.index.evict_cold(candidates, bytes_to_free=None)
                    if not evicted_metas: return False
                    files_to_delete = [(meta.offset, meta.size) for meta in evicted_metas]
                if files_to_delete: self._delete_files(files_to_delete)
                self._notify_monitor()
            return False
        finally:
            self._end_op()

    def allocate(self, keys: List[Union[str, int]], sizes: List[int]):
        self._begin_op()
        try:
            paths, add_sizes = [], []
            with self.rwlock.write_lock():
                free_space = self._get_free_space_locked()
                existing = self.index.get(keys)
                pending_keys, pending_sizes, pending_paths = [], [], []
                for i, (key, size) in enumerate(zip(keys, sizes)):
                    if existing[i] is not None:
                        paths.append(None); add_sizes.append(None); continue
                    if size > free_space:
                        paths.append(None); add_sizes.append(None); continue
                    path = self._get_shard_path(key)
                    self.pending[key] = (path, size)
                    self._in_flight_keys.add(key)
                    free_space -= size
                    paths.append(path); add_sizes.append(size)
                    pending_keys.append(key); pending_sizes.append(size); pending_paths.append(path)
                if pending_keys:
                    self.index.submit_insert(pending_keys, pending_sizes, pending_paths)
            self._notify_monitor()
            return paths, add_sizes
        finally: self._end_op()

    def finish_put(self, keys: List[Union[str, int]]):
        self._begin_op()
        try:
            with self.rwlock.write_lock():
                self.index.finish_insert(keys)
                for key in keys:
                    self.pending.pop(key, None)
                    self._in_flight_keys.discard(key)
            self._notify_monitor()
        finally: self._end_op()

    def lookup(self, keys: List[Union[str, int]]):
        self._begin_op()
        try:
            with self.rwlock.read_lock(): return self.index.lookup(keys)
        finally: self._end_op()

    def get_meta(self, keys: List[Union[str, int]], *, touch: bool = False, require_valid: bool = True):
        self._begin_op()
        try:
            lock_ctx = self.rwlock.write_lock() if touch else self.rwlock.read_lock()
            with lock_ctx:
                metas = self.index.touch(keys) if touch else self.index.get(keys)
                results = []
                for key, meta in zip(keys, metas):
                    if meta is None or (require_valid and not meta.valid):
                        raise KeyError(f"Key '{key}' not found in storage.")
                    results.append((meta.offset, meta.size))
                return results
        finally: self._end_op()

    def delete(self, keys: List[Union[str, int]]):
        self._begin_op()
        try:
            keys_to_delete = keys
            if self.before_delete_entries is not None:
                try:
                    maybe_keys = self.before_delete_entries(list(keys))
                    if maybe_keys is not None: keys_to_delete = maybe_keys
                except Exception: pass
            if not keys_to_delete: return
            files_to_delete = []
            with self.rwlock.write_lock():
                for key in keys_to_delete:
                    self._in_flight_keys.discard(key)
                    entry = self.pending.pop(key, None)
                    if entry is not None: files_to_delete.append(entry)
                files_to_delete = files_to_delete + self._delete_index_entries_locked(keys_to_delete)
            if files_to_delete: self._delete_files(files_to_delete)
            self._notify_monitor()
        finally: self._end_op()

    def clean(self):
        self._begin_op()
        try:
            with self.rwlock.write_lock():
                index_entries = [(meta.offset, meta.size) for meta in self.index.get_all_meta() if meta.offset is not None]
                all_entries = index_entries + list(self.pending.values())
                self.index.clean(); self.pending.clear()
                self.pending_deletes.clear(); self._in_flight_keys.clear()
            if all_entries:
                self._delete_files(all_entries)
            self._notify_monitor()
        finally: self._end_op()

    def clean_local(self):
        """Remove local disk files and bookkeeping without clearing the index."""
        self._begin_op()
        try:
            with self.rwlock.write_lock():
                self.pending.clear()
                self.pending_deletes.clear()
                self._in_flight_keys.clear()
            # Wipe files under each shard directory but keep the directory structure.
            for i in range(self.shards):
                shard_dir = os.path.join(self.root_dir, f"{i:02x}")
                if not os.path.isdir(shard_dir):
                    continue
                for name in os.listdir(shard_dir):
                    path = os.path.join(shard_dir, name)
                    try:
                        os.remove(path)
                    except OSError as exc:
                        logger.error(f"clean_local: error removing {path}: {exc}")
            self._notify_monitor()
        finally: self._end_op()

    def close(self):
        with self._ops_cond: self.closed = True
        self._notify_monitor()
        with self._ops_cond:
            while self._active_ops > 0: self._ops_cond.wait()
        if hasattr(self, "monitor_thread") and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=max(1.0, self.monitor_interval_s * 2))
        self.executor.shutdown(wait=True)
        try:
            if os.path.exists(self.root_dir): shutil.rmtree(self.root_dir)
        except Exception as e: logger.error(f"Error cleaning up disk storage: {e}")