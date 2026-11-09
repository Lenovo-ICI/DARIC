import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Optional, Tuple, Union

from .local_disk_manager import DiskStorageManager
from .local_pinmem_manager import PinMemStorageManager
from ...logging import get_logger

logger = get_logger(__name__)


class HybridStorageManager:
    def __init__(
        self,
        shm_name: str,
        shm_size: int,
        root_dir: str,
        disk_size: int,
        mem_index=None,
        disk_index=None,
    ):
        if disk_size < shm_size:
            raise ValueError("Hybrid storage requires disk_size to be at least shm_size")

        self.copy_executor = ThreadPoolExecutor(max_workers=64)
        self.pending_disk_writes: dict = {}
        self.protected_disk_write_keys: set = set()
        self._pending_lock = threading.Lock()

        # ---- read-lease bookkeeping (pure refcount, NO expiry) ----
        # active_read_leases: key -> refcount (>0 means leased, cannot evict)
        # read_lease_sizes:   key -> size in bytes (for byte-quota accounting)
        # read_lease_bytes:   sum of sizes of all keys whose refcount > 0
        self.active_read_leases: Dict[str, int] = {}
        self._read_lease_sizes: Dict[str, int] = {}
        self._read_lease_bytes: int = 0
        self._read_lease_lock = threading.Lock()
        self._read_lease_cond = threading.Condition(self._read_lease_lock)
        # Quota: outstanding leased bytes must not exceed this.  Guarantees
        # that at least (1 - ratio) of mem is evictable for promotes.
        self.read_lease_capacity_ratio = 0.5
        self._read_lease_capacity = int(shm_size * self.read_lease_capacity_ratio)

        # ---- promote safety nets ----
        self._promotion_lock = threading.Lock()
        # Hard cap on a single promote's retry loop.  0 disables (compat).
        self.promotion_retry_timeout = 120.0
        # Bound concurrent promotes so stuck promotes cannot drain all
        # ZMQ workers.
        self._promote_concurrency = 8
        self._promote_sem = threading.Semaphore(self._promote_concurrency)

        self.mem_storage = PinMemStorageManager(
            shm_name, shm_size, eviction_protector=self._should_protect_mem_key,
            protected_window=0.0, index=mem_index,
        )
        self.disk_storage = DiskStorageManager(
            root_dir, disk_size, before_delete_entries=self._filter_evictable_disk_keys,
            protected_window=0.0, index=disk_index,
        )

    # ------------------------------------------------------------------ #
    # read-lease: pure refcount + byte-quota admission control
    # ------------------------------------------------------------------ #
    def _should_protect_mem_key(self, key: str) -> bool:
        key = str(key)
        with self._pending_lock:
            if key in self.pending_disk_writes or key in self.protected_disk_write_keys:
                return True
        return self._is_read_lease_active(key)

    def _filter_evictable_disk_keys(self, keys: List[Union[str, int]]) -> List[Union[str, int]]:
        return [key for key in keys if not self._is_read_lease_active(str(key))]

    def _is_read_lease_active(self, key: str) -> bool:
        with self._read_lease_lock:
            return self.active_read_leases.get(str(key), 0) > 0

    def _acquire_lease(
        self, key: Union[str, int], size: int, timeout: Optional[float] = None
    ) -> bool:
        """Acquire (increment) the read lease for ``key``.

        * ``timeout == 0``  : non-blocking; return False immediately if the
          byte-quota would be exceeded by a brand-new lease.
        * ``timeout is None``: block forever until the lease is acquired.
        * ``timeout > 0``   : block up to ``timeout`` seconds.

        If ``key`` already has a lease (refcount > 0) the call always
        succeeds immediately without consuming extra quota — only the
        0→1 transition counts against the byte budget.
        """
        key = str(key)
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._read_lease_cond:
            while True:
                current = self.active_read_leases.get(key, 0)
                if current > 0:
                    # Already leased: bump refcount, no extra byte cost.
                    self.active_read_leases[key] = current + 1
                    return True
                # Brand-new lease: check byte quota.
                if self._read_lease_bytes + size <= self._read_lease_capacity:
                    self.active_read_leases[key] = 1
                    self._read_lease_sizes[key] = size
                    self._read_lease_bytes += size
                    return True
                # Quota full.
                if timeout == 0:
                    return False
                if deadline is not None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        return False
                    self._read_lease_cond.wait(timeout=remaining)
                else:
                    self._read_lease_cond.wait()

    def release_read(self, keys: List[Union[str, int]]) -> None:
        if not keys:
            return
        with self._read_lease_cond:
            notify = False
            for key in keys:
                key = str(key)
                current = self.active_read_leases.get(key, 0)
                if current <= 1:
                    self.active_read_leases.pop(key, None)
                    size = self._read_lease_sizes.pop(key, 0)
                    self._read_lease_bytes -= size
                    if self._read_lease_bytes < 0:
                        self._read_lease_bytes = 0
                    notify = True
                else:
                    self.active_read_leases[key] = current - 1
            if notify:
                self._read_lease_cond.notify_all()

    def _get_active_read_lease_count(self) -> int:
        with self._read_lease_lock:
            return len(self.active_read_leases)

    def _get_read_lease_bytes(self) -> int:
        with self._read_lease_lock:
            return self._read_lease_bytes

    # ------------------------------------------------------------------ #
    # pending disk writes
    # ------------------------------------------------------------------ #
    def _wait_pending_disk_writes(self, keys: Optional[List[Union[str, int]]] = None) -> None:
        if keys is None:
            with self._pending_lock:
                futures = list(self.pending_disk_writes.values())
        else:
            str_keys = [str(k) for k in keys]
            with self._pending_lock:
                futures = [self.pending_disk_writes[k] for k in str_keys if k in self.pending_disk_writes]
        for future in futures:
            try:
                future.result()
            except Exception:
                pass

    def _copy_mem_to_disk_one(self, disk_path: str, mem_offset: int, size: int) -> None:
        view = memoryview(self.mem_storage.shm.buf)[mem_offset: mem_offset + size]
        try:
            with open(disk_path, "wb", buffering=0) as handle:
                if handle.write(view) != size:
                    raise IOError("Failed to write all bytes to disk.")
        finally:
            view.release()

    def _copy_disk_to_mem_one(self, disk_path: str, mem_offset: int, size: int) -> None:
        view = memoryview(self.mem_storage.shm.buf)[mem_offset: mem_offset + size]
        try:
            with open(disk_path, "rb", buffering=0) as handle:
                if handle.readinto(view) != size:
                    raise IOError("Failed to read all bytes from disk.")
        finally:
            view.release()

    def _background_finish_disk_put(self, key: Union[str, int], disk_path: str, mem_offset: int, size: int) -> None:
        try:
            self._copy_mem_to_disk_one(disk_path, mem_offset, size)
            self.disk_storage.finish_put([key])
        except Exception:
            self.mem_storage.delete([key])
            self.disk_storage.delete([key])
            raise
        finally:
            with self._pending_lock:
                self.pending_disk_writes.pop(str(key), None)

    def _format_mem_pressure(self, request_size: int) -> str:
        return (
            f"request={request_size / (1024**3):.2f} GB, "
            f"free={self.mem_storage.get_free_space() / (1024**3):.2f} GB, "
            f"total={self.mem_storage.get_storage_space() / (1024**3):.2f} GB, "
            f"active_read_leases={self._get_active_read_lease_count()}, "
            f"leased_bytes={self._get_read_lease_bytes() / (1024**3):.2f}/"
            f"{self._read_lease_capacity / (1024**3):.2f} GB"
        )

    # ------------------------------------------------------------------ #
    # promote: bounded concurrency + hard timeout
    # ------------------------------------------------------------------ #
    def _promote_from_disk(self, key: Union[str, int]) -> Tuple[int, int]:
        self._wait_pending_disk_writes([key])
        try:
            disk_path, disk_size = self.disk_storage.get_meta([key], require_valid=True)[0]
        except KeyError:
            raise KeyError(f"Key '{key}' not found in disk storage.")
        if disk_size > self.mem_storage.get_storage_space():
            raise MemoryError(
                f"promote key={key} size={disk_size} exceeds mem total "
                f"{self.mem_storage.get_storage_space()}"
            )

        retry_interval = 0.05
        max_retry_interval = 0.2
        attempt = 0
        start_time = time.monotonic()
        hard_deadline = (
            start_time + self.promotion_retry_timeout
            if self.promotion_retry_timeout > 0 else None
        )

        self._promote_sem.acquire()
        try:
            while True:
                mem_offsets, _ = self.mem_storage.allocate([key], [disk_size])
                if mem_offsets[0] is not None:
                    mem_offset = mem_offsets[0]
                    try:
                        self._copy_disk_to_mem_one(disk_path, mem_offset, disk_size)
                        self.mem_storage.finish_put([key])
                    except Exception:
                        self.mem_storage.delete([key])
                        raise
                    return mem_offset, disk_size

                # Another thread may be writing / may have finished.
                existing_meta = self.mem_storage.index.get([key])[0]
                if existing_meta is not None:
                    if existing_meta.valid:
                        return existing_meta.offset, existing_meta.size
                    time.sleep(retry_interval)
                    continue

                # Try evicting old data.  Because of the byte-quota this is
                # guaranteed to make progress eventually (at least
                # (1-ratio) of mem is evictable).
                if self.mem_storage.evict_to_free(disk_size):
                    continue

                attempt += 1
                elapsed = time.monotonic() - start_time
                if attempt % 10 == 1:
                    logger.debug(
                        "promote key=%s evict failed, waiting for pending writes; "
                        "%s, elapsed=%.1fs",
                        key, self._format_mem_pressure(disk_size), elapsed,
                    )

                if hard_deadline is not None and time.monotonic() > hard_deadline:
                    logger.warning(
                        "promote key=%s timed out after %.1fs; %s; raising MemoryError",
                        key, elapsed, self._format_mem_pressure(disk_size),
                    )
                    raise MemoryError(
                        f"promote key={key} timed out after {elapsed:.1f}s: "
                        f"cannot free {disk_size} bytes from mem "
                        f"({self._format_mem_pressure(disk_size)})"
                    )

                self._wait_pending_disk_writes()
                time.sleep(retry_interval)
                retry_interval = min(retry_interval * 1.5, max_retry_interval)
        finally:
            self._promote_sem.release()

    def _promote_from_disk_batch(
        self,
        keys: List[Union[str, int]],
        disk_sizes: List[int],
    ) -> Dict[Union[str, int], Tuple[int, int]]:
        if not keys:
            return {}
        if len(keys) == 1:
            # No parallelism to exploit, and the single-key path carries the
            # full eviction-retry + hard timeout — skip the executor.
            return {keys[0]: self._promote_from_disk(keys[0])}

        # File paths (sizes already in disk_sizes; lookup already pinned).
        disk_metas = self.disk_storage.get_meta(keys, require_valid=False)

        try:
            mem_offsets, _ = self.mem_storage.allocate(keys, disk_sizes)
        except Exception:
            mem_offsets = [None] * len(keys)

        results: Dict[Union[str, int], Tuple[int, int]] = {}
        fallback: List[Union[str, int]] = []

        # Parallel-copy the keys that got a mem slot.
        to_copy = [
            (keys[i], disk_metas[i][0], disk_sizes[i], mem_offsets[i])
            for i in range(len(keys))
            if mem_offsets[i] is not None
        ]
        if to_copy:
            futures = {}
            for key, path, size, offset in to_copy:
                fut = self.copy_executor.submit(self._copy_disk_to_mem_one, path, offset, size)
                futures[fut] = (key, offset, size)
            for fut in as_completed(futures):
                key, offset, size = futures[fut]
                try:
                    fut.result()
                    results[key] = (offset, size)
                except Exception:
                    # Free the slot we allocated; single-key retry below.
                    self.mem_storage.delete([key])
                    fallback.append(key)
            if results:
                self.mem_storage.finish_put(list(results.keys()))

        # Keys allocate() returned None for: either already in mem (another
        # thread promoted them) or genuinely unallocatable.  A read-only mem
        # index get tells them apart; the unallocatable ones fall back.
        for i, key in enumerate(keys):
            if mem_offsets[i] is not None:
                continue  # handled above (ok, or copy-failed -> fallback)
            existing = self.mem_storage.index.get([key])[0]
            if existing is not None and existing.valid:
                results[key] = (existing.offset, existing.size)
            else:
                fallback.append(key)

        # Hard cases: single-key promote with full retry/timeout.  This is
        # also where a too-big key raises MemoryError and a vanished key
        # raises KeyError.
        for key in fallback:
            results[key] = self._promote_from_disk(key)

        return results

    # ------------------------------------------------------------------ #
    # space / info helpers
    # ------------------------------------------------------------------ #
    def get_storage_space(self):
        return self.mem_storage.get_storage_space()

    def get_used_space(self):
        return self.mem_storage.get_used_space()

    def get_free_space(self):
        return self.mem_storage.get_free_space()

    def get_shm_details(self):
        return self.mem_storage.get_shm_details()

    def lookup(self, keys: List[Union[str, int]]):
        self._wait_pending_disk_writes(keys)
        mem_results = self.mem_storage.lookup(keys)
        results, disk_keys = [], []
        for i, key in enumerate(keys):
            if mem_results[i]:
                results.append(True)
            else:
                disk_keys.append(key)
                results.append(None)
        if disk_keys:
            disk_results = self.disk_storage.lookup(disk_keys)
            disk_idx = 0
            for i in range(len(keys)):
                if results[i] is None:
                    results[i] = disk_results[disk_idx]
                    disk_idx += 1
        return results

    # ------------------------------------------------------------------ #
    # allocate / finish_put (write path — unchanged logic)
    # ------------------------------------------------------------------ #
    def allocate(self, keys: List[Union[str, int]], sizes: List[int]):
        n = len(keys)
        offsets: List[Optional[int]] = [None] * n
        add_sizes: List[Optional[int]] = [None] * n

        mem_existing = self.mem_storage.lookup(keys)
        disk_existing = self.disk_storage.lookup(keys)

        new_indices = [i for i in range(n) if not mem_existing[i] and not disk_existing[i]]
        if not new_indices:
            return offsets, add_sizes

        new_keys = [keys[i] for i in new_indices]
        new_sizes = [sizes[i] for i in new_indices]

        disk_paths, _ = self.disk_storage.allocate(new_keys, new_sizes)

        disk_fail_idx = [j for j, p in enumerate(disk_paths) if p is None]
        if disk_fail_idx:
            evict_keys = [new_keys[j] for j in disk_fail_idx]
            evict_sizes = [new_sizes[j] for j in disk_fail_idx]
            total_evict = sum(evict_sizes)
            if total_evict > 0 and self.disk_storage.evict_to_free(total_evict):
                retry_paths, _ = self.disk_storage.allocate(evict_keys, evict_sizes)
                for j, p in zip(disk_fail_idx, retry_paths):
                    disk_paths[j] = p

        disk_ok_idx = [j for j, p in enumerate(disk_paths) if p is not None]
        disk_fail_final = [j for j, p in enumerate(disk_paths) if p is None]
        for j in disk_fail_final:
            offsets[new_indices[j]] = None
            add_sizes[new_indices[j]] = None

        if not disk_ok_idx:
            return offsets, add_sizes

        ok_orig = [new_indices[j] for j in disk_ok_idx]
        ok_keys = [new_keys[j] for j in disk_ok_idx]
        ok_sizes = [new_sizes[j] for j in disk_ok_idx]

        try:
            mem_offsets, mem_sizes = self.mem_storage.allocate(ok_keys, ok_sizes)
        except Exception:
            self.disk_storage.delete(ok_keys)
            raise

        mem_ok_orig = []
        mem_fail_orig = []
        for idx, orig in enumerate(ok_orig):
            if mem_offsets[idx] is not None:
                offsets[orig] = mem_offsets[idx]
                add_sizes[orig] = mem_sizes[idx]
                mem_ok_orig.append(orig)
            else:
                mem_fail_orig.append((orig, ok_keys[idx], ok_sizes[idx], idx))

        if mem_fail_orig:
            fail_keys = [k for _, k, _, _ in mem_fail_orig]
            fail_present = self.mem_storage.lookup(fail_keys)
            genuine_fail = [(orig, key, size) for slot, (orig, key, size, _) in enumerate(mem_fail_orig) if not fail_present[slot]]
            dup_fail = [(orig, key) for slot, (orig, key, _, _) in enumerate(mem_fail_orig) if fail_present[slot]]

            for orig, key in dup_fail:
                self.disk_storage.delete([key])
                offsets[orig] = None
                add_sizes[orig] = None

            if genuine_fail:
                evict_to_fit = getattr(self.mem_storage, "evict_to_fit", None)
                mem_total = self.mem_storage.get_storage_space()
                groups: list[list[tuple]] = []
                cur_group: list[tuple] = []
                cur_bytes = 0
                for orig, key, size in genuine_fail:
                    if size > mem_total:
                        self.disk_storage.delete([key])
                        offsets[orig] = None
                        add_sizes[orig] = None
                        continue
                    if cur_bytes + size > mem_total:
                        if cur_group:
                            groups.append(cur_group)
                        cur_group = [(orig, key, size)]
                        cur_bytes = size
                    else:
                        cur_group.append((orig, key, size))
                        cur_bytes += size
                if cur_group:
                    groups.append(cur_group)

                for grp in groups:
                    g_keys = [key for _, key, _ in grp]
                    g_sizes = [size for _, _, size in grp]
                    can = evict_to_fit(g_sizes) if evict_to_fit is not None else self.mem_storage.evict_to_free(sum(g_sizes))
                    if not can:
                        for orig, key, _ in grp:
                            self.disk_storage.delete([key])
                            offsets[orig] = None
                            add_sizes[orig] = None
                        continue
                    mo, ms = self.mem_storage.allocate(g_keys, g_sizes)
                    for idx, (orig, key, _) in enumerate(grp):
                        if mo[idx] is not None:
                            offsets[orig] = mo[idx]
                            add_sizes[orig] = ms[idx]
                        else:
                            self.disk_storage.delete([key])
                            offsets[orig] = None
                            add_sizes[orig] = None

        return offsets, add_sizes

    def finish_put(self, keys: List[Union[str, int]]):
        mem_metas = self.mem_storage.get_meta(keys, require_valid=False, protect_read=False)
        disk_metas = self.disk_storage.get_meta(keys, require_valid=False)
        self.mem_storage.finish_put(keys)
        with self._pending_lock:
            for key in keys:
                self.protected_disk_write_keys.add(str(key))
        futures = []
        for idx, key in enumerate(keys):
            mem_offset, mem_size = mem_metas[idx]
            disk_path, _ = disk_metas[idx]
            futures.append(self.copy_executor.submit(self._background_finish_disk_put, key, disk_path, mem_offset, mem_size))
        with self._pending_lock:
            for idx, key in enumerate(keys):
                self.pending_disk_writes[str(key)] = futures[idx]
                self.protected_disk_write_keys.discard(str(key))

    # ------------------------------------------------------------------ #
    # read path
    # ------------------------------------------------------------------ #
    def _lookup_key_size(self, key: Union[str, int]) -> Optional[int]:
        """Best-effort size lookup for lease quota check.

        Tries mem index first (valid entries only), then disk index.
        Returns None if the key exists in neither. The disk path uses lookup
        (not get_meta) so a valid disk key's ts is touched: with a non-zero
        protected_window this pins the key against eviction for the rest of
        get_meta (acquire_lease + _promote_from_disk), closing the TOCTOU
        window where a concurrent evict_to_free could delete it between this
        size check and the promote.
        """
        key = str(key)
        meta = self.mem_storage.index.get([key])[0]
        if meta is not None and meta.valid:
            return meta.size
        return self.disk_storage.lookup([key])[0]

    def _lookup_key_sizes(self, keys: List[Union[str, int]]) -> List[Optional[int]]:
        """Batched _lookup_key_size: one mem index get + one disk lookup,
        regardless of N.

        Like ``_lookup_key_size``, the disk leg uses ``lookup`` (not
        ``get_meta``) so each valid disk key's ts is touched, pinning it for
        the subsequent promote via the protected_window.
        """
        if not keys:
            return []
        sizes: List[Optional[int]] = [None] * len(keys)
        mem_metas = self.mem_storage.index.get(keys)
        missing_idx: List[int] = []
        for i, meta in enumerate(mem_metas):
            if meta is not None and meta.valid:
                sizes[i] = meta.size
            else:
                missing_idx.append(i)
        if missing_idx:
            missing_keys = [keys[i] for i in missing_idx]
            disk_sizes = self.disk_storage.lookup(missing_keys)
            for j, i in enumerate(missing_idx):
                sizes[i] = disk_sizes[j]
        return sizes

    def get_meta(self, keys: List[Union[str, int]], *, touch: bool = False, require_valid: bool = True):
        if touch and require_valid:
            mem_metas = self.mem_storage.index.touch(keys)
        else:
            mem_metas = self.mem_storage.index.get(keys)

        results: List[Optional[Tuple[int, int]]] = [None] * len(keys)
        disk_idx: List[int] = []
        leased_keys: List[Union[str, int]] = []
        try:
            for i, meta in enumerate(mem_metas):
                if meta is not None and (not require_valid or meta.valid):
                    self._acquire_lease(keys[i], meta.size, timeout=None)
                    leased_keys.append(keys[i])
                    results[i] = (meta.offset, meta.size)
                else:
                    disk_idx.append(i)

            if not disk_idx:
                return results

            disk_keys = [keys[i] for i in disk_idx]
            self._wait_pending_disk_writes(disk_keys)
            disk_sizes = self.disk_storage.lookup(disk_keys)
            for j, i in enumerate(disk_idx):
                if disk_sizes[j] is None:
                    raise KeyError(f"Key '{keys[i]}' not found in disk storage.")
            # Acquire every disk-miss lease up front so the freshly-promoted
            # keys are protected the instant finish_put runs.
            for j, i in enumerate(disk_idx):
                self._acquire_lease(keys[i], disk_sizes[j], timeout=None)
                leased_keys.append(keys[i])
            promoted = self._promote_from_disk_batch(disk_keys, disk_sizes)
            for j, i in enumerate(disk_idx):
                results[i] = promoted[keys[i]]

            return results
        except BaseException:
            if leased_keys:
                self.release_read(leased_keys)
            raise

    def get_readable_prefix_length(self, keys: List[Union[str, int]], sizes: List[int]) -> int:
        self._wait_pending_disk_writes(keys)
        promote_sizes = []
        readable_count = 0
        for key, size in zip(keys, sizes):
            if self.mem_storage.lookup([key])[0]:
                readable_count += 1
                continue

            try:
                self.disk_storage.get_meta([key], require_valid=True)
            except KeyError:
                break

            promote_sizes.append(size)
            if self.mem_storage.get_allocatable_prefix_length(promote_sizes) < len(promote_sizes):
                if not self.mem_storage.evict_to_free(sum(promote_sizes)):
                    break
                if self.mem_storage.get_allocatable_prefix_length(promote_sizes) < len(promote_sizes):
                    break
            readable_count += 1
        return readable_count

    def delete(self, keys: List[Union[str, int]]):
        if not keys:
            return
        self._wait_pending_disk_writes(keys)
        self.mem_storage.delete(keys)
        self.disk_storage.delete(keys)
        # Drop any lingering leases on deleted keys so they stop consuming
        # the byte quota.
        with self._read_lease_cond:
            notify = False
            for key in keys:
                key = str(key)
                if self.active_read_leases.pop(key, None) is not None:
                    size = self._read_lease_sizes.pop(key, 0)
                    self._read_lease_bytes -= size
                    if self._read_lease_bytes < 0:
                        self._read_lease_bytes = 0
                    notify = True
            if notify:
                self._read_lease_cond.notify_all()

    def clean(self):
        self._wait_pending_disk_writes()
        self.mem_storage.clean()
        self.disk_storage.clean()
        with self._read_lease_cond:
            self.active_read_leases.clear()
            self._read_lease_sizes.clear()
            self._read_lease_bytes = 0
            self._read_lease_cond.notify_all()

    def clean_local(self):
        """Reset local physical storage (mem + disk) without clearing indexes."""
        self._wait_pending_disk_writes()
        self.mem_storage.clean_local()
        self.disk_storage.clean_local()
        with self._read_lease_cond:
            self.active_read_leases.clear()
            self._read_lease_sizes.clear()
            self._read_lease_bytes = 0
            self._read_lease_cond.notify_all()

    def close(self):
        try:
            self._wait_pending_disk_writes()
            self.copy_executor.shutdown(wait=True)
        finally:
            try:
                self.mem_storage.close()
            finally:
                self.disk_storage.close()
