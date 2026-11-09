import ctypes
import os
import threading
import time
import numpy as np
import torch
from typing import List, Optional

from ... import cache_ops
from ...utils import (
    import_kv_tensor,
    release_ipc_tensors,
    device_index_from_uuid,
    import_event,
    make_interprocess_event,
    export_event,
    get_gpu_host_runtime,
    HOST_MEMORY_ALREADY_REGISTERED,
)
from ...logging import get_logger

logger = get_logger(__name__)

POSITION_SIZE = 4096
INT64_SIZE = 8
READ_CHUNK_LEASE_FRAC = 0.4
REGISTRATION_CHUNK_BYTES = 1024**3


def _resolve_staging_bytes() -> int:
    mb = int(os.environ.get("DARIC_KV_IPC_STAGING_MB", "256"))
    if mb <= 0:
        raise ValueError(f"DARIC_KV_IPC_STAGING_MB must be positive, got {mb}")
    return mb * 1024 * 1024


STAGING_BYTES = _resolve_staging_bytes()


class HostMemoryRegistration:
    def __init__(self, runtime, chunk_bytes: int):
        self.runtime = runtime
        self.chunk_bytes = chunk_bytes
        self.addresses: list[int] = []

    def register(self, buf):
        if self.addresses:
            raise RuntimeError("Cannot register store shm while previous registrations remain.")
        address = ctypes.addressof(ctypes.c_char.from_buffer(buf))
        page_size = os.sysconf("SC_PAGE_SIZE")
        if address % page_size or self.chunk_bytes <= 0 or self.chunk_bytes % page_size:
            raise ValueError("Store shm and registration chunk size must be page-aligned.")
        size = buf.nbytes
        if size <= 0:
            raise ValueError("Store shm must not be empty.")
        started = time.monotonic()
        logger.info(
            "Registering store shm for GPU access (%.2f GiB, chunks of %.2f GiB)...",
            size / 1024**3, self.chunk_bytes / 1024**3,
        )
        last_percent = -1
        try:
            for offset in range(0, size, self.chunk_bytes):
                length = min(self.chunk_bytes, size - offset)
                ret = self.runtime.register(address + offset, length)
                if ret == HOST_MEMORY_ALREADY_REGISTERED:
                    raise RuntimeError(
                        "Store shm registration overlaps existing pinned memory; "
                        "cannot verify the entire chunk is registered."
                    )
                if ret != 0:
                    raise RuntimeError(
                        f"{self.runtime.prefix}HostRegister failed at offset {offset} "
                        f"with code {ret}."
                    )
                self.addresses.append(address + offset)
                completed = offset + length
                percent = completed * 100 // size
                if percent > last_percent or completed == size:
                    logger.info(
                        "Store shm GPU registration: %.1f%% (%d / %d bytes), elapsed %.1f s.",
                        100 * completed / size, completed, size, time.monotonic() - started,
                    )
                    last_percent = percent
        except BaseException:
            self.unregister()
            raise
        logger.info("Registered store shm for GPU access in %.1f s.", time.monotonic() - started)

    def unregister(self):
        failed = []
        for address in reversed(self.addresses):
            try:
                ret = self.runtime.unregister(address)
                if ret != 0:
                    logger.error(
                        "%sHostUnregister failed for store shm address %s with code %s.",
                        self.runtime.prefix, address, ret,
                    )
                    failed.append(address)
            except Exception:
                failed.append(address)
                logger.exception("HostUnregister failed for store shm address %s.", address)
        self.addresses = list(reversed(failed))
        if failed:
            raise RuntimeError(f"Failed to unregister {len(failed)} store shm chunk(s).")
        logger.info("Unregistered store shm from GPU.")


def _split_registration_copies(stg_offsets, shm_offsets, sizes, chunk_bytes):
    if not (len(stg_offsets) == len(shm_offsets) == len(sizes)):
        raise ValueError("Copy offsets and sizes must have the same length.")
    if chunk_bytes <= 0:
        raise ValueError("Registration chunk size must be positive.")
    split_stg, split_shm, split_sizes = [], [], []
    for stg_offset, shm_offset, size in zip(stg_offsets, shm_offsets, sizes):
        if stg_offset < 0 or shm_offset < 0 or size < 0:
            raise ValueError("Copy offsets and sizes must be non-negative.")
        while size:
            length = min(size, chunk_bytes - shm_offset % chunk_bytes)
            split_stg.append(stg_offset)
            split_shm.append(shm_offset)
            split_sizes.append(length)
            stg_offset += length
            shm_offset += length
            size -= length
    return split_stg, split_shm, split_sizes


class RankIPCContext:
    MAX_BATCH_SIZE = 4

    def __init__(self, device_index: int, layer_names: List[str], group_idxs: List[int],
                 layer_bpbs: List[int], page_firsts: List[bool]):
        self.device_index = device_index
        torch.cuda.set_device(device_index)

        self.read_stream = torch.cuda.Stream(device=f"cuda:{device_index}", priority=1)
        self.write_stream = torch.cuda.Stream(device=f"cuda:{device_index}", priority=1)
        self.read_copier = cache_ops.CacheCopier()   # store = shm (scatter)
        self.write_copier = cache_ops.CacheCopier()  # store = staging (gather)
        self.layer_names = list(layer_names)
        self.group_idxs = list(group_idxs)
        self.group_idxs_map = {n: g for n, g in zip(self.layer_names, self.group_idxs)}
        self.layer_bpbs = list(layer_bpbs)
        self.page_firsts = list(page_firsts)
        self.imported_tensors: list[torch.Tensor] = []  # keep IPC refs alive
        self._read_store_registered = False

        self.staging = torch.empty(STAGING_BYTES, dtype=torch.uint8, device=device_index)
        self.read_staging = torch.empty(STAGING_BYTES, dtype=torch.uint8, device=device_index)

        self._lifecycle = threading.Condition()
        self._inflight = 0
        self._closed_flag = False

    def acquire(self):
        with self._lifecycle:
            if self._closed_flag:
                raise RuntimeError("KV IPC: rank context is closed")
            self._inflight += 1

    def release(self):
        with self._lifecycle:
            self._inflight -= 1
            if self._inflight == 0:
                self._lifecycle.notify_all()

    def register_layers(self, shm_tensor: torch.Tensor, ipc_wrappers: List[dict]):
        
        torch.cuda.set_device(self.device_index)
        for name, wrapper, is_pf in zip(self.layer_names, ipc_wrappers, self.page_firsts):
            t = import_kv_tensor(wrapper)
            self.imported_tensors.append(t)
            self.read_copier.register_layer(name, t, bool(is_pf))
            self.write_copier.register_layer(name, t, bool(is_pf))
        if not self._read_store_registered:
            self.read_copier.register_store(self.read_staging)
            self._read_store_registered = True
        self.write_copier.register_store(self.staging)

    def close(self):
        with self._lifecycle:
            self._closed_flag = True
            while self._inflight > 0:
                logger.warning(
                    "KV IPC: closing ctx (dev=%s) with %s in-flight op(s); waiting",
                    self.device_index, self._inflight,
                )
                self._lifecycle.wait(timeout=5.0)
        tensors = self.imported_tensors
        self.read_copier = None
        self.write_copier = None
        self.staging = None
        self.read_staging = None
        self.imported_tensors = []
        release_ipc_tensors(tensors)
        try:
            torch.cuda.set_device(self.device_index)
            logger.info("KV IPC: after unregister rank dev=%s allocated=%.1fMB reserved=%.1fMB",
                        self.device_index,
                        torch.cuda.memory_allocated(self.device_index) / 1e6,
                        torch.cuda.memory_reserved(self.device_index) / 1e6)
        except Exception:
            pass


class KVIOManager:
    def __init__(self, storage_manager):
        self.storage_manager = storage_manager
        mem_storage = getattr(storage_manager, "mem_storage", storage_manager)
        self.shm = mem_storage.shm
        self.shm_size = mem_storage.shm_size
        self.shm_tensor = torch.frombuffer(self.shm.buf, dtype=torch.uint8, count=self.shm_size)

        self._shm_registered = False
        self._shm_registration: Optional[HostMemoryRegistration] = None
        self.rank_contexts: dict[int, RankIPCContext] = {}
        self._lock = threading.Lock()
        self._device_index_from_uuid = device_index_from_uuid
        self._rank_owner_pids: dict[int, int] = {}
        self._closed = threading.Event()
        self._owner_monitor = threading.Thread(
            target=self._monitor_rank_owners,
            name="kv-ipc-owner-monitor",
            daemon=True,
        )
        self._owner_monitor.start()

    # ------------------------------------------------------------------ #
    # registration
    # ------------------------------------------------------------------ #
    def register_kv_cache(
        self, rank: int, ipc_wrappers: List[dict], layer_names: List[str],
        group_idxs: List[int], layer_bpbs: List[int], page_firsts: List[bool],
        owner_pid: Optional[int] = None,
    ):
        device_index = self._device_index_from_uuid(ipc_wrappers[0]["device_uuid"])
        with self._lock:
            old_ctx = self.rank_contexts.get(rank)
            ctx = RankIPCContext(device_index, layer_names, group_idxs, layer_bpbs, page_firsts)
            self.rank_contexts[rank] = ctx
            if owner_pid is not None:
                self._rank_owner_pids[rank] = owner_pid
            if not self._shm_registered:
                try:
                    if self._shm_registration is None:
                        self._shm_registration = HostMemoryRegistration(
                            get_gpu_host_runtime(), REGISTRATION_CHUNK_BYTES
                        )
                    elif self._shm_registration.addresses:
                        self._shm_registration.unregister()
                    self._shm_registration.register(self.shm.buf)
                    self._shm_registered = True
                except Exception:
                    self.rank_contexts.pop(rank, None)
                    self._rank_owner_pids.pop(rank, None)
                    ctx.close()
                    raise

        # Close the replaced ctx outside the service lock: close() may wait for
        # in-flight store/load ops, which must not block unrelated RPCs.
        if old_ctx is not None:
            logger.info("KV IPC: rank %s re-registered; closing previous ctx", rank)
            old_ctx.close()
        try:
            ctx.register_layers(self.shm_tensor, ipc_wrappers)
        except Exception:
            # Don't leave a half-initialized ctx behind: a later
            # submit_store/load would otherwise run against partially
            # registered layers.
            with self._lock:
                if self.rank_contexts.get(rank) is ctx:
                    self.rank_contexts.pop(rank, None)
                    self._rank_owner_pids.pop(rank, None)
            ctx.close()
            raise

    def unregister_kv_cache(self, rank: int):
        with self._lock:
            ctx = self.rank_contexts.pop(rank, None)
            self._rank_owner_pids.pop(rank, None)
        if ctx is not None:
            ctx.close()
            logger.info(f"KV IPC: unregistered rank {rank}.")

    @staticmethod
    def _pid_is_alive(pid: int) -> bool:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _monitor_rank_owners(self):
        while not self._closed.wait(timeout=1.0):
            with self._lock:
                owners = list(self._rank_owner_pids.items())
            for rank, pid in owners:
                if self._pid_is_alive(pid):
                    continue
                with self._lock:
                    # Re-verify under the lock: if the rank re-registered under a
                    # new owner since the snapshot, the dead pid is stale and the
                    # live ctx must not be torn down.
                    if self._rank_owner_pids.get(rank) != pid:
                        continue
                logger.info(
                    "KV IPC: owner process %s for rank %s exited; unregistering",
                    pid,
                    rank,
                )
                self.unregister_kv_cache(rank)

    # ------------------------------------------------------------------ #
    # size / offset math (ported from LocalStorageClient)
    # ------------------------------------------------------------------ #
    def _per_meta_cache_size(self, ctx: RankIPCContext, block_ids_list, layer_name: Optional[str]) -> List[int]:
        if layer_name is not None:
            grp_idx = ctx.group_idxs_map[layer_name]
            bpb = ctx.layer_bpbs[ctx.layer_names.index(layer_name)]
            return [bpb * len(b[grp_idx]) for b in block_ids_list]
        sizes = []
        for b in block_ids_list:
            total = 0
            for li in range(len(ctx.layer_names)):
                g = ctx.group_idxs[li]
                total += ctx.layer_bpbs[li] * len(b[g])
            sizes.append(total)
        return sizes

    def _build_copy_plan(self, ctx: RankIPCContext, valid_offsets: List[int],
                         block_ids_list_valid, layer_name: Optional[str]):
        device = f"cuda:{ctx.device_index}"
        N = len(valid_offsets)
        flat_list = []
        indices_offsets = [0]

        if layer_name is not None:
            grp_idx = ctx.group_idxs_map[layer_name]
            li = ctx.layer_names.index(layer_name)
            bpb = ctx.layer_bpbs[li]
            all_offsets = []
            all_sizes = []
            for i in range(N):
                bids = block_ids_list_valid[i][grp_idx]
                flat_list.extend(bids)
                indices_offsets.append(len(flat_list))
                all_offsets.append(valid_offsets[i])
                all_sizes.append(bpb * len(bids))
            all_layer_names = [layer_name] * N
        else:
            L = len(ctx.layer_names)
            group_idxs = ctx.group_idxs
            layer_bpbs = ctx.layer_bpbs
            all_offsets = []
            all_sizes = []
            for i in range(N):
                sub = block_ids_list_valid[i]
                base = valid_offsets[i]
                intra = 0
                for j in range(L):
                    g = group_idxs[j]
                    bids = sub[g]
                    flat_list.extend(bids)
                    indices_offsets.append(len(flat_list))
                    sz = layer_bpbs[j] * len(bids)
                    all_offsets.append(base + intra)
                    all_sizes.append(sz)
                    intra += sz
            all_layer_names = list(ctx.layer_names) * N

        if flat_list:
            flat = torch.from_numpy(np.asarray(flat_list, dtype=np.int32)).to(device)
        else:
            flat = torch.empty(0, dtype=torch.int32, device=device)
        return flat, all_offsets, indices_offsets, all_layer_names, all_sizes

    def _position_offsets(self, valid_offsets: List[int], cache_sizes_valid: List[int],
                          layer_name: Optional[str]) -> List[int]:
        if layer_name is not None:
            return [o + s for o, s in zip(valid_offsets, cache_sizes_valid)]
        off_arr = np.array(valid_offsets, dtype=np.int64)
        # cache_sizes_valid is per-meta total; base offset is valid_offsets[i].
        return (off_arr + np.array(cache_sizes_valid, dtype=np.int64)).tolist()

    def _write_positions(self, position_offsets: List[int], saved_positions: List[int]):
        N = len(position_offsets)
        if N == 0:
            return
        shm_flat = self.shm_tensor.view(-1)
        payload = torch.zeros((N, POSITION_SIZE), dtype=torch.uint8)
        pos_tensor = torch.tensor(saved_positions, dtype=torch.int64)
        payload[:, :INT64_SIZE] = pos_tensor.view(torch.uint8).reshape(N, INT64_SIZE)
        off_tensor = torch.tensor(position_offsets, dtype=torch.long)
        idx_tensor = off_tensor.view(-1, 1) + torch.arange(POSITION_SIZE, dtype=torch.long).view(1, -1)
        shm_flat[idx_tensor.view(-1)] = payload.view(-1)

    def _read_positions(self, position_offsets: List[int]) -> List[int]:
        N = len(position_offsets)
        if N == 0:
            return []
        shm_flat = self.shm_tensor.view(-1)
        off_tensor = torch.tensor(position_offsets, dtype=torch.long)
        idx_tensor = off_tensor.view(-1, 1) + torch.arange(INT64_SIZE, dtype=torch.long).view(1, -1)
        gathered = shm_flat[idx_tensor.view(-1)].view(N, INT64_SIZE)
        return gathered.view(torch.int64).view(-1).tolist()

    # ------------------------------------------------------------------ #
    # write (store)
    # ------------------------------------------------------------------ #
    def submit_store(self, rank: int, keys: List[str], block_ids_list, layer_name: Optional[str],
                     saved_positions: List[int], producer_event_handle: bytes) -> None:
        ctx = self._get_ctx(rank)
        ctx.acquire()
        try:
            self._do_store(rank, ctx, list(keys), block_ids_list, layer_name,
                           list(saved_positions), producer_event_handle)
        finally:
            ctx.release()

    def _do_store(self, rank, ctx, keys, block_ids_list, layer_name,
                  saved_positions, producer_event_handle):
        allocated_keys = []
        try:
            torch.cuda.set_device(ctx.device_index)
            producer_ev = import_event(producer_event_handle, ctx.device_index)
            ctx.write_stream.wait_event(producer_ev)

            cache_sizes = self._per_meta_cache_size(ctx, block_ids_list, layer_name)
            offsets, _ = self.storage_manager.allocate(keys, [s + POSITION_SIZE for s in cache_sizes])
            valid_indices = [i for i, o in enumerate(offsets) if o is not None]
            if not valid_indices:
                return  # nothing allocated

            valid_offsets = [offsets[i] for i in valid_indices]
            valid_keys = [keys[i] for i in valid_indices]
            allocated_keys = valid_keys
            valid_block_ids = [block_ids_list[i] for i in valid_indices]
            valid_cache_sizes = [cache_sizes[i] for i in valid_indices]
            valid_saved_pos = [saved_positions[i] for i in valid_indices]

            B = ctx.MAX_BATCH_SIZE
            cap = ctx.staging.numel()
            L_per_meta = len(ctx.layer_names) if layer_name is None else 1
            # Greedy byte-bounded batches: each batch must fit the fixed
            # staging buffer (count cap MAX_BATCH_SIZE also applies).
            i = 0
            n_valid = len(valid_indices)
            while i < n_valid:
                j = i
                acc = 0
                while j < n_valid and j - i < B and acc + valid_cache_sizes[j] <= cap:
                    acc += valid_cache_sizes[j]
                    j += 1
                if j == i:
                    raise RuntimeError(
                        f"KV IPC: single meta is {valid_cache_sizes[i]} bytes, larger than "
                        f"the {cap}-byte staging buffer; increase DARIC_KV_IPC_STAGING_MB"
                    )
                sub_offsets = valid_offsets[i:j]
                sub_block_ids = valid_block_ids[i:j]
                sub_cache_sizes = valid_cache_sizes[i:j]
                sub_flat, sub_shm_offsets, sub_indices_offsets, sub_layer_names, sub_sizes = \
                    self._build_copy_plan(ctx, sub_offsets, sub_block_ids, layer_name)
                stg_offsets = []
                acc = 0
                for sz in sub_sizes:
                    stg_offsets.append(acc)
                    acc += sz
                ctx.write_copier.gather_pages(
                    sub_layer_names, stg_offsets, sub_sizes, sub_flat,
                    sub_indices_offsets, ctx.write_stream.cuda_stream,
                )
                # Per-meta (chunk) DMA regions: coalesce each meta's L layers.
                n_meta = len(sub_offsets)
                chunk_stg = [stg_offsets[k * L_per_meta] for k in range(n_meta)]
                chunk_shm = [sub_shm_offsets[k * L_per_meta] for k in range(n_meta)]
                copy_stg, copy_shm, copy_sizes = _split_registration_copies(
                    chunk_stg, chunk_shm, sub_cache_sizes, REGISTRATION_CHUNK_BYTES
                )
                cache_ops.batched_memcpy_d2h(
                    ctx.staging, self.shm_tensor, copy_stg, copy_shm,
                    copy_sizes, ctx.write_stream.cuda_stream,
                )
                i = j

            ctx.write_stream.synchronize()
            self._write_positions(
                self._position_offsets(valid_offsets, valid_cache_sizes, layer_name),
                valid_saved_pos,
            )
            self.storage_manager.finish_put(valid_keys)
        except Exception:
            if allocated_keys:
                try:
                    self.storage_manager.delete(allocated_keys)
                except Exception:
                    logger.exception("KV IPC: failed to reclaim store allocation (rank=%s)", rank)
            logger.exception("KV IPC: _do_store failed (rank=%s)", rank)
            raise

    # ------------------------------------------------------------------ #
    # read (load)
    # ------------------------------------------------------------------ #
    def submit_load(self, rank: int, keys: List[str], block_ids_list, layer_name: Optional[str],
                    producer_event_handle: bytes):
        ctx = self._get_ctx(rank)
        ctx.acquire()
        try:
            return self._do_load(rank, ctx, keys, block_ids_list, layer_name,
                                 producer_event_handle)
        finally:
            ctx.release()

    def _do_load(self, rank: int, ctx: RankIPCContext, keys: List[str], block_ids_list,
                 layer_name: Optional[str], producer_event_handle: bytes):
        torch.cuda.set_device(ctx.device_index)
        producer_ev = import_event(producer_event_handle, ctx.device_index)
        ctx.read_stream.wait_event(producer_ev)

        chunks = self._plan_read_chunks(keys, self._read_chunk_budget(ctx))
        saved: List[Optional[int]] = [None] * len(keys)
        total_valid = 0
        for chunk_idx in chunks:
            chunk_keys = [keys[i] for i in chunk_idx]
            chunk_blocks = [block_ids_list[i] for i in chunk_idx]

            results = self.storage_manager.get_meta(
                list(chunk_keys), touch=True, require_valid=True)
            try:
                chunk_offsets = [r[0] for r in results]
                chunk_cache_sizes = [r[1] - POSITION_SIZE for r in results]
                total_valid += len(chunk_offsets)

                flat, all_offsets, indices_offsets, all_layer_names, all_sizes = \
                    self._build_copy_plan(ctx, chunk_offsets, chunk_blocks, layer_name)

                # Copy-engine H2D of each meta's contiguous shm region into
                # GPU staging, then scatter from staging.
                L_per_meta = len(all_sizes) // max(len(chunk_offsets), 1)
                n_meta = len(chunk_offsets)
                stg_offsets = []
                acc = 0
                for sz in all_sizes:
                    stg_offsets.append(acc)
                    acc += sz
                chunk_shm = [all_offsets[k * L_per_meta] for k in range(n_meta)]
                chunk_stg = [stg_offsets[k * L_per_meta] for k in range(n_meta)]
                chunk_sizes = [
                    sum(all_sizes[k * L_per_meta:(k + 1) * L_per_meta])
                    for k in range(n_meta)
                ]
                if acc > ctx.read_staging.numel():
                    raise RuntimeError(
                        f"KV IPC: read chunk is {acc} bytes, larger than the "
                        f"{ctx.read_staging.numel()}-byte staging buffer; increase "
                        "DARIC_KV_IPC_STAGING_MB"
                    )
                copy_stg, copy_shm, copy_sizes = _split_registration_copies(
                    chunk_stg, chunk_shm, chunk_sizes, REGISTRATION_CHUNK_BYTES
                )
                cache_ops.batched_memcpy_h2d(
                    self.shm_tensor, ctx.read_staging, copy_shm, copy_stg,
                    copy_sizes, ctx.read_stream.cuda_stream,
                )
                ctx.read_copier.scatter_pages(
                    all_layer_names, stg_offsets, all_sizes, flat, indices_offsets,
                    ctx.read_stream.cuda_stream,
                )

                ctx.read_stream.synchronize()
                chunk_saved = self._read_positions(
                    self._position_offsets(chunk_offsets, chunk_cache_sizes, layer_name))
            finally:
                self.storage_manager.release_read(list(chunk_keys))
            for j, i in enumerate(chunk_idx):
                saved[i] = chunk_saved[j]

        comp = make_interprocess_event(ctx.device_index)
        comp.record(ctx.read_stream)
        return export_event(comp), saved

    def _read_chunk_budget(self, ctx: RankIPCContext) -> Optional[int]:
        quota = getattr(self.storage_manager, "_read_lease_capacity", None)
        lease_budget = int(quota * READ_CHUNK_LEASE_FRAC) if quota else None
        cap = ctx.read_staging.numel()
        return min(lease_budget, cap) if lease_budget is not None else cap

    def _plan_read_chunks(self, keys: List[str], budget: Optional[int]) -> List[List[int]]:
        n = len(keys)
        if budget is None or n == 0:
            return [list(range(n))]
        budget = max(budget, 1)
        # One batched size lookup instead of N per-key calls. Fall back to
        # the single-key path only if the storage manager pre-dates
        # _lookup_key_sizes.
        lookup_sizes = getattr(self.storage_manager, "_lookup_key_sizes", None)
        if lookup_sizes is not None:
            sizes = lookup_sizes(keys)
        else:
            lookup_size = getattr(self.storage_manager, "_lookup_key_size", None)
            sizes = [lookup_size(k) if lookup_size is not None else None for k in keys]

        chunks: List[List[int]] = []
        cur: List[int] = []
        cur_bytes = 0
        for i, key in enumerate(keys):
            size = sizes[i]
            if size is None:
                # Unknown/missing key: flush the run and isolate it so get_meta
                # raises KeyError against the right key instead of stalling.
                if cur:
                    chunks.append(cur); cur = []; cur_bytes = 0
                chunks.append([i])
                continue
            if cur and cur_bytes + size > budget:
                chunks.append(cur); cur = []; cur_bytes = 0
            cur.append(i)
            cur_bytes += size
        if cur:
            chunks.append(cur)
        return chunks

    # ------------------------------------------------------------------ #
    # misc
    # ------------------------------------------------------------------ #
    def close(self):
        self._closed.set()
        self._owner_monitor.join(timeout=2.0)
        with self._lock:
            ctxs = list(self.rank_contexts.values())
            self.rank_contexts.clear()
        for ctx in ctxs:
            ctx.close()
        if self._shm_registration is not None and self._shm_registration.addresses:
            self._shm_registration.unregister()
        self._shm_registered = False
        self._shm_registration = None

    def _get_ctx(self, rank: int) -> RankIPCContext:
        ctx = self.rank_contexts.get(rank)
        if ctx is None:
            raise RuntimeError(
                f"KV IPC: rank {rank} has not registered its KV cache "
                "(or was unregistered because its owner process exited)"
            )
        return ctx
