import torch
import numpy as np
from collections import defaultdict
from typing import List, Optional, TYPE_CHECKING

from .policy import DARICPolicy
from ..configs import AdaptionConfig, ProcessLocalConfig

if TYPE_CHECKING:
    from ..reuse_controller.kvcache import KVCache

from ..logging import get_logger

logger = get_logger(__name__)

import triton
import triton.language as tl


def _get_rerotate_page_size(kv_layer: torch.Tensor) -> int:
    if kv_layer.ndim in (4, 5):
        return int(kv_layer.shape[2])
    if kv_layer.ndim == 3:
        return int(kv_layer.shape[1])
    raise ValueError(
        f"Unsupported KV cache rank for rerotate: {kv_layer.ndim}"
    )


@triton.jit
def _fused_rerotate_kernel(
    kv_ptr,
    idx_ptr,
    cos_ptr,
    sin_ptr,
    total_len,
    stride_kv_b,
    stride_kv_s,
    stride_kv_h,
    PAGE_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_ROT: tl.constexpr,
    BLOCK_ROT: tl.constexpr,
    IS_NEOX: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_h = tl.program_id(1)

    if pid_t >= total_len:
        return

    idx = tl.load(idx_ptr + pid_t)
    blk = idx // PAGE_SIZE
    slot = idx % PAGE_SIZE
    head_offset = blk * stride_kv_b + slot * stride_kv_s + pid_h * stride_kv_h

    d_half_range = tl.arange(0, BLOCK_ROT)
    mask_half = d_half_range < HALF_ROT

    cs_offset = pid_t * HALF_ROT + d_half_range
    cos = tl.load(cos_ptr + cs_offset, mask=mask_half, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + cs_offset, mask=mask_half, other=0.0).to(tl.float32)

    if IS_NEOX:
        # --- NeoX Style (LLaMA) ---
        k1 = tl.load(kv_ptr + head_offset + d_half_range, mask=mask_half, other=0.0).to(tl.float32)
        k2 = tl.load(kv_ptr + head_offset + d_half_range + HALF_ROT, mask=mask_half, other=0.0).to(tl.float32)

        out1 = k1 * cos - k2 * sin
        out2 = k2 * cos + k1 * sin

        tl.store(kv_ptr + head_offset + d_half_range, out1.to(kv_ptr.dtype.element_ty), mask=mask_half)
        tl.store(kv_ptr + head_offset + d_half_range + HALF_ROT, out2.to(kv_ptr.dtype.element_ty), mask=mask_half)
    else:
        # --- GPT-J Style ---
        d_even = d_half_range * 2
        d_odd = d_half_range * 2 + 1
        mask_even = mask_half & (d_even < HEAD_DIM)
        mask_odd = mask_half & (d_odd < HEAD_DIM)

        k1 = tl.load(kv_ptr + head_offset + d_even, mask=mask_even, other=0.0).to(tl.float32)
        k2 = tl.load(kv_ptr + head_offset + d_odd, mask=mask_odd, other=0.0).to(tl.float32)

        out1 = k1 * cos - k2 * sin
        out2 = k2 * cos + k1 * sin

        tl.store(kv_ptr + head_offset + d_even, out1.to(kv_ptr.dtype.element_ty), mask=mask_even)
        tl.store(kv_ptr + head_offset + d_odd, out2.to(kv_ptr.dtype.element_ty), mask=mask_odd)


class Adapter:
    def __init__(self, config: AdaptionConfig):
        self.config = config
        self.num_layers = len(ProcessLocalConfig.cache_topo.get_all_layer_names())
        self.kvcaches = {}
        self.rotary_embeds = {}
        if config.recompute_policy == "daric":
            self.recompute_policy = DARICPolicy(
                config.recompute_avg_num, function=config.function, r=config.r)
        else:
            self.recompute_policy = None
        self.rerotate = config.rerotate

    def register_kvcaches(
        self, kvcaches: dict[str, torch.Tensor]
    ) -> None:
        self.kvcaches = kvcaches

    def register_rotary_embed(
        self, layer_name: str, rotary_embed: torch.nn.Module
    ) -> None:
        if not layer_name in self.rotary_embeds:
            self.rotary_embeds[layer_name] = rotary_embed

    def register_done(self):
        return len(self.rotary_embeds) == self.num_layers

    def apply_rerotate_multi(
        self,
        group_ptrs: List[List[int]],
        block_ids_list: List[tuple[list[int]]],
    ):
        if not self.rerotate:
            return
        assert len(group_ptrs) == len(block_ids_list)

        delta_list, len_list = [], [] 
        tok_spans = []
        for req_idx, group_ptr in enumerate(group_ptrs):
            n_groups = len(group_ptr) - 1
            if n_groups <= 2:
                continue
            base = group_ptr[0]
            n_tok = 0
            for i in range(1, n_groups - 1):
                glen = group_ptr[i + 1] - group_ptr[i]
                if glen > 0:
                    delta_list.append(group_ptr[i] - base)
                    len_list.append(glen)
                    n_tok += glen
            if n_tok > 0:
                tok_spans.append((req_idx, group_ptr[1] - base, n_tok))
        if len(tok_spans) == 0:
            return

        device = next(iter(self.kvcaches.values())).device
        positions = torch.from_numpy(
            np.repeat(np.asarray(delta_list, dtype=np.int64),
                      np.asarray(len_list, dtype=np.int64))
        ).to(device, non_blocking=True)

        layers_by_group = defaultdict(list)
        for cur_layer_name in self.kvcaches.keys():
            g_idx = ProcessLocalConfig.cache_topo.get_group_idx(cur_layer_name)
            layers_by_group[g_idx].append(cur_layer_name)

        for group_idx, layers in layers_by_group.items():
            first_layer_name = layers[0]
            kv_layer_first = self.kvcaches[first_layer_name]
            page_size = _get_rerotate_page_size(kv_layer_first)

            gather_parts = []
            for req_idx, start_tok, n_tok in tok_spans:
                layer_block_ids = block_ids_list[req_idx][group_idx]
                if isinstance(layer_block_ids, torch.Tensor):
                    bl = layer_block_ids.to("cpu", torch.long)
                else:
                    bl = torch.tensor(layer_block_ids, dtype=torch.long)
                flat_pos = torch.arange(start_tok, start_tok + n_tok, dtype=torch.long)
                gather_parts.append(bl[flat_pos // page_size] * page_size + flat_pos % page_size)
            gather_idx = torch.cat(gather_parts).to(device, non_blocking=True).contiguous()

            combined_cache = {}
            for cur_layer_name in layers:
                kv_layer = self.kvcaches[cur_layer_name]
                rotary = self.rotary_embeds[cur_layer_name]
                cos_sin_cache = rotary.cos_sin_cache

                cache_key = cos_sin_cache.data_ptr()
                if cache_key not in combined_cache:
                    cs = cos_sin_cache.index_select(0, positions)
                    c, s = cs.chunk(2, dim=-1)
                    combined_cache[cache_key] = (c.contiguous(), s.contiguous())
                combined_cos, combined_sin = combined_cache[cache_key]

                self._launch_triton_rerotate(
                    kv_layer,
                    cur_layer_name,
                    gather_idx,
                    combined_cos,
                    combined_sin,
                    rotary.rotary_dim,
                    rotary.is_neox_style,
                )
    
    def apply_rerotate_load(
        self,
        metas: List["KVCache"],
        layer_name: Optional[str] = None,
    ) -> None:
        if not self.rerotate:
            return

        plan_metas: List = []
        plan_new_pos: List[int] = []
        plan_old_pos: List[int] = []
        plan_chunk_lens: List[int] = []
        for meta in metas:
            old_pos = int(meta.saved_pos)
            new_pos = int(meta.ptr[0])
            chunk_len = int(meta.ptr[1] - meta.ptr[0])
            if chunk_len <= 0 or old_pos == new_pos:
                continue
            plan_metas.append(meta)
            plan_new_pos.append(new_pos)
            plan_old_pos.append(old_pos)
            plan_chunk_lens.append(chunk_len)

        if len(plan_metas) == 0:
            return

        layer_names = [layer_name] if layer_name is not None else list(self.kvcaches.keys())

        device = next(iter(self.kvcaches.values())).device
        chunk_lens_t = torch.tensor(plan_chunk_lens, dtype=torch.long, device=device)
        max_len = int(chunk_lens_t.max().item()) if len(chunk_lens_t) > 0 else 0
        if max_len == 0:
            return

        arange = torch.arange(max_len, device=device, dtype=torch.long)
        mask = arange.unsqueeze(0) < chunk_lens_t.unsqueeze(1)
        old_starts = torch.tensor(plan_old_pos, dtype=torch.long, device=device)
        new_starts = torch.tensor(plan_new_pos, dtype=torch.long, device=device)

        all_old_pos = (old_starts.unsqueeze(1) + arange.unsqueeze(0))[mask]
        all_new_pos = (new_starts.unsqueeze(1) + arange.unsqueeze(0))[mask]

        layers_by_group = defaultdict(list)
        for cur_layer_name in layer_names:
            if cur_layer_name in self.kvcaches:
                g_idx = ProcessLocalConfig.cache_topo.get_group_idx(cur_layer_name)
                layers_by_group[g_idx].append(cur_layer_name)

        for group_idx, layers in layers_by_group.items():
            block_id_chunks = []
            for meta in plan_metas:
                layer_block_ids = meta.block_ids[group_idx]
                if not isinstance(layer_block_ids, torch.Tensor):
                    layer_block_ids = torch.tensor(layer_block_ids, dtype=torch.long, device=device)
                else:
                    layer_block_ids = layer_block_ids.to(torch.long).to(device)
                block_id_chunks.append(layer_block_ids)

            first_layer_name = layers[0]
            kv_layer_first = self.kvcaches[first_layer_name]
            page_size = _get_rerotate_page_size(kv_layer_first)

            gather_idx_parts = [None] * len(block_id_chunks)
            for i, (block_ids, chunk_len) in enumerate(zip(block_id_chunks, plan_chunk_lens)):
                if chunk_len > 0:
                    pos = torch.arange(chunk_len, device=device)
                    block_idx = pos // page_size
                    local_pos = pos % page_size
                    gather_idx_parts[i] = block_ids[block_idx] * page_size + local_pos
            gather_idx = torch.cat([t for t in gather_idx_parts if t is not None]).contiguous()

            combined_cache = {}
            for cur_layer_name in layers:
                kv_layer = self.kvcaches[cur_layer_name]
                rotary = self.rotary_embeds[cur_layer_name]
                rotary_dim = rotary.rotary_dim
                cos_sin_cache = rotary.cos_sin_cache

                cache_key = cos_sin_cache.data_ptr()
                if cache_key not in combined_cache:
                    old_cos_sin = cos_sin_cache.index_select(0, all_old_pos)
                    new_cos_sin = cos_sin_cache.index_select(0, all_new_pos)
                    old_cos, old_sin = old_cos_sin.chunk(2, dim=-1)
                    new_cos, new_sin = new_cos_sin.chunk(2, dim=-1)

                    combined_cos = (new_cos * old_cos + new_sin * old_sin).contiguous()
                    combined_sin = (new_sin * old_cos - new_cos * old_sin).contiguous()
                    combined_cache[cache_key] = (combined_cos, combined_sin)

                combined_cos, combined_sin = combined_cache[cache_key]

                self._launch_triton_rerotate(
                    kv_layer,
                    cur_layer_name,
                    gather_idx,
                    combined_cos,
                    combined_sin,
                    rotary_dim,
                    rotary.is_neox_style,
                )

    def _launch_triton_rerotate(
        self,
        kv_layer: torch.Tensor,
        layer_name: str,
        gather_idx: torch.Tensor,
        combined_cos: torch.Tensor,
        combined_sin: torch.Tensor,
        rotary_dim: int,
        is_neox_style: bool,
    ) -> None:
        total_len = gather_idx.shape[0]
        if total_len == 0:
            return

        page_first = ProcessLocalConfig.cache_topo.layers[layer_name].page_first
        if kv_layer.ndim == 5:
            _, _, page_size, num_head, head_dim = kv_layer.shape
            if page_first:
                kv_base = kv_layer
                stride_kv_b = kv_layer.stride(0)
                stride_kv_s = kv_layer.stride(2)
                stride_kv_h = kv_layer.stride(3)
            else:
                kv_base = kv_layer[0]
                stride_kv_b = kv_base.stride(0)
                stride_kv_s = kv_base.stride(1)
                stride_kv_h = kv_base.stride(2)
        elif kv_layer.ndim == 4:
            constraint = None
            if not page_first:
                constraint = "page_first must be true"
            elif kv_layer.shape[3] <= 0 or kv_layer.shape[3] % 2 != 0:
                constraint = "packed content dimension must be positive and even"
            elif kv_layer.stride(3) != 1:
                constraint = "packed content feature stride must equal 1"

            _, num_head, page_size, packed_dim = kv_layer.shape
            head_dim = packed_dim // 2
            if constraint is None and (
                rotary_dim <= 0
                or rotary_dim % 2 != 0
                or rotary_dim > head_dim
            ):
                constraint = (
                    "rotary_dim must be positive, even, and no greater than "
                    "the packed K head dimension"
                )
            if constraint is not None:
                raise ValueError(
                    "Invalid 4D KV cache for rerotate: "
                    f"layer={layer_name}, shape={tuple(kv_layer.shape)}, "
                    f"stride={tuple(kv_layer.stride())}, dtype={kv_layer.dtype}, "
                    f"rotary_dim={rotary_dim}, constraint={constraint}"
                )

            kv_base = kv_layer
            stride_kv_b = kv_layer.stride(0)
            stride_kv_s = kv_layer.stride(2)
            stride_kv_h = kv_layer.stride(1)
        elif kv_layer.ndim == 3:
            # 3D (num_blocks, page_size, hidden_dim)
            _, page_size, hidden_dim = kv_layer.shape
            num_head = 1
            head_dim = hidden_dim
            kv_base = kv_layer
            stride_kv_b = kv_layer.stride(0)
            stride_kv_s = kv_layer.stride(1)
            stride_kv_h = 0
        else:
            raise ValueError(
                f"Unsupported KV cache rank for rerotate: {kv_layer.ndim}"
            )

        if kv_base.dtype == torch.uint8:
            kv_to_pass = kv_base.view(torch.float8_e4m3fn)
        else:
            kv_to_pass = kv_base

        combined_cos = combined_cos.to(kv_to_pass.device)
        combined_sin = combined_sin.to(kv_to_pass.device)

        half_rot = rotary_dim // 2
        block_rot = triton.next_power_of_2(half_rot)
        grid = (total_len, num_head)

        _fused_rerotate_kernel[grid](
            kv_to_pass,
            gather_idx,
            combined_cos,
            combined_sin,
            total_len,
            stride_kv_b,
            stride_kv_s,
            stride_kv_h,
            PAGE_SIZE=page_size,
            HEAD_DIM=head_dim,
            HALF_ROT=half_rot,
            BLOCK_ROT=block_rot,
            IS_NEOX=is_neox_style,
        )

    def get_recompute_length(self, group_ptr: List[int]) -> List[int]:
        if self.recompute_policy is None:
            return []
        recompute_ptrs = self.recompute_policy.get_recompute_ptrs(group_ptr)
        return recompute_ptrs
