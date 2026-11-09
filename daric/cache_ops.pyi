from collections.abc import Sequence

import torch


class CacheCopier:
    def __init__(self) -> None: ...

    def register_layer(
        self, name: str, tensor: torch.Tensor, is_page_first: bool
    ) -> None: ...

    def register_store(self, buffer: torch.Tensor) -> None: ...

    def gather_pages(
        self,
        layer_names: Sequence[str],
        offsets: Sequence[int],
        byte_sizes: Sequence[int],
        indices_flat: torch.Tensor,
        indices_offsets: Sequence[int],
        stream: int = 0,
    ) -> None: ...

    def scatter_pages(
        self,
        layer_names: Sequence[str],
        offsets: Sequence[int],
        byte_sizes: Sequence[int],
        indices_flat: torch.Tensor,
        indices_offsets: Sequence[int],
        stream: int = 0,
    ) -> None: ...


def batched_memcpy_d2h(
    staging: torch.Tensor,
    shm: torch.Tensor,
    stg_offsets: Sequence[int],
    shm_offsets: Sequence[int],
    sizes: Sequence[int],
    stream: int = 0,
) -> None: ...


def batched_memcpy_h2d(
    shm: torch.Tensor,
    staging: torch.Tensor,
    shm_offsets: Sequence[int],
    stg_offsets: Sequence[int],
    sizes: Sequence[int],
    stream: int = 0,
) -> None: ...
