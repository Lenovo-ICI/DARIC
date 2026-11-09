import gc
import threading
import torch
from typing import Any

_device_uuid_to_index: dict[str, int] = {}
_device_map_lock = threading.Lock()


def _discover_devices() -> None:
    with _device_map_lock:
        if _device_uuid_to_index:
            return
        for i in range(torch.cuda.device_count()):
            try:
                uuid = str(torch.cuda.get_device_properties(i).uuid)
            except Exception:
                continue
            _device_uuid_to_index[uuid] = i


def device_uuid(device_index: int) -> str:
    return str(torch.cuda.get_device_properties(device_index).uuid)


def device_index_from_uuid(uuid: str, fallback: int = 0) -> int:
    _discover_devices()
    return _device_uuid_to_index.get(uuid, fallback)


def export_kv_tensor(tensor: torch.Tensor) -> dict:
    if not tensor.is_cuda:
        raise ValueError("export_kv_tensor requires a CUDA tensor")
    storage = tensor.untyped_storage()
    handle = storage._share_cuda_()
    return {
        "handle": handle,
        "dtype": tensor.dtype,
        "shape": tuple(tensor.shape),
        "stride": tuple(tensor.stride()),
        "storage_offset": int(tensor.storage_offset()),
        "device_uuid": device_uuid(tensor.device.index),
    }


def import_kv_tensor(wrapper: dict) -> torch.Tensor:
    device_index = device_index_from_uuid(wrapper["device_uuid"])
    torch.cuda.set_device(device_index)
    handle = wrapper["handle"]
    storage = torch.UntypedStorage._new_shared_cuda(device_index, *handle[1:])
    t = torch.empty((), device=f"cuda:{device_index}", dtype=wrapper["dtype"])
    t.set_(storage, wrapper["storage_offset"], wrapper["shape"], wrapper["stride"])
    return t


def make_interprocess_event(device_index: int) -> torch.cuda.Event:
    torch.cuda.set_device(device_index)
    return torch.cuda.Event(interprocess=True)


def export_event(event: torch.cuda.Event) -> bytes:
    return event.ipc_handle()


def import_event(handle: bytes, device_index: int) -> torch.cuda.Event:
    torch.cuda.set_device(device_index)
    return torch.cuda.Event.from_ipc_handle(
        torch.device(f"cuda:{device_index}"), handle
    )


def release_ipc_tensors(tensors: list[torch.Tensor]) -> None:
    tensors.clear()
    gc.collect()
    torch.cuda.empty_cache()
    ipc_collect = getattr(torch.cuda, "ipc_collect", None)
    if ipc_collect is not None:
        ipc_collect()
