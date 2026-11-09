import ctypes
import os
import threading

from ..logging import get_logger

logger = get_logger(__name__)

HOST_MEMORY_ALREADY_REGISTERED = 712
HOST_MEMORY_NOT_REGISTERED = 708


_RUNTIME_CANDIDATES = (
    ("libcudart.so", "cuda"),
)

_HOST_REGISTER_PORTABLE = 1


class GpuHostRuntime:
    def __init__(self, lib: ctypes.CDLL, prefix: str, path: str):
        register = getattr(lib, f"{prefix}HostRegister")
        register.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        register.restype = ctypes.c_int
        unregister = getattr(lib, f"{prefix}HostUnregister")
        unregister.argtypes = [ctypes.c_void_p]
        unregister.restype = ctypes.c_int
        self._register_fn = register
        self._unregister_fn = unregister
        self.prefix = prefix
        self.path = path

    def register(self, address: int, size: int) -> int:
        return self._register_fn(
            ctypes.c_void_p(address),
            ctypes.c_size_t(size),
            ctypes.c_uint(_HOST_REGISTER_PORTABLE),
        )

    def unregister(self, address: int) -> int:
        return self._unregister_fn(ctypes.c_void_p(address))


_runtime = None
_runtime_lock = threading.Lock()


def _find_in_process_maps(soname: str):
    try:
        with open("/proc/self/maps", "r") as f:
            for line in f:
                if soname in line:
                    path = line.strip().split()[-1]
                    if os.path.exists(path):
                        return path
    except OSError:
        pass
    return None


def _runtime_base_dirs():
    dirs = []
    for env in ("CUDA_HOME", "CUDA_PATH", "ROCM_HOME", "ROCM_PATH", "DTKROOT"):
        d = os.environ.get(env)
        if d:
            dirs.append(d)
    try:
        from torch.utils import cpp_extension
        for attr in ("CUDA_HOME", "ROCM_HOME"):
            d = getattr(cpp_extension, attr, None)
            if d:
                dirs.append(d)
    except Exception:
        pass
    return dirs


def _try_load(soname: str, prefix: str):
    paths = []
    mapped = _find_in_process_maps(soname)
    if mapped:
        paths.append(mapped)
    paths.append(soname)  # rely on the dynamic linker search path
    for base in _runtime_base_dirs():
        paths.append(os.path.join(base, "lib64", soname))
        paths.append(os.path.join(base, "lib", soname))
    for path in paths:
        try:
            lib = ctypes.CDLL(path)
        except OSError:
            continue
        try:
            return GpuHostRuntime(lib, prefix, path)
        except AttributeError:
            continue
    return None


def get_gpu_host_runtime() -> GpuHostRuntime:
    global _runtime
    with _runtime_lock:
        if _runtime is None:
            for soname, prefix in _RUNTIME_CANDIDATES:
                _runtime = _try_load(soname, prefix)
                if _runtime is not None:
                    logger.info("Using %s for host memory registration.", _runtime.path)
                    break
        if _runtime is None:
            raise RuntimeError(
                "No GPU runtime found (looked for libcudart.so); "
                "cannot register host memory for GPU access."
            )
        return _runtime
