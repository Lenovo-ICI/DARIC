import os
from functools import wraps
from contextlib import contextmanager


def _get_multiplier():
    return int(os.getenv("DARIC_MAX_GROUP", "1"))


def _get_max_num_seqs(config):
    original = config.max_num_seqs
    if original >= 512:
        return original
    return min(original * _get_multiplier(), 512)


@contextmanager
def _patch_scheduler_config(config):
    original = config.max_num_seqs
    config.max_num_seqs = _get_max_num_seqs(config)
    try:
        yield
    finally:
        config.max_num_seqs = original


@contextmanager
def _patch_max_num_reqs(self):
    original = self.scheduler_config.max_num_seqs
    patched = _get_max_num_seqs(self.scheduler_config)
    self.scheduler_config.max_num_seqs = patched
    self.max_num_reqs = patched
    try:
        yield
    finally:
        self.scheduler_config.max_num_seqs = original
        self.max_num_reqs = original


def wrap_runner_init(original_init):
    @wraps(original_init)
    def _wrapper(self, *args, **kwargs):
        vllm_config = args[0] if args else kwargs.get("vllm_config")
        original_max_num_seqs = vllm_config.scheduler_config.max_num_seqs
        with _patch_scheduler_config(vllm_config.scheduler_config):
            original_init(self, *args, **kwargs)
        self.max_num_reqs = original_max_num_seqs
    return _wrapper


def wrap_runner_func(original_func):
    @wraps(original_func)
    def _wrapper(self, *args, **kwargs):
        with _patch_max_num_reqs(self):
            return original_func(self, *args, **kwargs)
    return _wrapper


def wrap_gpu_runner(cls):
    cls.__init__ = wrap_runner_init(cls.__init__)
    methods_to_wrap = [
        "initialize_kv_cache",
        "prepare_inputs",
        "profile_cudagraph_memory",
        "capture_model",
        "load_model"    # In vllm 0.29.0, the initialization of sampler is done during model loading
    ]
    for method_name in methods_to_wrap:
        if hasattr(cls, method_name):
            original_method = getattr(cls, method_name)
            setattr(cls, method_name, wrap_runner_func(original_method))
            
    return cls