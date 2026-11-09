import importlib
import inspect
import os
from functools import wraps


def _add_daric_sep_token(tokenizer, sep_token):
    add_special_tokens = tokenizer.add_special_tokens
    parameters = inspect.signature(add_special_tokens).parameters
    replacement_parameter = next(
        (
            name
            for name in (
                "replace_extra_special_tokens",
                "replace_additional_special_tokens",
            )
            if name in parameters
        ),
        None,
    )
    if replacement_parameter is None:
        raise RuntimeError(
            "Unsupported tokenizer.add_special_tokens signature: "
            f"{inspect.signature(add_special_tokens)}"
        )

    add_special_tokens(
        {"additional_special_tokens": [sep_token]},
        **{replacement_parameter: False},
    )


def _get_hf_tokenizer_module():
    try:
        return importlib.import_module("vllm.tokenizers.hf")
    except ModuleNotFoundError as exc:
        if exc.name not in {"vllm.tokenizers", "vllm.tokenizers.hf"}:
            raise
    return importlib.import_module("vllm.transformers_utils.tokenizer")


def apply_parse_result(parser, request):
    prompt_token_ids = getattr(request, "prompt_token_ids", None)
    sampling_params = getattr(request, "sampling_params", None)
    if prompt_token_ids is None or sampling_params is None:
        return request

    extra_args = sampling_params.extra_args or {}
    kv_transfer_params = extra_args.get("kv_transfer_params", None)
    if (
        isinstance(kv_transfer_params, dict)
        and "chunk_ptr" in kv_transfer_params
    ):
        return request

    parse_result = parser.parse(prompt_token_ids)
    request.prompt_token_ids = parse_result.prompt_token_ids
    sampling_params.extra_args = {
        **extra_args,
        "kv_transfer_params": {
            **(kv_transfer_params if isinstance(kv_transfer_params, dict) else {}),
            "chunk_ptr": parse_result.indptr,
        }
    }
    return request


def wrap_engine_process_inputs(original_process_inputs):
    @wraps(original_process_inputs)
    def _wrapped_process_inputs(self, *args, **kwargs):
        request = original_process_inputs(self, *args, **kwargs)
        if not os.environ.get("USE_DARIC_PARSE", "0") == "1":
            return request
        request_parse = request[1] if isinstance(request, tuple) else request 
        request_parse = apply_parse_result(self.parser, request_parse)
        return (request[0], request_parse) if isinstance(request, tuple) else request_parse
    return _wrapped_process_inputs


def wrap_engine_add_request(original_add_request):
    from vllm.v1.engine import EngineCoreRequest
    @wraps(original_add_request)
    def _wrapped_add_request(self, *args, **kwargs):
        if not os.environ.get("USE_DARIC_PARSE", "0") == "1":
            return original_add_request(self, *args, **kwargs)
        prompt = kwargs.get("prompt")
        if prompt is None and len(args) >= 2:
            prompt = args[1]
        if isinstance(prompt, EngineCoreRequest):
            prompt = apply_parse_result(self.parser, prompt)
        return original_add_request(self, *args, **kwargs)
    return _wrapped_add_request


def wrap_engine_init(original_init):
    from daric.reuse_controller.parser import SepParser, ChunkParser
    @wraps(original_init)
    def _wrapper(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        if not os.environ.get("USE_DARIC_PARSE", "0") == "1":
            return
        block_size = self.vllm_config.cache_config.block_size
        mamba_block_size = self.vllm_config.cache_config.mamba_block_size
        sliding_window = self.vllm_config.cache_config.sliding_window
        no_mamba_layer = mamba_block_size is None or mamba_block_size == 0
        no_sliding_window_layer = sliding_window is None
        if no_mamba_layer and no_sliding_window_layer:
            self.parser = SepParser(
                self.tokenizer,
                block_size=block_size
            )
        else:
            self.parser = ChunkParser(
                self.tokenizer,
                block_size=block_size
            )
        try:
            self.input_processor.parser = self.parser
        except AttributeError:
            self.processor.parser = self.parser
    return _wrapper


def wrap_llm_engine(cls):
    cls.__init__ = wrap_engine_init(cls.__init__)
    cls.add_request = wrap_engine_add_request(cls.add_request)
    return cls

def wrap_processor(cls):
    cls.process_inputs = wrap_engine_process_inputs(cls.process_inputs)
    return cls

def register_tokenizer_hook():
    hf_tokenizer = _get_hf_tokenizer_module()
    from transformers import AddedToken

    original = hf_tokenizer.get_cached_tokenizer
    if getattr(original, "_daric_wrapped", False):
        return

    @wraps(original)
    def get_cached_tokenizer_with_daric(tokenizer):
        if (
            os.getenv("USE_DARIC_PARSE", "0") == "1"
            and os.getenv("USE_DARIC_SEP", "1") == "1"
        ):
            sep = os.getenv("DARIC_SEP_TOKEN", "<DARIC_SEP>")
            _add_daric_sep_token(
                tokenizer,
                AddedToken(
                    sep,
                    normalized=False,
                    lstrip=False,
                    rstrip=False,
                ),
            )

        return original(tokenizer)

    get_cached_tokenizer_with_daric._daric_wrapped = True
    hf_tokenizer.get_cached_tokenizer = get_cached_tokenizer_with_daric