"""Register DARIC's vLLM integration without modifying vLLM's files."""

import importlib
import importlib.util
import logging
import os
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_ENV_FILE = Path(__file__).with_name("plugin.env")


def _load_env_defaults(env_file: Path = _DEFAULT_ENV_FILE) -> None:
    for line_number, raw_line in enumerate(
        env_file.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"{env_file}:{line_number}: expected KEY=VALUE")
        key, value = line.split("=", 1)
        key = key.strip()
        if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key) is None:
            raise ValueError(f"{env_file}:{line_number}: invalid environment key {key!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


_TARGETS = (
    ("vllm.v1.worker.gpu_worker", "Worker", "worker"),
    ("vllm.v1.worker.gpu_model_runner", "GPUModelRunner", "runner"),
    ("vllm.v1.worker.gpu.model_runner", "GPUModelRunner", "runner"),
    ("vllm.v1.engine.llm_engine", "LLMEngine", "engine"),
    ("vllm.v1.engine.async_llm", "AsyncLLM", "engine"),
    ("vllm.v1.engine.input_processor", "InputProcessor", "processor"),
    ("vllm.v1.engine.processor", "Processor", "processor"),
)

_CONNECTORS = (
    ("DARICConnector", "daric.integration.vllm.daric_connector"),
)


def _find_target(module_name):
    try:
        return importlib.util.find_spec(module_name)
    except ModuleNotFoundError as exc:
        if exc.name and (module_name == exc.name or module_name.startswith(exc.name + ".")):
            return None
        raise


def _check_legacy_patch(spec):
    if spec.origin and spec.origin.endswith(".py"):
        source = Path(spec.origin).read_text(encoding="utf-8")
        if "# === INJECTED CODE ===" in source:
            raise RuntimeError(
                f"Legacy DARIC injection found in {spec.origin}. Restore the "
                "original vLLM files or reinstall vLLM before enabling the plugin."
            )


def _register_connectors(factory):
    registered = factory.__dict__.get("_daric_plugin_connectors")
    if registered is None:
        registered = set()
        factory._daric_plugin_connectors = registered
    for name, module_name in _CONNECTORS:
        if name not in registered:
            factory.register_connector(name, module_name, name)
            registered.add(name)


def register():
    """Entry point called by vLLM in each process that loads general plugins."""
    _load_env_defaults()

    from daric.integration.vllm.plugins.processor import register_tokenizer_hook
    register_tokenizer_hook()
    
    factory_name = "vllm.distributed.kv_transfer.kv_connector.factory"
    factory_spec = _find_target(factory_name)
    if factory_spec is None:
        raise RuntimeError(f"Unsupported vLLM installation: missing {factory_name}")
    _check_legacy_patch(factory_spec)
    targets = []
    for module_name, class_name, kind in _TARGETS:
        spec = _find_target(module_name)
        if spec is not None:
            _check_legacy_patch(spec)
            targets.append((module_name, class_name, kind))
        else:
            logger.debug("vLLM module %s is unavailable", module_name)

    missing = {"worker", "runner", "engine", "processor"} - {
        kind for _, _, kind in targets
    }
    if missing:
        raise RuntimeError(
            "Unsupported vLLM installation: no DARIC patch target for "
            + ", ".join(sorted(missing))
        )

    from daric.integration.vllm.plugins.model_runner import wrap_gpu_runner
    from daric.integration.vllm.plugins.processor import wrap_llm_engine, wrap_processor
    from daric.integration.vllm.plugins.worker import wrap_worker

    wrappers = {
        "worker": wrap_worker,
        "runner": wrap_gpu_runner,
        "engine": wrap_llm_engine,
        "processor": wrap_processor,
    }
    factory_module = importlib.import_module(factory_name)
    _register_connectors(factory_module.KVConnectorFactory)
    for module_name, class_name, kind in targets:
        module = importlib.import_module(module_name)
        cls = getattr(module, class_name)
        marker = f"_daric_plugin_{kind}_wrapped"
        if not cls.__dict__.get(marker, False):
            wrappers[kind](cls)
            setattr(cls, marker, True)
            logger.info("DARIC plugin wrapped %s.%s", module_name, class_name)
