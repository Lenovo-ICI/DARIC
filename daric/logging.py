import os
import sys
import logging
import threading
from typing import Optional

log_levels = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}
_DATE_FORMAT = "%m-%d %H:%M:%S"

_lock = threading.Lock()
_default_handler: Optional[logging.Handler] = None


def _get_default_logging_level() -> int:
    env_level_str = os.getenv("DARIC_LOG_LEVEL", None)
    if env_level_str:
        if env_level_str in log_levels:
            return log_levels[env_level_str]
        else:
            logging.getLogger().warning(
                f"Unknown option DARIC_LOG_LEVEL={env_level_str}, "
                f"has to be one of: {', '.join(log_levels.keys())}"
            )
    return logging.INFO


def _get_library_name() -> str:
    return __name__.split(".")[0]


def _get_library_root_logger() -> logging.Logger:
    return logging.getLogger(_get_library_name())


def _configure_library_root_logger() -> None:
    global _default_handler

    with _lock:
        if _default_handler:
            return
        _default_handler = logging.StreamHandler()
        if sys.stderr is None:
            sys.stderr = open(os.devnull, "w")

        _default_handler.flush = sys.stderr.flush

        library_root_logger = _get_library_root_logger()
        library_root_logger.addHandler(_default_handler)
        library_root_logger.setLevel(_get_default_logging_level())
        formatter = logging.Formatter(
            "[DARIC_%(levelname)s %(asctime)s|%(filename)s:%(lineno)d]: %(message)s",
            datefmt=_DATE_FORMAT,
        )
        _default_handler.setFormatter(formatter)

        is_ci = os.getenv("CI") is not None and os.getenv("CI").upper() in {
            "1",
            "ON",
            "YES",
            "TRUE",
        }
        library_root_logger.propagate = True if is_ci else False


def get_logger(name: Optional[str] = None) -> logging.Logger:
    if name is None:
        name = _get_library_name()

    _configure_library_root_logger()
    return logging.getLogger(name)
