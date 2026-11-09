from dataclasses import dataclass
from ..utils import check_key_valid


@dataclass
class LocalStoreConfig:
    layer_wise: bool = False
    host: str = "127.0.0.1"
    port: int = 50097


def get_backend_config(backend_info: dict) -> LocalStoreConfig:
    check_key_valid(backend_info, ["port"])
    storage_config = LocalStoreConfig(
        port=backend_info.get("port", 50097),
        host=backend_info.get("host", "127.0.0.1"),
        layer_wise=backend_info.get("layer_wise", False),
    )

    return storage_config
    