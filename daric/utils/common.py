import msgpack
import xxhash
import sys
from array import array
from typing import Optional, List, Any, Union, Callable


def token_ids_to_bytes(token_ids: List[int]) -> bytes:
    token_array = array("I", token_ids)
    if token_array.itemsize != 4:
        raise RuntimeError(f"Unexpected unsigned int size: {token_array.itemsize}")
    if sys.byteorder != "little":
        token_array.byteswap()
    return token_array.tobytes()

def get_hash_id(
    hash_input: bytes, prefix: Optional[bytes] = None, type: str = "str"
) -> Union[str, int]:
    hasher = xxhash.xxh64()
    if prefix is not None:
        hasher.update(prefix)
    hasher.update(hash_input)
    if type == "int":
        return hasher.intdigest()
    else:
        return hasher.hexdigest()

def check_key_valid(object: dict, required_keys: List[str]) -> None:
    for key in required_keys:
        if key not in object:
            raise KeyError(f"Missing required key: '{key}' in object: {object}")

def serialize(obj: Any) -> bytes:
    return msgpack.packb(obj, use_bin_type=True)

def deserialize(data: bytes) -> Any:
    obj = msgpack.unpackb(data, raw=False)
    return obj