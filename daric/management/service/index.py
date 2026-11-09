import time
import threading
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, DefaultDict, Dict, List, Optional, Tuple, Union

@dataclass
class IndexMeta:
	size: int
	ts: float
	freq: float
	offset: Optional[Union[int, str]] = None
	valid: bool = False


class KVCacheIndex:
	def __init__(
		self,
		max_size: int,
		protected_window: float = 6.0,
	):
		self.ts_divider = 2e11
		self.protected_window = protected_window

		self.space = max_size
		self.meta: Dict[str, IndexMeta] = {}
		self._lock = threading.RLock()

		self._used_bytes = 0

	def close(self):
		pass

	def _score(self, meta: IndexMeta) -> float:
		score = meta.freq + (meta.ts / self.ts_divider)
		return score

	def _split_cache_key(self, key: str) -> Tuple[str, Optional[int], Optional[int]]:
		parts = key.rsplit("_", 2)
		if len(parts) != 3:
			return key, None, None

		cache_key, tp_rank_str, layer_str = parts
		if not tp_rank_str.isdigit() or not layer_str.isdigit():
			return key, None, None

		return cache_key, int(tp_rank_str), int(layer_str)

	def _group_meta_keys(self) -> DefaultDict[str, List[str]]:
		grouped_keys: DefaultDict[str, List[str]] = defaultdict(list)
		for key in list(self.meta.keys()):
			cache_key, _, _ = self._split_cache_key(key)
			grouped_keys[cache_key].append(key)
		return grouped_keys

	def get_storage_space(self) -> int:
			return self.space

	def contains(self,
			keys: List[Union[str, int]]
		) -> List[bool]:
		with self._lock:
			return [str(key) in self.meta for key in keys]

	def lookup(self, 
			keys: List[Union[str, int]]
		) -> List[Any]:
		cur_ts = time.time()
		with self._lock:
			results = []
			for key in keys:
				meta = self.meta.get(str(key), None)
				if meta is not None:
					if meta.valid:
						results.append(meta.size)
						meta.freq += 1.0
						meta.ts = cur_ts
						continue
				results.append(None)
			return results

	def finish_insert(self,
			keys: List[Union[str, int]],
		):
		with self._lock:
			for key in keys:
				meta = self.meta.get(str(key), None)
				if meta is not None:
					meta.valid = True
				else:
					raise KeyError(f"Key {key} not found in index for validation")

	def get(
		self,
		keys: List[Union[str, int]],
	) -> List[Optional[IndexMeta]]:
		with self._lock:
			results = []
			for key in keys:
				key_str = str(key)
				meta = self.meta.get(key_str, None)
				if meta is not None:
					results.append(meta)
				else:
					results.append(None)
			return results

	def touch(
		self,
		keys: List[Union[str, int]],
	) -> List[Optional[IndexMeta]]:
		cur_ts = time.time()
		with self._lock:
			results = []
			for key in keys:
				key_str = str(key)
				meta = self.meta.get(key_str, None)
				if meta is not None and meta.valid:
					meta.freq += 1.0
					meta.ts = cur_ts
					results.append(meta)
					continue
				results.append(None)
			return results

	def submit_insert(
		self, 
		keys: List[Union[str, int]],
		sizes: List[int],
		offsets: List[int],
	):
		if not (len(keys) == len(sizes) == len(offsets)):
			raise ValueError("All input lists must have the same length")
		if not keys:
			return
		current_ts = time.time()
		with self._lock:
			for key, size, offset in zip(keys, sizes, offsets):
				key_str = str(key)
				existing_meta = self.meta.get(key_str)
				if existing_meta is not None:
					self._used_bytes -= existing_meta.size
				self.meta[key_str] = IndexMeta(
					size=size,
					ts=current_ts,
					freq=1.0,
					offset=offset,
				)
				self._used_bytes += size

	def delete(self, keys: List[Union[str, int]]):
		if not keys:
			return
		with self._lock:
			for key in keys:
				key_str = str(key)
				meta = self.meta.pop(key_str, None)
				if meta is None:
					continue
				self._used_bytes -= meta.size
				if self._used_bytes < 0:
					raise ValueError("Used bytes cannot be negative after deletion")

	def get_all_meta(self) -> List[IndexMeta]:
		with self._lock:
			return list(self.meta.values())

	def get_used_space(self) -> int:
		with self._lock:
			return self._used_bytes

	def get_free_space(self) -> int:
		return self.space - self.get_used_space()

	def get_eviction_candidates(self, bytes_to_free: Optional[int] = None) -> List[str]:
		if bytes_to_free is not None and bytes_to_free <= 0:
			return []

		with self._lock:
			current_ts = time.time()
			grouped_keys = self._group_meta_keys()
			sorted_groups = sorted(
				grouped_keys.items(),
				key=lambda item: (
					max(self._score(self.meta[sub_key]) for sub_key in item[1]),
					item[0],
				),
			)

			def is_group_protected(group_keys: List[str]) -> bool:
				for candidate_key in group_keys:
					meta = self.meta.get(candidate_key)
					if meta is None:
						continue
					if not meta.valid:
						continue
					if current_ts - meta.ts < self.protected_window:
						return True
				return False

			scored: List[Tuple[str, float]] = []
			for _, group_keys in sorted_groups:
				if is_group_protected(group_keys):
					continue
				for key in group_keys:
					meta = self.meta.get(key)
					if meta is None:
						continue
					scored.append((key, self._score(meta)))
			scored.sort(key=lambda x: x[1])
			return [k for k, _ in scored]

	def evict_cold(self,
			keys: List[Union[str, int]],
			bytes_to_free: Optional[int] = None,
		) -> Tuple[List["IndexMeta"], int]:
		with self._lock:
			current_ts = time.time()
			freed = 0
			evicted: List[IndexMeta] = []
			for key in keys:
				key_str = str(key)
				meta = self.meta.get(key_str)
				if meta is None or not meta.valid:
					continue
				if current_ts - meta.ts < self.protected_window:
					continue
				del self.meta[key_str]
				self._used_bytes -= meta.size
				evicted.append(meta)
				freed += meta.size
				if bytes_to_free is not None and freed >= bytes_to_free:
					break
			if self._used_bytes < 0:
				raise ValueError("Used bytes cannot be negative after deletion")
			return evicted, freed

	def clean(self):
		with self._lock:
			self.meta.clear()
			self._used_bytes = 0


class KVCacheIndexManager:
    def __init__(self):
        self.indexes: Dict[str, KVCacheIndex] = {}

    def create_index(self, name: str, size: int, protected_window: float = 0.0) -> KVCacheIndex:
        self.indexes[name] = KVCacheIndex(size, protected_window=protected_window)
        return self.get_index(name)

    def replace_index(self, name: str, size: int, protected_window: float = 0.0) -> KVCacheIndex:
        return self.create_index(name, size, protected_window=protected_window)

    def get_index(self, name: str) -> KVCacheIndex:
        index = self.indexes.get(name)
        if index is None:
            raise KeyError(f"Index '{name}' is not available.")
        return index

    def get_storage_space(self, name: str):
        return self.get_index(name).get_storage_space()

    def contains(self, name: str, keys: List[Union[str, int]]):
        return self.get_index(name).contains(keys)

    def lookup(self, name: str, keys: List[Union[str, int]]):
        return self.get_index(name).lookup(keys)

    def finish_insert(self, name: str, keys: List[Union[str, int]]):
        return self.get_index(name).finish_insert(keys)

    def get(self, name: str, keys: List[Union[str, int]]):
        return self.get_index(name).get(keys)

    def touch(self, name: str, keys: List[Union[str, int]]):
        return self.get_index(name).touch(keys)

    def submit_insert(self, name: str, keys: List[Union[str, int]], sizes: List[int], offsets: List[int]):
        return self.get_index(name).submit_insert(keys, sizes, offsets)

    def delete(self, name: str, keys: List[Union[str, int]]):
        return self.get_index(name).delete(keys)

    def get_all_meta(self, name: str):
        return self.get_index(name).get_all_meta()

    def get_used_space(self, name: str):
        return self.get_index(name).get_used_space()

    def get_free_space(self, name: str):
        return self.get_index(name).get_free_space()

    def get_eviction_candidates(self, name: str, bytes_to_free: Optional[int] = None):
        return self.get_index(name).get_eviction_candidates(bytes_to_free=bytes_to_free)

    def evict_cold(self, name: str, keys: List[Union[str, int]], bytes_to_free: Optional[int] = None):
        return self.get_index(name).evict_cold(keys, bytes_to_free=bytes_to_free)

    def clean(self, name: str):
        return self.get_index(name).clean()