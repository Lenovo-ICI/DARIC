from .searcher import PrefixSearcher, IndependentSearcher
from .planner import HybridPlanner, LookUpResult, ChunkStatus
from .lookup_client import LookupClient


def set_searcher_planner(
        reuse_mode: str,
        config,
        max_chunk_len: int,
        strict_pic: bool = False
    ):
    if reuse_mode == "prefix":
        searcher = PrefixSearcher(
            config=config,
            max_chunk_len=max_chunk_len,
        )
    elif reuse_mode == "independent":
        searcher = IndependentSearcher(
            config=config,
            max_chunk_len=max_chunk_len,
        )
    else:
        raise ValueError(f"Unknown reuse mode: {reuse_mode}")
    planner = HybridPlanner(strict_pic=strict_pic)
    return searcher, planner