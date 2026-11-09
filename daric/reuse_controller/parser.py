import os
from typing import List, TYPE_CHECKING
from dataclasses import dataclass

from ..logging import get_logger

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

logger = get_logger(__name__)

@dataclass
class ParseResult:
    prompt_token_ids: List[int]
    indptr: List[int]


class SepParser:
    def __init__(
        self, 
        tokenizer: "PreTrainedTokenizer",
        block_size: int = 16
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        if os.getenv("USE_DARIC_SEP", "1") == "0":
            special_str = tokenizer.all_special_tokens[0]
            sep = f"{special_str}{special_str}{special_str}"
        else:
            sep = os.getenv("DARIC_SEP_TOKEN", "<DARIC_SEP>")
        self.sep_ids = self.tokenizer.encode(sep, add_special_tokens=False)
        if self.tokenizer.pad_token_id is None:
            self.pad_token_id = self.tokenizer.eos_token_id
        else:
            self.pad_token_id = self.tokenizer.pad_token_id
    
    def parse(self, prompt_token_ids: List[int]) -> ParseResult:
        n = len(prompt_token_ids)
        sep_len = len(self.sep_ids)
        
        if not prompt_token_ids:
            prompt_splits = []
        elif sep_len == 0:
            prompt_splits = [prompt_token_ids.copy()]
        else:
            prompt_splits = []
            start = 0
            idx = 0
            first = self.sep_ids[0]
            while idx <= n - sep_len:
                try:
                    idx = prompt_token_ids.index(first, idx)
                except ValueError:
                    break
                if prompt_token_ids[idx:idx+sep_len] == self.sep_ids:
                    if idx > start:
                        prompt_splits.append(prompt_token_ids[start:idx])
                    start = idx + sep_len
                    idx = start
                else:
                    idx += 1
            if start < n:
                prompt_splits.append(prompt_token_ids[start:])

        new_token_ids = []
        indptr = [0]
        for idx, split in enumerate(prompt_splits):
            is_last_split = (idx == len(prompt_splits) - 1)
            remainder = len(split) % self.block_size
            if not is_last_split and remainder != 0:
                pad_len = self.block_size - remainder
                split = split + [self.pad_token_id] * pad_len

            new_token_ids.extend(split)
            indptr.append(len(new_token_ids))

        return ParseResult(
            prompt_token_ids=new_token_ids, indptr=indptr
        )


class ChunkParser:
    def __init__(
        self, 
        tokenizer: "PreTrainedTokenizer",
        block_size: int = 16
    ):
        self.tokenizer = tokenizer
        self.block_size = block_size
        if os.getenv("USE_DARIC_SEP", "1") == "0":
            special_str = tokenizer.all_special_tokens[0]
            sep = f"{special_str}{special_str}{special_str}"
        else:
            sep = os.getenv("DARIC_SEP_TOKEN", "<DARIC_SEP>")
        self.sep_ids = self.tokenizer.encode(sep, add_special_tokens=False)
    
    def parse(self, prompt_token_ids: List[int]) -> ParseResult:
        n = len(prompt_token_ids)
        sep_len = len(self.sep_ids)
        
        if not prompt_token_ids:
            prompt_splits = []
        elif sep_len == 0:
            prompt_splits = [prompt_token_ids.copy()]
        else:
            prompt_splits = []
            start = 0
            idx = 0
            first = self.sep_ids[0]
            while idx <= n - sep_len:
                try:
                    idx = prompt_token_ids.index(first, idx)
                except ValueError:
                    break
                if prompt_token_ids[idx:idx+sep_len] == self.sep_ids:
                    if idx > start:
                        prompt_splits.append(prompt_token_ids[start:idx])
                    start = idx + sep_len
                    idx = start
                else:
                    idx += 1
            if start < n:
                prompt_splits.append(prompt_token_ids[start:])
        
        new_token_ids = []
        for idx, split in enumerate(prompt_splits):
            new_token_ids.extend(split)
        indptr = [0, len(new_token_ids)]
        return ParseResult(
            prompt_token_ids=new_token_ids, indptr=indptr
        )