# Qwen3 Tokenizer wrapper for LLMzip
# Uses HuggingFace AutoTokenizer

from transformers import AutoTokenizer
from logging import getLogger
from typing import List
import os

logger = getLogger()


class Qwen3Tokenizer:
    def __init__(self, model_path: str):
        """
        Initialize Qwen3 tokenizer from HuggingFace checkpoint.
        
        Args:
            model_path: Path to the Qwen3 model directory
        """
        assert os.path.isdir(model_path), f"Model path {model_path} does not exist"
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path,
            trust_remote_code=True
        )
        
        logger.info(f"Loaded Qwen3 tokenizer from {model_path}")
        
        # Set token IDs - Qwen3 specific
        self.n_words: int = self.tokenizer.vocab_size
        
        # Qwen3 uses specific special tokens
        # BOS token
        if hasattr(self.tokenizer, 'bos_token_id') and self.tokenizer.bos_token_id is not None:
            self.bos_id: int = self.tokenizer.bos_token_id
        else:
            self.bos_id: int = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')
        
        # EOS token
        if hasattr(self.tokenizer, 'eos_token_id') and self.tokenizer.eos_token_id is not None:
            self.eos_id: int = self.tokenizer.eos_token_id
        else:
            self.eos_id: int = self.tokenizer.convert_tokens_to_ids('<|endoftext|>')
        
        # PAD token
        if hasattr(self.tokenizer, 'pad_token_id') and self.tokenizer.pad_token_id is not None:
            self.pad_id: int = self.tokenizer.pad_token_id
        else:
            # Use EOS as pad if not defined
            self.pad_id: int = self.eos_id
        
        logger.info(
            f"#words: {self.n_words} - BOS ID: {self.bos_id} - EOS ID: {self.eos_id} - PAD ID: {self.pad_id}"
        )

    def encode(self, s: str, bos: bool = False, eos: bool = False) -> List[int]:
        """
        Encode string to token IDs.
        
        Args:
            s: Input string
            bos: Whether to add BOS token at start
            eos: Whether to add EOS token at end
            
        Returns:
            List of token IDs
        """
        assert type(s) is str
        
        # Use HuggingFace tokenizer's encode method
        # add_special_tokens=False to manually control BOS/EOS
        t = self.tokenizer.encode(s, add_special_tokens=False)
        
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        
        return t

    def decode(self, t: List[int]) -> str:
        """
        Decode token IDs to string.
        
        Args:
            t: List of token IDs
            
        Returns:
            Decoded string
        """
        return self.tokenizer.decode(t, skip_special_tokens=False)
