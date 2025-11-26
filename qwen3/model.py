# Qwen3 Model wrapper for LLMzip
# Uses HuggingFace AutoModelForCausalLM

import torch
from transformers import AutoModelForCausalLM
from logging import getLogger
import os

logger = getLogger()


class Qwen3Model:
    def __init__(self, model_path: str, max_batch_size: int = 1):
        """
        Initialize Qwen3 model from HuggingFace checkpoint.
        
        Args:
            model_path: Path to the Qwen3 model directory
            max_batch_size: Maximum batch size (default: 1 for compression)
        """
        assert os.path.isdir(model_path), f"Model path {model_path} does not exist"
        
        self.model_path = model_path
        self.max_batch_size = max_batch_size
        
        # Determine device: prefer CUDA, then MPS, otherwise CPU
        if torch.cuda.is_available():
            self.device = torch.device('cuda')
        elif torch.backends.mps.is_available():
            self.device = torch.device('mps')
        else:
            self.device = torch.device('cpu')
        
        # Determine dtype: CUDA uses bfloat16 if supported else float16
        # MPS and CPU use float32 for compatibility
        if self.device.type == "cuda":
            if torch.cuda.is_bf16_supported():
                self.dtype = torch.bfloat16
                logger.info("Selected dtype: bfloat16 (CUDA with bfloat16 support)")
            else:
                self.dtype = torch.float16
                logger.info("Selected dtype: float16 (CUDA without bfloat16 support)")
        else:
            self.dtype = torch.float32
            logger.info(f"Selected dtype: float32 (for {self.device.type} compatibility)")
        
        logger.info(f"Loading Qwen3 model from {model_path}...")
        logger.info(f"Device: {self.device.type}")
        
        # Load model: use device_map='auto' only on CUDA
        if self.device.type == "cuda":
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                device_map="auto",
                trust_remote_code=True
            )
        else:
            # For MPS and CPU, load without device_map and move explicitly
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=self.dtype,
                trust_remote_code=True
            )
            self.model = self.model.to(self.device)
        
        # Set to eval mode
        self.model.eval()
        
        # Get model config
        config = self.model.config
        self.vocab_size = config.vocab_size
        self.max_position_embeddings = config.max_position_embeddings
        
        logger.info(f"Model loaded successfully")
        logger.info(f"Vocab size: {self.vocab_size}")
        logger.info(f"Max position embeddings: {self.max_position_embeddings}")
        
        # Initialize KV cache for efficient inference
        # This will be managed internally by HuggingFace transformers
        self.past_key_values = None
    
    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int):
        """
        Forward pass through the model to get next token logits.
        
        Args:
            tokens: Input token IDs, shape (batch_size, seq_len)
            start_pos: Starting position in the sequence (for KV cache)
            
        Returns:
            Logits for next token prediction, shape (batch_size, vocab_size)
        """
        # Move tokens to model device
        if tokens.device != next(self.model.parameters()).device:
            tokens = tokens.to(next(self.model.parameters()).device)
        
        # For HuggingFace models, we pass the full sequence each time
        # The model internally handles the KV cache through past_key_values
        # However, for LLMzip we use a sliding window approach
        
        # Get model outputs
        # We don't use past_key_values for simplicity in the sliding window case
        # Each forward pass is independent with its own context
        outputs = self.model(tokens, use_cache=False)
        
        # Get logits for the last token in the sequence
        logits = outputs.logits[:, -1, :]
        
        return logits.float()
    
    def reset_cache(self):
        """Reset the KV cache (for compatibility with LLaMA interface)"""
        self.past_key_values = None
