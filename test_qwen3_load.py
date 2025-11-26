#!/usr/bin/env python3
"""
Test script to verify Qwen3-0.6B model loads correctly with HuggingFace Transformers.
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def test_qwen3_loading():
    """Test that Qwen3-0.6B model loads successfully."""

    model_path = "/home/user/Qwen3-0.6B"

    print("Testing Qwen3-0.6B model loading...")
    print(f"Model path: {model_path}")
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Load tokenizer
    print("\n1. Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        print(f"   ✓ Tokenizer loaded successfully")
        print(f"   Vocab size: {tokenizer.vocab_size}")
    except Exception as e:
        print(f"   ✗ Failed to load tokenizer: {e}")
        return False

    # Load model
    print("\n2. Loading model...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            low_cpu_mem_usage=True
        )
        print(f"   ✓ Model loaded successfully")
        print(f"   Device: {next(model.parameters()).device}")
        print(f"   Dtype: {next(model.parameters()).dtype}")

        # Count parameters
        total_params = sum(p.numel() for p in model.parameters())
        print(f"   Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    except Exception as e:
        print(f"   ✗ Failed to load model: {e}")
        return False

    # Test inference
    print("\n3. Testing inference...")
    try:
        # Simple test without chat template for compression use case
        test_text = "The quick brown fox"
        inputs = tokenizer(test_text, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

        print(f"   ✓ Inference successful")
        print(f"   Input shape: {inputs['input_ids'].shape}")
        print(f"   Output logits shape: {logits.shape}")

        # Get next token probabilities (useful for compression)
        next_token_logits = logits[0, -1, :]
        probs = torch.softmax(next_token_logits, dim=-1)
        top5_probs, top5_indices = torch.topk(probs, 5)

        print(f"\n   Top 5 next token predictions:")
        for prob, idx in zip(top5_probs, top5_indices):
            token = tokenizer.decode([idx])
            print(f"     {token!r}: {prob.item():.4f}")

    except Exception as e:
        print(f"   ✗ Inference failed: {e}")
        return False

    print("\n✓ All tests passed! Qwen3-0.6B is ready for use.")
    return True

if __name__ == "__main__":
    success = test_qwen3_loading()
    exit(0 if success else 1)
