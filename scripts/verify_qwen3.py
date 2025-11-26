#!/usr/bin/env python3
"""
Verification script for Qwen3-0.6B model.

This script:
1. Loads the Qwen3-0.6B model from models/Qwen3-0.6B/
2. Loads the tokenizer from the same directory
3. Runs a simple inference test
4. Prints model information and success message
"""

import sys
import os
from pathlib import Path

def main():
    print("=== Qwen3-0.6B Model Verification ===\n")
    
    # Check if transformers is available
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as e:
        print(f"❌ Error: Required package not found: {e}")
        print("\nPlease install dependencies with: uv sync")
        return 1
    
    # Model directory
    model_dir = Path("models/Qwen3-0.6B")
    
    # Check if model directory exists
    if not model_dir.exists():
        print(f"❌ Error: Model directory not found: {model_dir}")
        print("\nPlease run: ./scripts/setup_qwen3.sh")
        return 1
    
    # Check if model file exists
    model_file = model_dir / "model.safetensors"
    if not model_file.exists():
        print(f"❌ Error: Model file not found: {model_file}")
        print("\nPlease run: ./scripts/setup_qwen3.sh")
        return 1
    
    print(f"✓ Model directory found: {model_dir}")
    print(f"✓ Model file found: {model_file}")
    print(f"  Size: {model_file.stat().st_size / (1024**3):.2f} GB\n")
    
    # Load tokenizer
    print("Loading tokenizer...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            str(model_dir),
            trust_remote_code=True
        )
        print(f"✓ Tokenizer loaded successfully")
        print(f"  Vocab size: {tokenizer.vocab_size:,}")
    except Exception as e:
        print(f"❌ Error loading tokenizer: {e}")
        return 1
    
    # Load model
    print("\nLoading model (this may take a minute)...")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(model_dir),
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print(f"✓ Model loaded successfully")
        
        # Get model info
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,} ({num_params/1e9:.2f}B)")
        print(f"  Architecture: {model.config.architectures[0]}")
        print(f"  Hidden size: {model.config.hidden_size}")
        print(f"  Number of layers: {model.config.num_hidden_layers}")
        print(f"  Number of attention heads: {model.config.num_attention_heads}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    # Run simple inference test
    print("\nRunning inference test...")
    try:
        test_text = "The quick brown fox"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        # Move inputs to same device as model
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Generate next token probabilities
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            
        # Get top 5 next token predictions
        next_token_logits = logits[0, -1, :]
        probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=5)
        
        print(f"✓ Inference test successful")
        print(f"\nInput text: '{test_text}'")
        print(f"Top 5 next token predictions:")
        for i, (prob, idx) in enumerate(zip(top_probs, top_indices), 1):
            token = tokenizer.decode([idx])
            print(f"  {i}. '{token}' (probability: {prob.item():.4f})")
        
    except Exception as e:
        print(f"❌ Error during inference: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    print("\n" + "="*50)
    print("✅ All checks passed! Qwen3-0.6B is ready to use.")
    print("="*50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
