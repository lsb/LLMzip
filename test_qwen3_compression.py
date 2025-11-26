#!/usr/bin/env python3
"""
Test script for Qwen3 LLMzip compression/decompression.

This script creates synthetic test data, compresses it using Qwen3,
decompresses it, and verifies round-trip correctness.

Usage:
    python test_qwen3_compression.py [--model_path models/Qwen3-0.6B] [--small_only]
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


def create_synthetic_data(size_kb, output_path):
    """
    Create synthetic text data of approximately the given size in KB.
    
    Args:
        size_kb: Target size in kilobytes
        output_path: Path to save the synthetic data
    """
    # Create varied synthetic text with different patterns
    # to make compression more interesting
    patterns = [
        "The quick brown fox jumps over the lazy dog. ",
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. ",
        "In a hole in the ground there lived a hobbit. ",
        "It was the best of times, it was the worst of times. ",
        "To be or not to be, that is the question. ",
        "All animals are equal, but some animals are more equal than others. ",
        "1234567890 " * 5,  # Numbers
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ " * 2,  # Uppercase
        "abcdefghijklmnopqrstuvwxyz " * 2,  # Lowercase
    ]
    
    target_size = size_kb * 1024  # Convert to bytes
    text = ""
    
    while len(text.encode('utf-8')) < target_size:
        # Mix patterns to create varied content
        for pattern in patterns:
            text += pattern
            if len(text.encode('utf-8')) >= target_size:
                break
    
    # Truncate to exact size
    text_bytes = text.encode('utf-8')[:target_size]
    text = text_bytes.decode('utf-8', errors='ignore')
    
    with open(output_path, 'w') as f:
        f.write(text)
    
    actual_size = len(text.encode('utf-8'))
    print(f"  Created synthetic data: {actual_size:,} bytes (~{actual_size/1024:.1f} KB)")
    return actual_size


def test_compression_roundtrip(model_path, test_size_kb, win_len=255, compression_alg='ArithmeticCoding'):
    """
    Test compression and decompression round-trip for a given data size.
    
    Args:
        model_path: Path to Qwen3 model
        test_size_kb: Size of test data in KB
        win_len: Context window length
        compression_alg: Compression algorithm to use
        
    Returns:
        True if test passed, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"Testing {test_size_kb}KB data with win_len={win_len}")
    print(f"{'='*60}")
    
    # Create temporary directory for test files
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic test data
        test_file = os.path.join(tmpdir, 'test_data.txt')
        actual_size = create_synthetic_data(test_size_kb, test_file)
        
        # Read original text
        with open(test_file, 'r') as f:
            original_text = f.read()
        
        print(f"  Original text length: {len(original_text)} characters")
        
        # Import dependencies and Qwen3 modules
        import numpy as np
        from qwen3 import Qwen3Model, Qwen3Tokenizer, Qwen3_encode, Qwen3_decode
        
        # Load model and tokenizer
        print(f"  Loading model from {model_path}...")
        try:
            tokenizer = Qwen3Tokenizer(model_path=model_path)
            model = Qwen3Model(model_path=model_path, max_batch_size=1)
        except Exception as e:
            print(f"  ❌ Error loading model: {e}")
            return False
        
        # Create encoder and decoder
        encoder = Qwen3_encode(model, tokenizer)
        decoder = Qwen3_decode(model, tokenizer)
        
        # Encode text
        print(f"  Encoding with {compression_alg}...")
        compression_folder = os.path.join(tmpdir, 'compressed')
        os.makedirs(compression_folder, exist_ok=True)
        compressed_file_name = os.path.join(compression_folder, 'test')
        
        try:
            tokens_full = np.array(tokenizer.encode(original_text, bos=False, eos=False))
            print(f"  Token count: {len(tokens_full)}")
            
            encoder.encode_from_tokens(
                win_size=win_len,
                compression_alg=compression_alg,
                compressed_file_name=compressed_file_name,
                tokens_full=tokens_full,
                batched_encode=False,
                with_context_start=False
            )
        except Exception as e:
            print(f"  ❌ Error during encoding: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Check compressed file exists
        if compression_alg in ['ArithmeticCoding', 'both']:
            compressed_file = compressed_file_name + '_AC.txt'
            if not os.path.exists(compressed_file):
                print(f"  ❌ Compressed file not found: {compressed_file}")
                return False
            compressed_size = os.path.getsize(compressed_file)
            print(f"  Compressed size: {compressed_size:,} bytes")
            print(f"  Compression ratio: {compressed_size / actual_size:.2%}")
        
        # Decode
        print(f"  Decoding...")
        try:
            # Read metrics to get total length
            import json
            with open(compressed_file_name + '_metrics.json') as metrics_file:
                total_length = json.load(metrics_file)['$N_T$'][0]
            
            if compression_alg in ['ArithmeticCoding', 'both']:
                decoded_text = decoder.decode_AC(
                    win_size=win_len,
                    starter_tokens=None,
                    total_length=total_length,
                    compressed_file_name=compressed_file
                )
            elif compression_alg == 'RankZip':
                compressed_file = compressed_file_name + '_RZ.txt'
                decoded_text = decoder.decode_ranks(
                    win_size=win_len,
                    starter_tokens=None,
                    compressed_file_name=compressed_file
                )
        except Exception as e:
            print(f"  ❌ Error during decoding: {e}")
            import traceback
            traceback.print_exc()
            return False
        
        # Verify round-trip
        print(f"  Decoded text length: {len(decoded_text)} characters")
        
        if original_text == decoded_text:
            print(f"  ✓ Round-trip verification PASSED")
            return True
        else:
            print(f"  ❌ Round-trip verification FAILED")
            print(f"  Original length: {len(original_text)}")
            print(f"  Decoded length: {len(decoded_text)}")
            
            # Find first difference
            min_len = min(len(original_text), len(decoded_text))
            for i in range(min_len):
                if original_text[i] != decoded_text[i]:
                    print(f"  First difference at position {i}")
                    print(f"    Original: {repr(original_text[max(0, i-20):i+20])}")
                    print(f"    Decoded:  {repr(decoded_text[max(0, i-20):i+20])}")
                    break
            
            return False


def main():
    parser = argparse.ArgumentParser(description='Test Qwen3 LLMzip compression/decompression')
    parser.add_argument('--model_path', type=str, default='models/Qwen3-0.6B',
                        help='Path to Qwen3 model directory')
    parser.add_argument('--small_only', action='store_true',
                        help='Only run smallest test (1KB) for quick validation')
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        print(f"Please run: ./scripts/setup_qwen3.sh")
        return 1
    
    print("="*60)
    print("Qwen3 LLMzip Compression Round-Trip Tests")
    print("="*60)
    
    # Test different data sizes
    if args.small_only:
        test_sizes = [1]  # Just 1KB for quick validation
        print("\nRunning quick validation test (1KB only)...")
    else:
        test_sizes = [1, 10, 50]  # 1KB, 10KB, 50KB
        print(f"\nTesting with data sizes: {', '.join(f'{s}KB' for s in test_sizes)}")
    
    print(f"Model path: {args.model_path}")
    
    results = []
    for size_kb in test_sizes:
        # Use smaller window for smaller files to make tests faster
        # Window length is capped at 255 to keep tests reasonably fast
        # For larger files, we use a proportional window (size_kb * 10)
        win_len = min(255, size_kb * 10)
        
        passed = test_compression_roundtrip(
            model_path=args.model_path,
            test_size_kb=size_kb,
            win_len=win_len,
            compression_alg='ArithmeticCoding'
        )
        results.append((size_kb, passed))
    
    # Print summary
    print(f"\n{'='*60}")
    print("Test Summary")
    print(f"{'='*60}")
    
    all_passed = True
    for size_kb, passed in results:
        status = "✓ PASSED" if passed else "❌ FAILED"
        print(f"  {size_kb}KB: {status}")
        if not passed:
            all_passed = False
    
    print(f"{'='*60}")
    
    if all_passed:
        print("✅ All tests PASSED!")
        return 0
    else:
        print("❌ Some tests FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(main())
