#!/usr/bin/env python3
# Qwen3-0.6B LLMzip runner script
# Simplified version without distributed setup (no torchrun required)

from typing import Tuple
import os
import sys
import torch
import fire
import time
import json
import numpy as np
from pathlib import Path

from qwen3 import Qwen3Model, Qwen3Tokenizer, Qwen3_encode, Qwen3_decode

### Command to run
# python qwen3_run.py --model_path models/Qwen3-0.6B --win_len 511 --text_file sample.txt --compression_folder output/

### For precise compression tests, use these options:
# compression_alg='both', encode_decode=0, batched_encode=True, verify_save_decoded=0, with_context_start=True


def load(
    model_path: str,
    max_batch_size: int = 1
):
    """
    Load Qwen3 model and tokenizer.
    
    Args:
        model_path: Path to Qwen3 model directory
        max_batch_size: Maximum batch size for inference
        
    Returns:
        Encoder and Decoder instances
    """
    start_time = time.time()
    
    print(f"Loading Qwen3 model from {model_path}...")
    
    # Load tokenizer
    tokenizer = Qwen3Tokenizer(model_path=model_path)
    
    # Load model
    model = Qwen3Model(model_path=model_path, max_batch_size=max_batch_size)
    
    # Create encoder and decoder
    Encoder = Qwen3_encode(model, tokenizer)
    Decoder = Qwen3_decode(model, tokenizer)
    
    print(f"Loaded in {time.time() - start_time:.2f} seconds")
    return Encoder, Decoder


def verify_text(compressed_file_name, text_file, text_decoded, context_txt, save_decoded, alg):
    """
    Verify decoded text matches original and optionally save it.
    
    Args:
        compressed_file_name: Base name for compressed files
        text_file: Original text file path
        text_decoded: Decoded text string
        context_txt: Context text to skip (or None)
        save_decoded: Whether to save decoded text
        alg: Algorithm name for reporting
    """
    with open(text_file, 'r') as txt_enc:
        text_encoded = txt_enc.read()
    
    if context_txt is not None:
        text_encoded = text_encoded[len(context_txt):]
        text_decoded = text_decoded[len(context_txt):]
    
    if text_encoded == text_decoded:
        print(f'✓ Successful decoding using {alg}')
    else:
        print("********!!!!! Error !!!!!*********")
        print("***********Encoded Text************")
        print(text_encoded[:500])  # Print first 500 chars
        print("...")
        print("***********Decoded Text************")
        print(text_decoded[:500])  # Print first 500 chars
        print("...")
    
    if save_decoded:
        if alg == 'ArithmeticCoding':
            with open(compressed_file_name + '_AC_decoded_text.txt', 'w') as txt_dec:
                txt_dec.write(text_decoded)
        else:
            with open(compressed_file_name + '_RZ_decoded_text.txt', 'w') as txt_dec:
                txt_dec.write(text_decoded)


def main(
    model_path: str = "models/Qwen3-0.6B",
    win_len: int = 511,
    text_file: str = None,
    compression_folder: str = "output",
    max_batch_size: int = 1,
    compression_alg: str = 'ArithmeticCoding',
    encode_decode: int = 2,
    batched_encode: bool = False,
    verify_save_decoded: int = 2,
    with_context_start: bool = False
):
    """
    Main function for Qwen3 LLMzip compression/decompression.
    
    Args:
        model_path: Path to Qwen3 model directory (default: models/Qwen3-0.6B)
        win_len: Context window length (must be reasonable for the model)
        text_file: Input text file to compress
        compression_folder: Output folder for compressed files
        max_batch_size: Maximum batch size for encoding (default: 1)
        compression_alg: 'ArithmeticCoding', 'RankZip', or 'both'
        encode_decode: 0=encode only, 1=decode only, 2=both
        batched_encode: Use batched encoding (faster but decode won't work)
        verify_save_decoded: 0=don't verify, 1=verify only, 2=verify and save
        with_context_start: Exclude initial context from encoding
    """
    # Validate inputs
    assert encode_decode in [0, 1, 2], f'encode_decode must be in {[0, 1, 2]}'
    assert compression_alg in ['ArithmeticCoding', 'RankZip', 'both'], \
        'compression_alg must be ArithmeticCoding, RankZip, or both'
    
    if text_file is None:
        print("Error: text_file is required")
        print("\nUsage example:")
        print("  python qwen3_run.py --model_path models/Qwen3-0.6B --win_len 511 --text_file sample.txt --compression_folder output/")
        return
    
    if not os.path.exists(text_file):
        print(f"Error: text_file '{text_file}' does not exist")
        return
    
    if batched_encode:
        print("Warning: Decoding doesn't work when using batched encode")
    
    start_time_main = time.time()
    
    # Determine encode/decode flags
    encode = encode_decode % 2 == 0  # Convert to Bool
    decode = encode_decode > 0       # Convert to Bool
    
    if decode:
        batched_encode = False
    
    # Load model and create encoder/decoder
    Encoder, Decoder = load(model_path, max_batch_size)
    
    # Create output folder
    os.makedirs(compression_folder, exist_ok=True)
    compressed_file_name = compression_folder + f'/LLMzip_{win_len}'
    
    # Read input text
    with open(text_file, 'r') as f_in:
        text_input = f_in.read()
    
    if encode:
        # Encoding
        print(f"\nEncoding {text_file}...")
        tokens_full = np.array(Encoder.tokenizer.encode(text_input, bos=False, eos=False))
        
        if with_context_start:
            starter_tokens = tokens_full[:win_len]
            np.save(compressed_file_name + '_starter_tokens.npy', starter_tokens)
        
        Encoder.encode_from_tokens(
            win_len,
            compression_alg,
            compressed_file_name,
            tokens_full=tokens_full,
            batched_encode=batched_encode,
            with_context_start=with_context_start
        )
    
    if decode:
        # Decoding
        print(f"\nDecoding...")
        with open(compressed_file_name + '_metrics.json') as metrics_file:
            total_length = json.load(metrics_file)['$N_T$'][0]
        
        if with_context_start:
            starter_tokens = np.load(compressed_file_name + '_starter_tokens.npy')
            context_txt = Encoder.tokenizer.decode(starter_tokens.tolist())
        else:
            starter_tokens = None
            context_txt = None
        
        if (compression_alg == 'ArithmeticCoding') or (compression_alg == 'both'):
            compressed_file_name_full = compressed_file_name + '_AC.txt'
            
            decoded_text_ac = Decoder.decode_AC(
                win_len,
                starter_tokens,
                total_length,
                compressed_file_name_full
            )
            if verify_save_decoded > 0:
                verify_text(
                    compressed_file_name,
                    text_file,
                    decoded_text_ac,
                    context_txt,
                    verify_save_decoded == 2,
                    'ArithmeticCoding'
                )
        
        if (compression_alg == 'RankZip') or (compression_alg == 'both'):
            compressed_file_name_full = compressed_file_name + '_RZ.txt'
            
            decoded_text_rz = Decoder.decode_ranks(
                win_len,
                starter_tokens,
                compressed_file_name_full
            )
            if verify_save_decoded > 0:
                verify_text(
                    compressed_file_name,
                    text_file,
                    decoded_text_rz,
                    context_txt,
                    verify_save_decoded == 2,
                    'RankZip'
                )
    
    print(f"\n✓ Completed in {time.time() - start_time_main:.2f} seconds")


if __name__ == "__main__":
    fire.Fire(main)
