#!/usr/bin/env python3
"""
Comprehensive enwik9 compression benchmarks for LLMzip with Qwen3.

This script runs compression benchmarks across different:
- Text sizes: 1K, 2K, 4K, 8K, 16K, 32K, 64K bytes
- Starting positions: 0, 100M, 200M, 300M, 400M, 500M, 600M, 700M, 800M, 900M
- Window lengths: 8, 16, 32, 64, 128, 256, 512, 1024

Outputs:
- Raw JSON results with all compression ratios and decompression timings
- Aggregated results averaged over starting positions
- PNG charts for each window length showing bits per character vs throughput

Usage:
    python benchmark_enwik9.py --model_path models/Qwen3-0.6B --output_dir benchmark_results
"""

import os
import sys
import json
import time
import tempfile
import urllib.request
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


# Benchmark parameters
TEXT_SIZES = [1024, 2048, 4096, 8192, 16384, 32768, 65536]  # bytes
STARTING_POSITIONS = [0, 100_000_000, 200_000_000, 300_000_000, 400_000_000,
                      500_000_000, 600_000_000, 700_000_000, 800_000_000, 900_000_000]
WINDOW_LENGTHS = [8, 16, 32, 64, 128, 256, 512, 1024]

ENWIK9_URL = "http://mattmahoney.net/dc/enwik9.zip"
ENWIK9_SIZE = 1_000_000_000  # 1 GB


def download_enwik9(data_dir: str) -> str:
    """
    Download and extract enwik9 dataset if not present.
    
    Args:
        data_dir: Directory to store the dataset
        
    Returns:
        Path to the enwik9 file
    """
    os.makedirs(data_dir, exist_ok=True)
    enwik9_path = os.path.join(data_dir, "enwik9")
    zip_path = os.path.join(data_dir, "enwik9.zip")
    
    if os.path.exists(enwik9_path):
        file_size = os.path.getsize(enwik9_path)
        if file_size == ENWIK9_SIZE:
            print(f"enwik9 already exists at {enwik9_path} ({file_size:,} bytes)")
            return enwik9_path
        else:
            print(f"enwik9 exists but has wrong size ({file_size:,} bytes, expected {ENWIK9_SIZE:,})")
    
    print(f"Downloading enwik9 from {ENWIK9_URL}...")
    print("This may take a while (file is ~1GB compressed)...")
    
    # Download with progress
    def reporthook(count, block_size, total_size):
        percent = int(count * block_size * 100 / total_size)
        downloaded = count * block_size / (1024 * 1024)
        total = total_size / (1024 * 1024)
        sys.stdout.write(f"\rDownloading: {percent}% ({downloaded:.1f}/{total:.1f} MB)")
        sys.stdout.flush()
    
    urllib.request.urlretrieve(ENWIK9_URL, zip_path, reporthook)
    print("\nDownload complete!")
    
    # Extract
    print("Extracting enwik9.zip...")
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_dir)
    
    # Verify
    if os.path.exists(enwik9_path):
        file_size = os.path.getsize(enwik9_path)
        print(f"Extracted: {enwik9_path} ({file_size:,} bytes)")
        
        # Remove zip file to save space
        os.remove(zip_path)
        print(f"Removed {zip_path}")
        
        return enwik9_path
    else:
        raise RuntimeError(f"Failed to extract enwik9 to {enwik9_path}")


def extract_text_chunk(enwik9_path: str, start_pos: int, size: int) -> str:
    """
    Extract a text chunk from enwik9.
    
    Args:
        enwik9_path: Path to enwik9 file
        start_pos: Starting byte position
        size: Number of bytes to extract
        
    Returns:
        Extracted text as string
    """
    with open(enwik9_path, 'rb') as f:
        f.seek(start_pos)
        data = f.read(size)
    
    # Decode as UTF-8 with error handling (enwik9 is mostly ASCII/UTF-8)
    text = data.decode('utf-8', errors='replace')
    return text


def run_compression_test(
    model,
    tokenizer,
    text: str,
    win_len: int,
    compression_alg: str = 'ArithmeticCoding'
) -> Dict[str, Any]:
    """
    Run a single compression/decompression test.
    
    Args:
        model: Qwen3Model instance
        tokenizer: Qwen3Tokenizer instance
        text: Text to compress
        win_len: Window length for compression
        compression_alg: Compression algorithm ('ArithmeticCoding' or 'RankZip')
        
    Returns:
        Dictionary with compression metrics
    """
    from qwen3 import Qwen3_encode, Qwen3_decode
    
    # Create encoder and decoder
    encoder = Qwen3_encode(model, tokenizer)
    decoder = Qwen3_decode(model, tokenizer)
    
    # Create temporary directory for compressed files
    with tempfile.TemporaryDirectory() as tmpdir:
        compressed_file_name = os.path.join(tmpdir, 'benchmark')
        
        # Tokenize
        tokens_full = np.array(tokenizer.encode(text, bos=False, eos=False))
        n_tokens = len(tokens_full)
        n_chars = len(text)
        original_size_bytes = len(text.encode('utf-8'))
        
        # Skip if tokens are less than window length
        if n_tokens <= win_len + 1:
            return {
                'error': f'Token count ({n_tokens}) <= window length + 1 ({win_len + 1})',
                'n_tokens': n_tokens,
                'n_chars': n_chars,
                'original_size_bytes': original_size_bytes
            }
        
        # Compression
        compression_start = time.time()
        encoder.encode_from_tokens(
            win_size=win_len,
            compression_alg=compression_alg,
            compressed_file_name=compressed_file_name,
            tokens_full=tokens_full,
            batched_encode=False,
            with_context_start=False
        )
        compression_time = time.time() - compression_start
        
        # Get compressed file size
        if compression_alg == 'ArithmeticCoding':
            compressed_file = compressed_file_name + '_AC.txt'
        else:
            compressed_file = compressed_file_name + '_RZ.txt'
        
        compressed_size_bytes = os.path.getsize(compressed_file)
        compressed_size_bits = compressed_size_bytes * 8
        
        # Load metrics
        with open(compressed_file_name + '_metrics.json') as f:
            metrics = json.load(f)
        
        total_length = metrics['$N_T$'][0]
        
        # Decompression
        decompression_start = time.time()
        if compression_alg == 'ArithmeticCoding':
            decoded_text = decoder.decode_AC(
                win_size=win_len,
                starter_tokens=None,
                total_length=total_length,
                compressed_file_name=compressed_file
            )
        else:
            decoded_text = decoder.decode_ranks(
                win_size=win_len,
                starter_tokens=None,
                compressed_file_name=compressed_file
            )
        decompression_time = time.time() - decompression_start
        
        # Verify
        verified = text == decoded_text
        
        # Calculate metrics
        compression_ratio = compressed_size_bytes / original_size_bytes
        bits_per_char = compressed_size_bits / n_chars
        
        return {
            'n_tokens': n_tokens,
            'n_chars': n_chars,
            'original_size_bytes': original_size_bytes,
            'compressed_size_bytes': compressed_size_bytes,
            'compressed_size_bits': compressed_size_bits,
            'compression_ratio': compression_ratio,
            'bits_per_char': bits_per_char,
            'compression_time_sec': compression_time,
            'decompression_time_sec': decompression_time,
            'verified': verified,
            'compression_speed_bps': original_size_bytes / compression_time if compression_time > 0 else 0,
            'decompression_speed_bps': original_size_bytes / decompression_time if decompression_time > 0 else 0
        }


def run_benchmarks(
    model_path: str,
    enwik9_path: str,
    output_dir: str,
    text_sizes: List[int] = None,
    starting_positions: List[int] = None,
    window_lengths: List[int] = None,
    compression_alg: str = 'ArithmeticCoding'
) -> Dict[str, Any]:
    """
    Run all benchmark combinations.
    
    Args:
        model_path: Path to Qwen3 model
        enwik9_path: Path to enwik9 file
        output_dir: Output directory for results
        text_sizes: List of text sizes to test (bytes)
        starting_positions: List of starting positions in enwik9
        window_lengths: List of window lengths
        compression_alg: Compression algorithm to use
        
    Returns:
        Dictionary with all results
    """
    from qwen3 import Qwen3Model, Qwen3Tokenizer
    
    # Use defaults if not provided
    if text_sizes is None:
        text_sizes = TEXT_SIZES
    if starting_positions is None:
        starting_positions = STARTING_POSITIONS
    if window_lengths is None:
        window_lengths = WINDOW_LENGTHS
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("LLMzip enwik9 Compression Benchmark")
    print("=" * 60)
    print(f"Model: {model_path}")
    print(f"Dataset: {enwik9_path}")
    print(f"Text sizes: {[f'{s//1024}K' for s in text_sizes]}")
    print(f"Starting positions: {[f'{p//1_000_000}M' for p in starting_positions]}")
    print(f"Window lengths: {window_lengths}")
    print(f"Compression algorithm: {compression_alg}")
    print("=" * 60)
    
    # Load model
    print("\nLoading model...")
    model = Qwen3Model(model_path=model_path, max_batch_size=1)
    tokenizer = Qwen3Tokenizer(model_path=model_path)
    print("Model loaded successfully!\n")
    
    # Results storage
    raw_results = []
    total_tests = len(text_sizes) * len(starting_positions) * len(window_lengths)
    test_count = 0
    
    # Run all combinations
    for text_size in text_sizes:
        for start_pos in starting_positions:
            # Check if we can extract this chunk
            if start_pos + text_size > ENWIK9_SIZE:
                print(f"Skipping: start_pos={start_pos}, text_size={text_size} exceeds enwik9 size")
                continue
            
            # Extract text chunk
            text = extract_text_chunk(enwik9_path, start_pos, text_size)
            
            for win_len in window_lengths:
                test_count += 1
                print(f"\n[{test_count}/{total_tests}] Testing: "
                      f"text_size={text_size//1024}K, "
                      f"start_pos={start_pos//1_000_000}M, "
                      f"win_len={win_len}")
                
                try:
                    result = run_compression_test(
                        model=model,
                        tokenizer=tokenizer,
                        text=text,
                        win_len=win_len,
                        compression_alg=compression_alg
                    )
                    
                    result['text_size_bytes'] = text_size
                    result['start_pos'] = start_pos
                    result['win_len'] = win_len
                    result['compression_alg'] = compression_alg
                    
                    if 'error' not in result:
                        print(f"  Compression ratio: {result['compression_ratio']:.4f}")
                        print(f"  Bits per char: {result['bits_per_char']:.4f}")
                        print(f"  Compression time: {result['compression_time_sec']:.2f}s")
                        print(f"  Decompression time: {result['decompression_time_sec']:.2f}s")
                        print(f"  Verified: {result['verified']}")
                    else:
                        print(f"  Skipped: {result['error']}")
                    
                except Exception as e:
                    result = {
                        'text_size_bytes': text_size,
                        'start_pos': start_pos,
                        'win_len': win_len,
                        'compression_alg': compression_alg,
                        'error': str(e)
                    }
                    print(f"  Error: {e}")
                
                raw_results.append(result)
    
    # Save raw results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_results_path = os.path.join(output_dir, f"raw_results_{timestamp}.json")
    with open(raw_results_path, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'model_path': model_path,
            'enwik9_path': enwik9_path,
            'compression_alg': compression_alg,
            'results': raw_results
        }, f, indent=2)
    print(f"\nRaw results saved to: {raw_results_path}")
    
    # Generate aggregated results
    aggregated = aggregate_results(raw_results, text_sizes, window_lengths)
    aggregated_path = os.path.join(output_dir, f"aggregated_results_{timestamp}.json")
    with open(aggregated_path, 'w') as f:
        json.dump(aggregated, f, indent=2)
    print(f"Aggregated results saved to: {aggregated_path}")
    
    # Generate visualizations
    generate_charts(aggregated, output_dir, timestamp)
    
    return {
        'raw_results': raw_results,
        'aggregated': aggregated,
        'raw_results_path': raw_results_path,
        'aggregated_path': aggregated_path
    }


def aggregate_results(
    raw_results: List[Dict],
    text_sizes: List[int],
    window_lengths: List[int]
) -> Dict[str, Any]:
    """
    Aggregate results by averaging over starting positions.
    
    Args:
        raw_results: List of raw result dictionaries
        text_sizes: List of text sizes
        window_lengths: List of window lengths
        
    Returns:
        Aggregated results dictionary
    """
    aggregated = {}
    
    for win_len in window_lengths:
        aggregated[win_len] = {}
        
        for text_size in text_sizes:
            # Filter results for this combination
            filtered = [
                r for r in raw_results
                if r.get('win_len') == win_len
                and r.get('text_size_bytes') == text_size
                and 'error' not in r
                and r.get('verified', False)
            ]
            
            if not filtered:
                continue
            
            # Calculate averages
            avg_result = {
                'text_size_bytes': text_size,
                'text_size_label': f"{text_size // 1024}K",
                'win_len': win_len,
                'n_samples': len(filtered),
                'avg_compression_ratio': np.mean([r['compression_ratio'] for r in filtered]),
                'std_compression_ratio': np.std([r['compression_ratio'] for r in filtered]),
                'avg_bits_per_char': np.mean([r['bits_per_char'] for r in filtered]),
                'std_bits_per_char': np.std([r['bits_per_char'] for r in filtered]),
                'avg_compression_time_sec': np.mean([r['compression_time_sec'] for r in filtered]),
                'std_compression_time_sec': np.std([r['compression_time_sec'] for r in filtered]),
                'avg_decompression_time_sec': np.mean([r['decompression_time_sec'] for r in filtered]),
                'std_decompression_time_sec': np.std([r['decompression_time_sec'] for r in filtered]),
                'avg_compression_speed_bps': np.mean([r['compression_speed_bps'] for r in filtered]),
                'avg_decompression_speed_bps': np.mean([r['decompression_speed_bps'] for r in filtered]),
            }
            
            aggregated[win_len][text_size] = avg_result
    
    return aggregated


def generate_charts(
    aggregated: Dict[str, Any],
    output_dir: str,
    timestamp: str
) -> List[str]:
    """
    Generate line charts for each window length.
    
    Args:
        aggregated: Aggregated results dictionary
        output_dir: Output directory for charts
        timestamp: Timestamp for file naming
        
    Returns:
        List of generated chart file paths
    """
    import matplotlib.pyplot as plt
    
    chart_paths = []
    
    for win_len in sorted(aggregated.keys()):
        data = aggregated[win_len]
        
        if not data:
            continue
        
        # Sort by text size
        sorted_sizes = sorted(data.keys())
        
        # Prepare data for plotting
        text_size_labels = []
        bits_per_char = []
        throughputs = []  # bytes per second
        
        for text_size in sorted_sizes:
            result = data[text_size]
            text_size_labels.append(result['text_size_label'])
            bits_per_char.append(result['avg_bits_per_char'])
            throughputs.append(result['avg_compression_speed_bps'])
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot with throughput as x-axis (log scale)
        ax.plot(throughputs, bits_per_char, 'o-', linewidth=2, markersize=8)
        
        # Add text size labels to points
        for i, (x, y, label) in enumerate(zip(throughputs, bits_per_char, text_size_labels)):
            ax.annotate(
                label,
                (x, y),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=9
            )
        
        ax.set_xscale('log')
        ax.set_xlabel('Compression Throughput (bytes/second)', fontsize=12)
        ax.set_ylabel('Bits per Character', fontsize=12)
        ax.set_title(f'LLMzip Compression Quality vs Speed\nWindow Length = {win_len}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Add legend explaining x-axis
        ax.text(
            0.02, 0.02,
            f'Higher throughput = faster compression\nLower bits/char = better compression',
            transform=ax.transAxes,
            fontsize=9,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(output_dir, f"chart_winlen_{win_len}_{timestamp}.png")
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        chart_paths.append(chart_path)
        print(f"Chart saved: {chart_path}")
    
    # Generate summary chart with all window lengths
    generate_summary_chart(aggregated, output_dir, timestamp)
    
    return chart_paths


def generate_summary_chart(
    aggregated: Dict[str, Any],
    output_dir: str,
    timestamp: str
) -> str:
    """
    Generate a summary chart showing all window lengths.
    
    Args:
        aggregated: Aggregated results dictionary
        output_dir: Output directory
        timestamp: Timestamp for file naming
        
    Returns:
        Path to generated chart
    """
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(aggregated)))
    
    for i, win_len in enumerate(sorted(aggregated.keys())):
        data = aggregated[win_len]
        
        if not data:
            continue
        
        sorted_sizes = sorted(data.keys())
        throughputs = [data[s]['avg_compression_speed_bps'] for s in sorted_sizes]
        bits_per_char = [data[s]['avg_bits_per_char'] for s in sorted_sizes]
        
        ax.plot(
            throughputs,
            bits_per_char,
            'o-',
            linewidth=2,
            markersize=6,
            color=colors[i],
            label=f'win_len={win_len}'
        )
    
    ax.set_xscale('log')
    ax.set_xlabel('Compression Throughput (bytes/second)', fontsize=12)
    ax.set_ylabel('Bits per Character', fontsize=12)
    ax.set_title('LLMzip Compression Quality vs Speed\n(All Window Lengths)', fontsize=14)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    chart_path = os.path.join(output_dir, f"chart_summary_{timestamp}.png")
    plt.savefig(chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Summary chart saved: {chart_path}")
    return chart_path


def main():
    parser = argparse.ArgumentParser(
        description='Run enwik9 compression benchmarks for LLMzip with Qwen3'
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default='models/Qwen3-0.6B',
        help='Path to Qwen3 model directory'
    )
    parser.add_argument(
        '--data_dir',
        type=str,
        default='data',
        help='Directory for enwik9 dataset'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='benchmark_results',
        help='Output directory for results'
    )
    parser.add_argument(
        '--compression_alg',
        type=str,
        default='ArithmeticCoding',
        choices=['ArithmeticCoding', 'RankZip'],
        help='Compression algorithm to use'
    )
    parser.add_argument(
        '--text_sizes',
        type=str,
        default=None,
        help='Comma-separated list of text sizes in bytes (e.g., "1024,2048,4096")'
    )
    parser.add_argument(
        '--window_lengths',
        type=str,
        default=None,
        help='Comma-separated list of window lengths (e.g., "8,16,32")'
    )
    parser.add_argument(
        '--starting_positions',
        type=str,
        default=None,
        help='Comma-separated list of starting positions (e.g., "0,100000000")'
    )
    parser.add_argument(
        '--skip_download',
        action='store_true',
        help='Skip enwik9 download (assume it exists)'
    )
    
    args = parser.parse_args()
    
    # Parse optional comma-separated lists
    text_sizes = None
    if args.text_sizes:
        text_sizes = [int(x.strip()) for x in args.text_sizes.split(',')]
    
    window_lengths = None
    if args.window_lengths:
        window_lengths = [int(x.strip()) for x in args.window_lengths.split(',')]
    
    starting_positions = None
    if args.starting_positions:
        starting_positions = [int(x.strip()) for x in args.starting_positions.split(',')]
    
    # Check model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        print("Please run: ./scripts/setup_qwen3.sh")
        return 1
    
    # Download or verify enwik9
    if args.skip_download:
        enwik9_path = os.path.join(args.data_dir, "enwik9")
        if not os.path.exists(enwik9_path):
            print(f"Error: enwik9 not found at {enwik9_path}")
            print("Run without --skip_download to download it")
            return 1
    else:
        enwik9_path = download_enwik9(args.data_dir)
    
    # Run benchmarks
    results = run_benchmarks(
        model_path=args.model_path,
        enwik9_path=enwik9_path,
        output_dir=args.output_dir,
        text_sizes=text_sizes,
        starting_positions=starting_positions,
        window_lengths=window_lengths,
        compression_alg=args.compression_alg
    )
    
    print("\n" + "=" * 60)
    print("Benchmark Complete!")
    print("=" * 60)
    print(f"Raw results: {results['raw_results_path']}")
    print(f"Aggregated results: {results['aggregated_path']}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
