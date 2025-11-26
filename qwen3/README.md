# Qwen3-0.6B LLMzip Wrapper

This module provides LLMzip compression/decompression using the Qwen3-0.6B model via HuggingFace Transformers.

## Quick Start

### Prerequisites

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Download and setup the Qwen3-0.6B model:
```bash
./scripts/setup_qwen3.sh
```

### Basic Usage

Compress a text file:
```bash
python qwen3_run.py \
  --model_path models/Qwen3-0.6B \
  --win_len 511 \
  --text_file sample.txt \
  --compression_folder output/ \
  --compression_alg ArithmeticCoding \
  --encode_decode 0
```

Compress and decompress (round-trip test):
```bash
python qwen3_run.py \
  --model_path models/Qwen3-0.6B \
  --win_len 511 \
  --text_file sample.txt \
  --compression_folder output/ \
  --compression_alg ArithmeticCoding \
  --encode_decode 2
```

## Command-Line Options

- `--model_path`: Path to Qwen3 model directory (default: `models/Qwen3-0.6B`)
- `--win_len`: Context window length (default: 511, max depends on model)
- `--text_file`: Input text file to compress (required)
- `--compression_folder`: Output folder for compressed files (default: `output`)
- `--compression_alg`: Compression algorithm:
  - `ArithmeticCoding`: Use arithmetic coding (recommended)
  - `RankZip`: Use rank-based compression with zlib
  - `both`: Use both methods for comparison
- `--encode_decode`: Operation mode:
  - `0`: Encode only
  - `1`: Decode only
  - `2`: Both encode and decode (default)
- `--verify_save_decoded`: Verification mode:
  - `0`: Don't verify or save decoded text
  - `1`: Verify only (compare decoded with original)
  - `2`: Verify and save decoded text (default)
- `--with_context_start`: Exclude initial context from encoding (advanced)
- `--batched_encode`: Use batched encoding for faster compression (disables decoding)

## Module Structure

```
qwen3/
├── __init__.py          # Module exports
├── model.py             # Qwen3Model wrapper (HuggingFace Transformers)
├── tokenizer.py         # Qwen3Tokenizer wrapper
├── llmzip_utils.py      # Utility functions for compression
└── LLMzip.py            # Qwen3_encode and Qwen3_decode classes
```

## Key Differences from LLaMA Wrapper

1. **No distributed setup required**: The Qwen3 wrapper runs on a single GPU or CPU
2. **HuggingFace Transformers**: Uses standard HuggingFace API instead of fairscale
3. **Larger vocabulary**: Qwen3 has 151,936 tokens vs LLaMA's ~32,000
4. **Simpler runner**: `qwen3_run.py` doesn't require `torchrun`
5. **Automatic device detection**: Falls back to CPU if CUDA is not available

## Example Output

After running compression, you'll see metrics like:
```
Compression Ratio for Arithmetic Coding: 2.45 bits/char
{
  '$N_C$': [1234],
  '$N_T$': [456],
  '$H_{ub}$': ['2.41'],
  'Qwen3+AC compressed file size': [3050],
  '$\rho_{Qwen3+AC}$': [2.45]
}
```

## Python API

```python
from qwen3 import Qwen3Model, Qwen3Tokenizer, Qwen3_encode, Qwen3_decode
import numpy as np

# Load model and tokenizer
model = Qwen3Model(model_path="models/Qwen3-0.6B")
tokenizer = Qwen3Tokenizer(model_path="models/Qwen3-0.6B")

# Create encoder/decoder
encoder = Qwen3_encode(model, tokenizer)
decoder = Qwen3_decode(model, tokenizer)

# Encode text
text = "Your text here"
tokens = np.array(tokenizer.encode(text, bos=False, eos=False))
encoder.encode_from_tokens(
    win_size=511,
    compression_alg='ArithmeticCoding',
    compressed_file_name='output/compressed',
    tokens_full=tokens
)

# Decode
decoded_text = decoder.decode_AC(
    win_size=511,
    starter_tokens=None,
    total_length=len(tokens),
    compressed_file_name='output/compressed_AC.txt'
)
```

## Performance Notes

- **GPU**: Recommended for faster compression (requires CUDA-capable GPU)
- **CPU**: Works but slower; suitable for small files
- **Memory**: Qwen3-0.6B requires ~1.2 GB in bfloat16, ~2.4 GB in float32
- **Context window**: Larger windows (up to 40960) give better compression but use more memory
- **Recommended window**: 511 is a good balance for most use cases

## Troubleshooting

**Error: Model directory not found**
- Run `./scripts/setup_qwen3.sh` to download and setup the model

**Error: CUDA out of memory**
- Reduce `--win_len` (e.g., from 511 to 255)
- Close other programs using GPU memory
- Fall back to CPU (automatic if CUDA unavailable)

**Error: text_file is required**
- Specify input file with `--text_file path/to/file.txt`

**Slow compression**
- Use `--batched_encode` for encoding-only (disables decoding)
- Use GPU instead of CPU
- Reduce window length
