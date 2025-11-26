# TODO: LLMzip Development

## Current Priority: Qwen3-0.6B Integration & Compression Ratio Comparison

### Immediate Tasks
- [x] Clone Qwen3-0.6B repository from GitHub.com/lsb/Qwen3-0.6B
- [x] Run `make concat` in the Qwen3-0.6B repository to assemble model files
- [x] Verify Qwen3-0.6B model loads correctly with HuggingFace Transformers
- [x] Create LLMzip inference wrapper for Qwen3-0.6B (similar to existing LLaMA wrapper)
- [ ] Implement zstandard level 19 compression for baseline comparison
- [ ] Create compression ratio comparison script (LLMzip+Qwen3 vs zstd-19)
- [x] Download enwik9 dataset for standardized compression benchmarking (via `benchmark_enwik9.py`)
- [x] Create comprehensive enwik9 benchmark script (`benchmark_enwik9.py`)
  - **Text sizes**: 1K, 2K, 4K, 8K, 16K, 32K, 64K bytes
  - **Starting positions**: 0, 100M, 200M, 300M, 400M, 500M, 600M, 700M, 800M, 900M (10 positions)
  - **Window lengths**: 8, 16, 32, 64, 128, 256, 512, 1024
  - **Metrics**: Compression ratio, decompression timing, bits per character
  - **Outputs**: Raw JSON results, aggregated results (averaged over positions), PNG charts per window length
- [ ] Run comprehensive benchmarks and document results
- [ ] Generate comparison report with charts and analysis

### Near-term
- [ ] Optimize Qwen3-0.6B inference performance for compression
- [ ] Test with various file types and document compression characteristics
- [ ] Add CLI flags for model selection (LLaMA vs Qwen3)
- [ ] Document memory requirements and performance trade-offs
- [ ] Add automated tests for Qwen3 integration

### Future: ONNX Multi-language Integration (Deferred)
> **Note**: ONNX integration is deferred until after Qwen3-0.6B baseline is established

- [x] Add an initial `pyproject.toml` configured for `uv` to manage Python dependencies (pin `onnxruntime`, `transformers`, and tooling) and wire CI to run `uv sync`.
- [ ] Export Qwen3-0.6B to ONNX format (or download SmolLM2-135M-Instruct ONNX checkpoint)
- [ ] Add Python ONNX runtime wrapper that outputs next-token log-probs
- [ ] Add JavaScript (Node + browser) ONNX runtime path using `transformers.js`
- [ ] Create shared tokenizer loader ensuring identical preprocessing across Python and JS
- [ ] Implement cross-language round-trip test: compress in Python, decompress in JS
- [ ] Benchmark CPU latency for both languages; record baseline numbers
- [ ] Explore WebGPU backend in browsers for potential speedup

### Future: Quantization Experiments
- [ ] Experiment with 4-bit and 8-bit quantization of Qwen3-0.6B
- [ ] Compare compression ratios: FP16 vs quantized models
- [ ] Measure speed improvements from quantization
- [ ] Script an ONNX export pipeline for arbitrary HF models with quantization
- [ ] Ensure quantized models maintain acceptable compression ratios

## Open Questions
- [ ] What is the optimal context window size for Qwen3-0.6B in compression tasks?
- [ ] How does compression ratio scale with model size (0.6B vs larger models)?
- [ ] What types of data benefit most from LLM-based compression vs traditional methods?
- [ ] Should we support streaming compression for large files?
