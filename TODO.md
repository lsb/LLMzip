# TODO: ONNX Multi-language Integration

## Immediate
- [ ] Add an initial `pyproject.toml` configured for `uv` to manage Python dependencies (pin `onnxruntime`, `transformers`, and tooling) and wire CI to run `uv sync`.
- [ ] Download and vendor `model_q4f16.onnx` plus tokenizer files for offline use (or document caching path) and record SHA256 in the plan (fail CI if the hash differs from the recorded value).
- [ ] Inspect ONNX graph inputs to confirm dynamic batch/sequence axes; document supported batch sizes and wire warnings if static shapes force batch size = 1, and validate that logits outputs expose the same symbolic batch dimension.
- [ ] Add Python ONNX runtime wrapper that outputs next-token log-probs and exposes a `--self-check` that prints input shapes and provider info; include a `--verify-sha` option that recomputes the ONNX hash before running.
- [ ] Add JavaScript (Node + browser) ONNX runtime path using `transformers.js`, including a metadata dump script mirroring the Python self-check and a startup hash check that matches the recorded SHA.
- [ ] Create shared tokenizer loader ensuring identical preprocessing across Python and JS.
- [ ] Implement cross-language round-trip test: compress `seq 1 1000` in Python, decompress in JS; repeat inverse direction with strict tokenizer/version parity and log-prob tolerance checks.
- [ ] Add automated test target (e.g., `npm test` + `pytest`) to run both directions in CI.

## Near-term
- [ ] Provide CLI/README instructions for running the ONNX-based compression test offline.
- [ ] Expose batch size configuration, defaulting to 1 until dynamic support is verified, with guardrails if ONNX shapes are static.
- [ ] Add model integrity checksums for ONNX artifacts.
- [ ] Benchmark CPU latency for both languages; record baseline numbers.
- [ ] Explore WebGPU backend in browsers for potential speedup.

## Quantization Expansion
- [ ] Script an ONNX export pipeline for arbitrary HF models with dynamic axes preserved.
- [ ] Implement 4-bit quantization using `MatMulNBits` and `GatherNBits` (group size 16 for weights and embeddings).
- [ ] Add validation comparing quantized vs FP16 logits on a small corpus; gate acceptance on acceptable KL divergence/perplexity delta.
- [ ] Ensure generated ONNX graphs remain compatible with `onnxruntime-web` (opset coverage, kernel availability).
- [ ] Package artifacts for browser delivery (chunked weights or `transformers.js` hub format).

## Open Questions
- [ ] Confirm best practice for sharing tokenizer files between Python and JS without duplicating assets.
- [ ] Decide on compression format for probability outputs (e.g., JSON + binary sidecar) to keep round-trip deterministic.
- [ ] Evaluate whether to include beam/temperature controls in the ONNX wrapper or enforce greedy decoding for reproducibility.
