# ONNX Integration Plan

## Objectives
- Provide multi-language LLM inference for LLMzip using ONNX:
  - **Python**: server-side inference via `onnxruntime` and Hugging Face `transformers` tooling.
  - **JavaScript**: browser and Node.js inference via `transformers.js` with ONNX backends.
- Use the SmolLM2 135M Instruct ONNX checkpoint (`model_q4f16.onnx`) for shared behavior across languages. Keep a provenance note (source URL and expected SHA256) to ensure CI and offline workflows fetch the exact artifact. Track the hash in both the plan and the download script so the fetch step is reproducible and verifiable.
- Keep batch size configurable; default to **1** until dynamic batch support is confirmed. Document detection logic for dynamic axes so both runtimes can agree on whether >1 is allowed.
- Enable future quantization of arbitrary Hugging Face models with 4-bit weights/embeddings using `MatMulNBits` and `GatherNBits` ops (group size 16).

## Model
- Source: [`HuggingFaceTB/SmolLM2-135M-Instruct`](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/main/onnx/model_q4f16.onnx).
- Format: ONNX with Q4F16 quantization.
- Dynamic axes: must verify whether inputs expose dynamic batch and sequence dimensions. If the batch dimension is static, keep batch size at 1; otherwise allow user-defined batches. Detection steps:
  - Inspect the ONNX graph with `onnxruntime.tools.onnx_model_info` or `python -c "import onnx; print(onnx.load('model_q4f16.onnx').graph.input)"`.
  - Confirm the presence of symbolic dimensions (e.g., `batch` or `seq`) for `input_ids`, `attention_mask`, and `position_ids`.
  - Validate that outputs (logits) carry the same symbolic batch and sequence axes; if outputs are static while inputs are dynamic, clamp batch size to 1 to avoid runtime shape mismatches.
  - If shapes are fixed (e.g., `[1, 1]`), enforce `batch_size=1` in both Python and JS wrappers and surface a warning in logs.
- Tokenizer: share a single tokenizer vocab across Python and JS via the Hugging Face tokenizer files in the repo or by exporting `tokenizer.json` alongside the ONNX artifact.

## Runtime Strategy
### Python (server side)
- Use `onnxruntime` CPU execution provider by default; optionally enable OpenMP/AVX if available.
- Manage dependencies via `pyproject.toml` + `uv` for reproducible installs (pinning `onnxruntime`, `transformers`, and helper tooling). Add a bootstrap target in CI that runs `uv sync` before tests.
- Provide a thin wrapper aligning inputs (input IDs, attention mask, position IDs if needed) to the ONNX graph signatures.
- Expose a simple probability API returning log-probs for the next token to feed compression.
- Include a self-check command (e.g., `python -m onnxruntime.tools.onnx_model_info model_q4f16.onnx`) that prints input shapes and confirms the selected execution provider.
- Surface a `--verify-sha` flag that recomputes the ONNX file hash and asserts it matches the recorded provenance; fail fast if it differs.

### JavaScript (browser + Node.js)
- Use `transformers.js` with the ONNX runtime backend (`onnxruntime-web` in browser, `onnxruntime-node` in Node.js).
- Mirror the Python input pipeline: tokenize -> run ONNX session -> extract logits -> softmax -> log-probs.
- Keep CPU backend initially; later explore WebGPU for browser acceleration if supported by the model opset.
- Add a small Node.js script to print the ONNX session input metadata and assert parity with the Python view.
- Mirror the Python integrity check by recomputing the ONNX hash during startup; refuse to run if it does not match the recorded SHA.

## Cross-language Compression Test
- Goal: demonstrate deterministic probability generation across Python and JS for the sequence `1..1000`.
- Plan:
  1. Python script tokenizes numbers 1–1000 (as text), runs the ONNX model, and writes compressed output (and any sideband metadata such as tokenizer info) to disk. Capture a JSON report with input shapes and `model_hash` to assert reproducibility.
  2. JavaScript script loads the same tokenizer and ONNX model, reconstructs probabilities, and decompresses to verify round-trip fidelity. Validate that decoded tokens and log-probs match within a small tolerance (e.g., `1e-5`).
  3. Invert the flow (compress in JS, decompress in Python) to ensure equivalence. Treat tokenizer vocab version mismatches as a hard error, not a warning.
- Tests should run fully offline using the committed ONNX and tokenizer assets.

## Quantization Roadmap
- Target: arbitrary Hugging Face transformer models.
- Pipeline:
  1. Export the model to ONNX with dynamic batch/sequence axes preserved.
  2. Apply 4-bit weight/embedding quantization using `MatMulNBits` and `GatherNBits` (group size 16 for both weights and embeddings).
  3. Validate numerics with small validation sets and compare log-probs to the FP16 baseline.
  4. Generate browser-friendly artifacts (optionally split weights) for `transformers.js` compatibility.
- Considerations:
  - Ensure opset compatibility with `onnxruntime` CPU and Web backends.
  - Prefer symmetric quantization for simplicity; document calibration/scale choices.
  - Add tooling to toggle batch size support based on the exported model's dynamic axes.

## Open Questions
- Confirm whether `model_q4f16.onnx` declares dynamic batch and sequence lengths; adapt test batch sizes accordingly.
- Verify tokenizer alignment between Python `transformers` and `transformers.js` for special tokens and padding.
- Determine the best packaging approach for browser delivery (e.g., using `files` config or CDN) while keeping offline test support.
