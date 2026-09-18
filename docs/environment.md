# Environment Lock

This project uses Python 3.12 managed via `uv`.

## Core Packages
- `torch`: 2.14.0 (CPU fallback currently active locally; requires CUDA for execution)
- `transformers`: 5.17.0 (Verified for `AutoModelForMultimodalLM` support)
- `peft`: 0.21.0 (Verified for `use_dora` integration)
- `bitsandbytes`: 0.50.2 (Verified for 4-bit NF4 quantization)
- `trl`: 1.13.0
- `datasets`: 5.0.1

## Hugging Face Cache
- `HF_HOME`: `.cache/huggingface` (Project-local to avoid Windows permission issues)
- `HF_HUB_CACHE`: `.cache/huggingface`
