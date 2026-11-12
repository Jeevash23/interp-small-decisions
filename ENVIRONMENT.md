# Environment

The studies ran with CPython 3.12.14 and these packages (the GPU runs used PyTorch's CUDA 13.0 build):

| Package | Version | Used for |
|---|---|---|
| numpy | 2.5.3 | everything |
| torch | 2.14.1 | the model runs; reading the saved `.pt` records in the number build and check |
| transformers | 5.19.0 | loading the models |
| tokenizers | 0.23.3 | tokenisers |
| safetensors | 0.8.0 | loading weights |
| huggingface-hub | 2.1.1 | the Hugging Face cache and `tools/fetch_models.py` |
| accelerate | 1.15.0 | loading Gemma-4-E2B onto the GPU layer by layer (`gemma_load.py`, D1's `xm3_common.py`) |
| pyarrow | 25.0.1 | reading Parquet files in D1's helper module (not used by this paper's runs) |

`pyproject.toml` pins them and `uv.lock` records the full environment, so

```sh
uv sync --frozen          # or: make setup
```

recreates it in `.venv`. On Linux x86-64 that downloads 56 packages, 3.11 GB of wheels, most of it PyTorch with its
CUDA libraries.

## Only the number check

`make check` needs only NumPy and PyTorch. A CPU build of PyTorch is enough and much smaller:

```sh
uv venv --python 3.12 .venv-check
uv pip install --python .venv-check/bin/python numpy==2.5.3
uv pip install --python .venv-check/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
make check PYTHON=.venv-check/bin/python
```

The scripts use only the standard library besides these two.

## Hardware

The model runs used one 6 GB NVIDIA GPU (compute capability 8.9) with CUDA 13.0. Each study's loader states the
precision of each model (for example `code/common.py`, `code/d5common.py`, `code/x1common.py`). The number build and check run on a CPU in about 3 GB of memory.
