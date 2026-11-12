# Small decisions, mostly separate parts: code, frozen protocols and saved outputs. `make help` lists the targets.
# PYTHON defaults to the environment that `make setup` creates; pass PYTHON=/path/to/python to use another one.
PYTHON ?= .venv/bin/python
PY := $(if $(findstring /,$(PYTHON)),$(abspath $(PYTHON)),$(PYTHON))
export PYTHONDONTWRITEBYTECODE := 1
.NOTPARALLEL:

.PHONY: help setup python-check unpack check-freezes numbers compare check tables paper models

help:
	@echo "make setup          create .venv with the pinned environment (uv sync; see ENVIRONMENT.md)"
	@echo "make check          CPU only, no network, about 10 minutes: unpack the saved outputs (checking every SHA-256),"
	@echo "                    check the four freeze records, regenerate every number and figure value of the paper"
	@echo "                    from the saved outputs, compare them with the shipped values, and run the paper's"
	@echo "                    independent check (1,276 checks)"
	@echo "make numbers        regenerate paper/numbers.tex and the figure data only"
	@echo "make tables         copy the source papers' table layouts into paper/tables/"
	@echo "make paper          build paper/build/main.pdf (needs a LaTeX installation with latexmk)"
	@echo "make models         download the model checkpoints at the recorded revisions (about 28 GB, network)"

setup:
	uv sync --frozen

python-check:
	@$(PY) -c "import numpy, torch" 2>/dev/null || { echo "No usable Python at $(PYTHON) (it needs numpy and torch): run make setup, or set PYTHON=..."; exit 1; }

unpack: python-check
	$(PY) tools/unpack.py
	$(PY) tools/make_links.py
	$(PY) tools/seal_frozen.py

check-freezes: unpack
	$(PY) tools/check_frozen_tables.py

numbers: unpack
	$(PY) -B paper/scripts/make_numbers.py

compare:
	$(PY) tools/compare_numbers.py

check: check-freezes numbers compare
	$(PY) -B paper/scripts/check_numbers.py

tables: unpack
	$(PY) -B paper/scripts/extract_tables.py

paper:
	cd paper && mkdir -p build && latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=build main.tex

models: python-check
	$(PY) tools/fetch_models.py --all
