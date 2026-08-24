REPO_ROOT := $(shell dirname $(CURDIR))

.PHONY: setup test test-tutte benchmark-family-pipeline benchmark-family visualize clean

# ===== Setup =====

setup:
	uv sync --extra dev

setup-viz:
	uv sync --extra dev --extra viz

# ===== Tests =====

test:
	uv run pytest tests/ -v

test-tutte:
	uv run pytest tests/tutte/ -v

# ===== Benchmark =====

benchmark-family-pipeline:
	PYTHONPATH=$(REPO_ROOT) PYTHONUNBUFFERED=1 uv run python -m pytest tests/test_benchmark_family_recognition.py::test_benchmark_family_pipeline -v -s 2>&1 | tee benchmark_family_pipeline_output.log

benchmark-family:
	PYTHONPATH=$(REPO_ROOT) PYTHONUNBUFFERED=1 FAMILY=$(FAMILY) MIN_E=$(MIN_E) MAX_E=$(MAX_E) uv run python -m pytest tests/test_benchmark_family_recognition.py::test_benchmark_family -v -s 2>&1 | tee benchmark_family_output.log

# ===== Visualizer =====

visualize:
	PYTHONPATH=$(REPO_ROOT) uv run python scripts/visualize_tutte.py

# ===== Clean =====

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +