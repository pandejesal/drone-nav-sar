#!/usr/bin/env bash
# Sprint 13 pre-commit: black, isort, flake8, mypy, pytest -q (SAR-only repo).
set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== black (check) ==="
black --check src/ scripts/ tests/ || { echo ">> run: black src/ scripts/ tests/"; exit 1; }

echo "=== isort (check) ==="
isort --check-only --profile black src/ scripts/ tests/ || { echo ">> run: isort --profile black src/ scripts/ tests/"; exit 1; }

echo "=== flake8 ==="
flake8 src/ scripts/ tests/ --max-line-length=100

echo "=== mypy (strict, src only) ==="
mypy --strict src/

echo "=== pytest -q ==="
pytest -q --tb=short

echo "ALL PRE-COMMIT CHECKS PASSED"
