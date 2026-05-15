#!/bin/bash
# Idempotent install of cell_lifetime + its dependencies into the active env.
# Safe to re-run; pip handles "already installed" cases.

set -euo pipefail

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_DIR="$(dirname "$THIS_DIR")"
WORKSPACE_ROOT="$(dirname "$PKG_DIR")"

echo "=== install_env.sh ==="
echo "  cell_lifetime dir: $PKG_DIR"
echo "  workspace root:    $WORKSPACE_ROOT"
echo "  active python:     $(which python)"
echo "  active python --version: $(python --version)"

# Required runtime deps (also captured in pyproject.toml; this catches
# environments without the extras flag and gives explicit failure surface)
python -m pip install --quiet \
    xgboost \
    interpret \
    scikit-survival \
    optuna \
    shap \
    scipy \
    pyyaml \
    pytest \
    polars \
    pandas \
    numpy \
    scikit-learn

# cell_classifier is a read-only consumer — install editable
python -m pip install --quiet -e "$WORKSPACE_ROOT/cell_classifier"

# cell_lifetime itself (with all extras since we need all three models)
python -m pip install --quiet -e "$PKG_DIR[xgb,ebm,survival]"

echo "=== install_env.sh OK ==="
python -c "import cell_lifetime, cell_classifier, xgboost, interpret, sksurv; print('cell_lifetime', cell_lifetime.__version__)"
