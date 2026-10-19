#!/usr/bin/env bash
set -euo pipefail

# ─────────────────────────────────────────────
# MLnode Python Backend — Setup Script
# ─────────────────────────────────────────────

PYTHON_MIN="3.11"
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC} $1"; }
ok()    { echo -e "${GREEN}[OK]${NC}   $1"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $1"; }
fail()  { echo -e "${RED}[FAIL]${NC} $1"; exit 1; }

echo ""
echo -e "${CYAN}╔══════════════════════════════════════╗${NC}"
echo -e "${CYAN}║     MLnode Backend Setup v1.0        ║${NC}"
echo -e "${CYAN}╚══════════════════════════════════════╝${NC}"
echo ""

# ── Check Python version ──
# Works on Linux/macOS and Git Bash on Windows (where the interpreter is usually "python").
info "Checking Python version..."
PYTHON="${PYTHON:-}"
if [ -z "$PYTHON" ]; then
    for candidate in python3 python; do
        if command -v "$candidate" &> /dev/null && "$candidate" -c "import sys" &> /dev/null; then
            PYTHON="$candidate"
            break
        fi
    done
fi
if [ -z "$PYTHON" ]; then
    fail "Python not found. Install Python >= $PYTHON_MIN (or set PYTHON=/path/to/python)"
fi

PY_VERSION=$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VERSION" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VERSION" | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 11 ]); then
    fail "Python $PY_VERSION found, but >= $PYTHON_MIN is required."
fi
ok "Python $PY_VERSION"

# ── Create virtual environment ──
VENV_DIR=".venv"
if [ ! -d "$VENV_DIR" ]; then
    info "Creating virtual environment in $VENV_DIR..."
    "$PYTHON" -m venv "$VENV_DIR"
    ok "Virtual environment created"
else
    ok "Virtual environment already exists"
fi

# ── Activate venv ──
info "Activating virtual environment..."
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    source "$VENV_DIR/Scripts/activate"   # Windows (Git Bash)
else
    source "$VENV_DIR/bin/activate"
fi
ok "Activated ($VENV_DIR)"

# ── Upgrade pip ──
info "Upgrading pip..."
python -m pip install --upgrade pip --quiet
ok "pip $(python -m pip --version | awk '{print $2}')"

# ── Install PyTorch ──
# PyPI has no torch wheels for Python 3.15 yet; they are on download.pytorch.org.
# Set TORCH_INDEX_URL to choose a build, e.g. https://download.pytorch.org/whl/cu128
if [ -z "${TORCH_INDEX_URL:-}" ] && [ "$PY_MINOR" -ge 15 ]; then
    TORCH_INDEX_URL="https://download.pytorch.org/whl/cpu"
    warn "Python $PY_VERSION: installing CPU torch from $TORCH_INDEX_URL (set TORCH_INDEX_URL for CUDA)"
fi
if [ -n "${TORCH_INDEX_URL:-}" ]; then
    info "Installing torch from $TORCH_INDEX_URL..."
    python -m pip install "torch>=2.4" --index-url "$TORCH_INDEX_URL" --quiet
    ok "torch installed"
fi

# ── Install package with dev dependencies ──
info "Installing mlnode with dev dependencies..."
python -m pip install -e ".[dev]" --quiet
ok "mlnode installed"

# ── Verify imports ──
info "Verifying installation..."
python -c "
from mlnode import MLnodeModel, BlockRegistry, LayerRegistry, ValidationResult
from mlnode.errors import MLnodeError, ValidationError, CycleDetectedError
import torch
print(f'  torch {torch.__version__}')
import transformers
print(f'  transformers {transformers.__version__}')
import networkx
print(f'  networkx {networkx.__version__}')
import pydantic
print(f'  pydantic {pydantic.__version__}')
"
ok "All imports verified"

# ── Run tests ──
info "Running test suite..."
echo ""
python -m pytest tests/ -v --tb=short 2>&1 | tail -20
echo ""

RESULT=$(python -m pytest tests/ --tb=no -q 2>&1 | tail -1)
if echo "$RESULT" | grep -q "passed"; then
    ok "Tests: $RESULT"
else
    warn "Tests: $RESULT"
fi

# ── Quick smoke test ──
info "Running smoke test..."
python -c "
from mlnode import MLnodeModel
import torch

model = MLnodeModel.from_dict({
    'mlnode_schema_version': '1.0',
    'input_shape': ['B', 3, 224, 224],
    'output_node': 'relu',
    'nodes': [
        {'id': 'input', 'type': 'Input', 'params': {}},
        {'id': 'conv', 'type': 'Conv2d', 'params': {'in_channels': 3, 'out_channels': 64, 'kernel_size': 3, 'padding': 1}},
        {'id': 'relu', 'type': 'ReLU', 'params': {}},
    ],
    'edges': [
        {'source': 'input', 'target': 'conv'},
        {'source': 'conv', 'target': 'relu'},
    ]
})
r = model.validate()
assert r.valid, f'Validation failed: {r.errors}'
module = model.build()
out = module(torch.randn(2, 3, 224, 224))
assert out.shape == (2, 64, 224, 224), f'Wrong shape: {out.shape}'
print(f'  Input:  (2, 3, 224, 224)')
print(f'  Output: {tuple(out.shape)}')
print(f'  Params: {sum(p.numel() for p in module.parameters()):,}')
"
ok "Smoke test passed"

# ── Done ──
echo ""
echo -e "${GREEN}╔══════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Setup complete!                  ║${NC}"
echo -e "${GREEN}╚══════════════════════════════════════╝${NC}"
echo ""
echo "  Usage:"
if [ -f "$VENV_DIR/Scripts/activate" ]; then
    echo "    source $VENV_DIR/Scripts/activate"
else
    echo "    source $VENV_DIR/bin/activate"
fi
echo "    python -c \"from mlnode import MLnodeModel\""
echo "    pytest tests/ -v"
echo ""
