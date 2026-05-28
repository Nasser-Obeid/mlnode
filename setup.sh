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
info "Checking Python version..."
if ! command -v python3 &> /dev/null; then
    fail "python3 not found. Install Python >= $PYTHON_MIN"
fi

PY_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
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
    python3 -m venv "$VENV_DIR"
    ok "Virtual environment created"
else
    ok "Virtual environment already exists"
fi

# ── Activate venv ──
info "Activating virtual environment..."
source "$VENV_DIR/bin/activate"
ok "Activated ($VENV_DIR)"

# ── Upgrade pip ──
info "Upgrading pip..."
pip install --upgrade pip --quiet
ok "pip $(pip --version | awk '{print $2}')"

# ── Install package with dev dependencies ──
info "Installing mlnode with dev dependencies..."
pip install -e ".[dev]" --quiet
ok "mlnode installed"

# ── Verify imports ──
info "Verifying installation..."
python3 -c "
from mlnode import MLnodeModel, BlockRegistry, ValidationResult
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
python3 -m pytest tests/ -v --tb=short 2>&1 | tail -20
echo ""

RESULT=$(python3 -m pytest tests/ --tb=no -q 2>&1 | tail -1)
if echo "$RESULT" | grep -q "passed"; then
    ok "Tests: $RESULT"
else
    warn "Tests: $RESULT"
fi

# ── Quick smoke test ──
info "Running smoke test..."
python3 -c "
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
echo "    source $VENV_DIR/bin/activate"
echo "    python -c \"from mlnode import MLnodeModel\""
echo "    pytest tests/ -v"
echo ""
