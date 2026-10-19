# ---------------------------------------------
# MLnode Python Backend - Setup Script (Windows PowerShell / PowerShell 7)
#
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
#
# Mirrors setup.sh. Optional environment variables:
#   PYTHON           interpreter to use (default: "py -3", then "python")
#   TORCH_INDEX_URL  PyTorch wheel index, e.g. https://download.pytorch.org/whl/cu128
# ---------------------------------------------

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$PythonMin = [version]"3.11"
$VenvDir = ".venv"

function Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Ok($msg)   { Write-Host "[OK]   $msg" -ForegroundColor Green }
function Warn($msg) { Write-Host "[WARN] $msg" -ForegroundColor Yellow }
function Fail($msg) { Write-Host "[FAIL] $msg" -ForegroundColor Red; exit 1 }

# Run a native command and stop on a non-zero exit code
# ($ErrorActionPreference does not apply to native executables).
function Invoke-Checked {
    param([string]$Exe, [string[]]$Arguments)
    & $Exe @Arguments
    if ($LASTEXITCODE -ne 0) { Fail "Command failed ($LASTEXITCODE): $Exe $($Arguments -join ' ')" }
}

Write-Host ""
Write-Host "+======================================+" -ForegroundColor Cyan
Write-Host "|     MLnode Backend Setup v1.0        |" -ForegroundColor Cyan
Write-Host "+======================================+" -ForegroundColor Cyan
Write-Host ""

# -- Find Python --
Info "Checking Python version..."
$candidates = @()
if ($env:PYTHON) { $candidates += , @($env:PYTHON) }
$candidates += , @("py", "-3")
$candidates += , @("python")

$PyExe = $null
$PyArgs = @()
foreach ($c in $candidates) {
    $exe = $c[0]
    $rest = @($c | Select-Object -Skip 1)
    if (Get-Command $exe -ErrorAction SilentlyContinue) {
        & $exe @rest -c "import sys" 2>$null
        if ($LASTEXITCODE -eq 0) { $PyExe = $exe; $PyArgs = $rest; break }
    }
}
if (-not $PyExe) { Fail "Python not found. Install Python >= $PythonMin from python.org (or set `$env:PYTHON)." }

$PyVersion = [version](& $PyExe @PyArgs -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if ($PyVersion -lt $PythonMin) { Fail "Python $PyVersion found, but >= $PythonMin is required." }
Ok "Python $PyVersion"

# -- Create virtual environment --
if (-not (Test-Path $VenvDir)) {
    Info "Creating virtual environment in $VenvDir..."
    Invoke-Checked $PyExe ($PyArgs + @("-m", "venv", $VenvDir))
    Ok "Virtual environment created"
} else {
    Ok "Virtual environment already exists"
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) { $VenvPython = Join-Path $VenvDir "bin/python" }  # pwsh on Linux/macOS
if (-not (Test-Path $VenvPython)) { Fail "No interpreter found in $VenvDir - delete it and re-run." }
Ok "Using $VenvPython"

# -- Upgrade pip --
Info "Upgrading pip..."
Invoke-Checked $VenvPython @("-m", "pip", "install", "--upgrade", "pip", "--quiet")
Ok "pip upgraded"

# -- Install PyTorch --
# PyPI has no torch wheels for Python 3.15 yet; they are on download.pytorch.org.
$TorchIndex = $env:TORCH_INDEX_URL
if (-not $TorchIndex -and $PyVersion -ge [version]"3.15") {
    $TorchIndex = "https://download.pytorch.org/whl/cpu"
    Warn "Python ${PyVersion}: installing CPU torch from $TorchIndex (set TORCH_INDEX_URL for CUDA)"
}
if ($TorchIndex) {
    Info "Installing torch from $TorchIndex..."
    Invoke-Checked $VenvPython @("-m", "pip", "install", "torch>=2.4", "--index-url", $TorchIndex, "--quiet")
    Ok "torch installed"
}

# -- Install package with dev dependencies --
Info "Installing mlnode with dev dependencies..."
Invoke-Checked $VenvPython @("-m", "pip", "install", "-e", ".[dev]", "--quiet")
Ok "mlnode installed"

# -- Verify imports --
Info "Verifying installation..."
$verify = @'
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
'@
Invoke-Checked $VenvPython @("-c", $verify)
Ok "All imports verified"

# -- Run tests --
Info "Running test suite..."
Write-Host ""
& $VenvPython -m pytest tests/ -q --tb=short
if ($LASTEXITCODE -eq 0) { Ok "Tests passed" } else { Warn "Some tests failed (exit code $LASTEXITCODE)" }
Write-Host ""

# -- Quick smoke test --
Info "Running smoke test..."
$smoke = @'
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
'@
Invoke-Checked $VenvPython @("-c", $smoke)
Ok "Smoke test passed"

# -- Done --
Write-Host ""
Write-Host "+======================================+" -ForegroundColor Green
Write-Host "|     Setup complete!                  |" -ForegroundColor Green
Write-Host "+======================================+" -ForegroundColor Green
Write-Host ""
Write-Host "  Usage:"
Write-Host "    .\$VenvDir\Scripts\Activate.ps1"
Write-Host "    python -c `"from mlnode import MLnodeModel`""
Write-Host "    pytest tests/ -v"
Write-Host ""
