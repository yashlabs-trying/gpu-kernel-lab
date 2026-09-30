param(
    [ValidateSet("cu128", "cu130")]
    [string]$CudaVariant = "cu128",
    [string]$VenvPath = ""
)

$ErrorActionPreference = "Stop"
$LlamaDir = Split-Path -Parent $PSScriptRoot
$RepoDir = Resolve-Path (Join-Path $LlamaDir "..\..")
if (-not $VenvPath) { $VenvPath = Join-Path $RepoDir ".venv-llama" }

if ($CudaVariant -eq "cu128") {
    $TorchVersion = "2.9.0"
    $LockFile = Join-Path $LlamaDir "requirements-cu128.lock"
} else {
    $TorchVersion = "2.13.0"
    $LockFile = Join-Path $LlamaDir "requirements-cu130.lock"
}

python -m venv $VenvPath
$Python = Join-Path $VenvPath "Scripts\python.exe"
& $Python -m pip install --upgrade pip wheel setuptools
& $Python -m pip install "torch==$TorchVersion" --index-url "https://download.pytorch.org/whl/$CudaVariant"
& $Python -m pip install -r $LockFile
& $Python -m pip install --no-deps -e $RepoDir
& $Python -m kernellab doctor

Write-Host "Environment ready: $VenvPath"
