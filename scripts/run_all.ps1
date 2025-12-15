# One-touch wrapper: set env, ensure models cached, run full pipeline
$ErrorActionPreference = "Stop"

$RepoRoot = Split-Path $PSScriptRoot -Parent
$SetvarsPath = "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
$ModelDir = Join-Path $RepoRoot "models"
$CacheDir = Join-Path $env:USERPROFILE ".cache\connor_family_models"
$PipelineScript = Join-Path $PSScriptRoot "run_full_pipeline.ps1"
$CondaEnv = "ipex-llm-xpu"

Write-Host "=== Connor pipeline bootstrap ===" -ForegroundColor Cyan

# 1) oneAPI environment
if (Test-Path $SetvarsPath) {
    Write-Host "Initializing oneAPI env via: $SetvarsPath" -ForegroundColor Yellow
    cmd /c "`"$SetvarsPath`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$' -and $_ -notmatch '^::') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
    Write-Host "✓ oneAPI environment set." -ForegroundColor Green
}
else {
    Write-Warning "oneAPI setvars.bat not found at $SetvarsPath; skipping GPU env init."
}

# 2) GPU-related env vars
$env:SYCL_CACHE_PERSISTENT = "1"
$env:SYCL_CACHE_DIR = "$env:USERPROFILE\.cache\intel_gpu_cache"
Write-Host "SYCL_CACHE_PERSISTENT=$($env:SYCL_CACHE_PERSISTENT)" -ForegroundColor White
Write-Host "SYCL_CACHE_DIR=$($env:SYCL_CACHE_DIR)" -ForegroundColor White

# 3) Ensure models cached
New-Item -ItemType Directory -Force -Path $ModelDir | Out-Null
New-Item -ItemType Directory -Force -Path $CacheDir | Out-Null

$Models = @(
    "scrfd_2.5g_bnkps.onnx",
    "face_recognition_sface_2021dec.onnx",
    "arcface_r100.onnx"
)

foreach ($m in $Models) {
    $src = Join-Path $ModelDir $m
    if (-not (Test-Path $src)) {
        $cacheSrc = Join-Path $CacheDir $m
        if (Test-Path $cacheSrc) {
            $src = $cacheSrc
        }
    }
    if (-not (Test-Path $src)) {
        Write-Warning "Model missing: $m (not found in models/ or cache). Please download to $ModelDir."
        continue
    }
    $dest = Join-Path $CacheDir $m
    if ($src -eq $dest) {
        Write-Host "✓ Cached $m (already in cache)" -ForegroundColor Green
    } else {
        Copy-Item -Force $src $dest
        Write-Host "✓ Cached $m" -ForegroundColor Green
    }
}

# 4) Select Python (prefer conda env intel-ipex-llm if available)
$PythonExe = "python"
$conda = Get-Command conda -ErrorAction SilentlyContinue
if ($conda) {
    try {
        $envs = conda env list
        if ($envs -match "^\s*$CondaEnv\s") {
            $PythonExe = "conda run -n $CondaEnv python"
            Write-Host "Using conda env: $CondaEnv" -ForegroundColor Yellow
        }
    } catch {}
}
$env:PYTHON_EXE = $PythonExe

# 5) Preflight: check torch XPU/CUDA/DML availability
Write-Host "Running preflight GPU check..." -ForegroundColor Yellow
$preflight = @"
import sys
ok = False
try:
    import torch
    has_xpu = hasattr(torch, 'xpu') and torch.xpu.is_available()
    has_cuda = torch.cuda.is_available() if hasattr(torch, 'cuda') else False
    print(f'torch {torch.__version__} | xpu={has_xpu} | cuda={has_cuda}')
    ok = ok or has_xpu or has_cuda
except Exception as e:
    print('torch not working:', e, file=sys.stderr)
try:
    import torch_directml as tdm
    dc = tdm.device_count()
    print(f'directml devices={dc}')
    ok = ok or dc > 0
except Exception as e:
    print('directml not working:', e, file=sys.stderr)
sys.exit(0 if ok else 1)
"@
cmd /c "$PythonExe -c `"$($preflight.Replace('`n',';'))`""
if ($LASTEXITCODE -ne 0) {
    Write-Warning "GPU backend not available (torch XPU/CUDA/DirectML missing). Install intel-extension-for-pytorch or torch-directml in $CondaEnv."
}

# 6) Run pipeline
if (-not (Test-Path $PipelineScript)) {
    throw "Pipeline script not found: $PipelineScript"
}
Write-Host "Running pipeline: $PipelineScript" -ForegroundColor Cyan
& $PipelineScript
exit $LASTEXITCODE
