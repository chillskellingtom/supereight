# Setup script to initialize oneAPI environment and run VRT with Intel GPU support
# This script sets up the oneAPI environment and ensures Intel GPUs are available

param(
    [string]$Command = "",
    [string[]]$Arguments = @()
)

# Initialize oneAPI environment
Write-Host "Initializing oneAPI environment..." -ForegroundColor Yellow
# Properly import env vars from setvars.bat into this PowerShell session
if (Test-Path "C:\\Program Files (x86)\\Intel\\oneAPI\\setvars.bat") {
    $proc = Start-Process -FilePath "cmd.exe" -ArgumentList "/c","\"C:\\Program Files (x86)\\Intel\\oneAPI\\setvars.bat\" && set" -NoNewWindow -PassThru -Wait -RedirectStandardOutput "$env:TEMP\\oneapi_env.txt" -RedirectStandardError "$env:TEMP\\oneapi_env.err"
    if ($proc.ExitCode -eq 0 -and (Test-Path "$env:TEMP\\oneapi_env.txt")) {
        Get-Content "$env:TEMP\\oneapi_env.txt" | ForEach-Object {
            if ($_ -match "^
            ") { return }
            $parts = $_.Split('=',2)
            if ($parts.Length -eq 2) {
                $name = $parts[0]
                $value = $parts[1]
                if ($name -and $value) { $env:$name = $value }
            }
        }
        Remove-Item "$env:TEMP\\oneapi_env.txt" -ErrorAction SilentlyContinue
        Remove-Item "$env:TEMP\\oneapi_env.err" -ErrorAction SilentlyContinue
        Write-Host "oneAPI environment imported into PowerShell." -ForegroundColor Green
    } else {
        Write-Warning "setvars.bat execution failed; Intel GPU may not initialize correctly."
    }
} else {
    Write-Warning "oneAPI setvars.bat not found; please install Intel oneAPI Base Toolkit."
}

# Set Intel GPU environment variables
$env:ZE_AFFINITY_MASK = "0,1"  # Use both Intel Arc A770 GPUs
$env:SYCL_CACHE_PERSISTENT = "1"
$env:SYCL_CACHE_DIR = "$env:USERPROFILE\.cache\intel_gpu_cache"

Write-Host "Intel GPU environment configured:" -ForegroundColor Green
Write-Host "  ZE_AFFINITY_MASK: $env:ZE_AFFINITY_MASK" -ForegroundColor White
Write-Host "  SYCL_CACHE_PERSISTENT: $env:SYCL_CACHE_PERSISTENT" -ForegroundColor White

# Test IPEX availability
Write-Host "`nTesting Intel GPU availability..." -ForegroundColor Yellow
$testScript = @"
import intel_extension_for_pytorch as ipex
import sys

try:
    print(f'IPEX version: {ipex.__version__}')
    if ipex.xpu.is_available():
        print(f'✓ Intel GPU (XPU) available: {ipex.xpu.device_count()} device(s)')
        for i in range(ipex.xpu.device_count()):
            print(f'  Device {i}: {ipex.xpu.get_device_name(i)}')
        sys.exit(0)
    else:
        print('✗ Intel GPU (XPU) not available')
        sys.exit(1)
except Exception as e:
    print(f'✗ Error: {e}')
    sys.exit(1)
"@

$testResult = python -c $testScript 2>&1
Write-Host $testResult

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ Intel GPU is ready!" -ForegroundColor Green
    
    # If a command was provided, run it
    if ($Command) {
        Write-Host "`nRunning: $Command $Arguments" -ForegroundColor Cyan
        & $Command $Arguments
    }
} else {
    Write-Host "`n✗ Intel GPU not available. Check:" -ForegroundColor Red
    Write-Host "  1. Intel Arc GPU drivers are installed" -ForegroundColor Yellow
    Write-Host "  2. oneAPI Base Toolkit is installed" -ForegroundColor Yellow
    Write-Host "  3. IPEX is installed: pip install intel-extension-for-pytorch --extra-index-url https://pytorch-extension.intel.com/release-whl/stable/xpu/us/" -ForegroundColor Yellow
}

