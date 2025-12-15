# Test script to verify VRT model loading and GPU parallelization
# This script runs comprehensive tests to ensure models load correctly on both GPUs

Write-Host "`n=== VRT GPU Model Loading Test ===" -ForegroundColor Cyan
Write-Host "Testing model loading and parallelization across Intel Arc GPUs`n" -ForegroundColor White

# Initialize oneAPI environment
$oneapiPath = "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
if (Test-Path $oneapiPath) {
    Write-Host "Initializing oneAPI environment..." -ForegroundColor Yellow
    cmd /c "`"$oneapiPath`" >nul 2>&1 && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$' -and $_ -notmatch '^::') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
    Write-Host "✓ oneAPI environment initialized`n" -ForegroundColor Green
} else {
    Write-Host "⚠ oneAPI not found at: $oneapiPath" -ForegroundColor Yellow
    Write-Host "Continuing without oneAPI initialization...`n" -ForegroundColor Yellow
}

# Set Intel GPU environment
$env:ZE_AFFINITY_MASK = "0,1"  # Use both GPUs
$env:SYCL_CACHE_PERSISTENT = "1"
$env:SYCL_CACHE_DIR = "$env:USERPROFILE\.cache\intel_gpu_cache"

Write-Host "GPU Environment:" -ForegroundColor Cyan
Write-Host "  ZE_AFFINITY_MASK: $env:ZE_AFFINITY_MASK" -ForegroundColor White
Write-Host "  SYCL_CACHE_PERSISTENT: $env:SYCL_CACHE_PERSISTENT" -ForegroundColor White
Write-Host "  SYCL_CACHE_DIR: $env:SYCL_CACHE_DIR`n" -ForegroundColor White

# Run the test script
Write-Host "Running GPU model loading test...`n" -ForegroundColor Yellow
$TestPath = Join-Path $PSScriptRoot "..\\tests\\manual\\test_gpu_model_loading.py"
python $TestPath

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n✓ GPU model loading test completed successfully!" -ForegroundColor Green
} else {
    Write-Host "`n✗ GPU model loading test failed (exit code: $LASTEXITCODE)" -ForegroundColor Red
    exit $LASTEXITCODE
}

