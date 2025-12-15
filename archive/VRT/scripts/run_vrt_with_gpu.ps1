# Run VRT with Intel GPU support
# This script initializes oneAPI and runs VRT processing

param(
    [string]$Task = "videosr_reds_16frames",
    # NOTE: PowerShell has an automatic variable $input (case-insensitive),
    # so using a parameter named $Input can behave unexpectedly. Use InputPath instead.
    [string]$InputPath = "",
    [string]$OutputPath = "",
    [string]$ComparisonPath = ""
)

if (-not $InputPath -or -not $OutputPath) {
    Write-Host "Usage: .\run_vrt_with_gpu.ps1 -Task <task> -InputPath <input.mp4> -OutputPath <output.mp4> [-ComparisonPath <comparison.mp4>]" -ForegroundColor Yellow
    Write-Host "`nCurrent parameters:" -ForegroundColor Yellow
    Write-Host "  Task: $Task" -ForegroundColor White
    Write-Host "  InputPath: $InputPath" -ForegroundColor White
    Write-Host "  OutputPath: $OutputPath" -ForegroundColor White
    Write-Host "  ComparisonPath: $ComparisonPath" -ForegroundColor White
    exit 1
}

Write-Host "=== Initializing Intel GPU Environment ===" -ForegroundColor Cyan

# Initialize oneAPI environment
$oneapiPath = "C:\Program Files (x86)\Intel\oneAPI\setvars.bat"
if (Test-Path $oneapiPath) {
    Write-Host "Initializing oneAPI..." -ForegroundColor Yellow
    cmd /c "`"$oneapiPath`" && set" | ForEach-Object {
        if ($_ -match '^([^=]+)=(.*)$' -and $_ -notmatch '^::') {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], 'Process')
        }
    }
    Write-Host "✓ oneAPI environment initialized" -ForegroundColor Green
} else {
    Write-Host "⚠ oneAPI not found at: $oneapiPath" -ForegroundColor Yellow
}

# Set Intel GPU environment variables
$env:ZE_AFFINITY_MASK = "0,1"  # Use both Intel Arc A770 GPUs
$env:SYCL_CACHE_PERSISTENT = "1"
$env:SYCL_CACHE_DIR = "$env:USERPROFILE\.cache\intel_gpu_cache"

Write-Host "`n=== Testing Intel GPU ===" -ForegroundColor Cyan
$testResult = cmd /c "`"$oneapiPath`" >nul 2>&1 && python -c `"try:
    import intel_extension_for_pytorch as ipex
    if ipex.xpu.is_available():
        print('✓ Intel GPU available:', ipex.xpu.device_count(), 'device(s)')
        for i in range(ipex.xpu.device_count()):
            print(f'  Device {i}: {ipex.xpu.get_device_name(i)}')
        exit(0)
    else:
        print('✗ Intel GPU not available')
        exit(1)
except Exception as e:
    print(f'✗ Error: {e}')
    exit(1)
`""

Write-Host $testResult

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== Running VRT Processing ===" -ForegroundColor Green
    
    $vrtScript = Join-Path $PSScriptRoot "vrt_enhance.py"
    $cmdArgs = @(
        $vrtScript,
        "--task", $Task,
        "--input", $InputPath,
        "--output", $OutputPath,
        "-v"
    )
    
    if ($ComparisonPath) {
        $cmdArgs += "--comparison", $ComparisonPath
    }
    
    # Run with oneAPI environment
    cmd /c "`"$oneapiPath`" >nul 2>&1 && python $($cmdArgs -join ' ')"
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ VRT processing complete!" -ForegroundColor Green
    } else {
        Write-Host "`n✗ VRT processing failed" -ForegroundColor Red
        exit 1
    }
} else {
    Write-Host "`n⚠ Intel GPU not available - will run on CPU (slow)" -ForegroundColor Yellow
    Write-Host "Continuing with CPU processing..." -ForegroundColor Yellow
    
    $vrtScript = Join-Path $PSScriptRoot "vrt_enhance.py"
    $cmdArgs = @(
        $vrtScript,
        "--task", $Task,
        "--input", $InputPath,
        "--output", $OutputPath,
        "-v"
    )
    if ($ComparisonPath) {
        $cmdArgs += "--comparison", $ComparisonPath
    }
    python $($cmdArgs -join ' ')
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`n✓ VRT processing complete (CPU)" -ForegroundColor Green
    } else {
        Write-Host "`n✗ VRT processing failed" -ForegroundColor Red
        exit 1
    }
}

