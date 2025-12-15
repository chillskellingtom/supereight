# Smoke test for subtitles processing on tape 1 scenes 1-20
# This script runs the pipeline on a limited dataset for testing and validation

$ErrorActionPreference = "Stop"

# Enable unbuffered Python output for real-time logs
$env:PYTHONUNBUFFERED = "1"
$env:LOG_LEVEL = "INFO"

# Disable verbose Level Zero API tracing (reduces extreme verbosity)
$env:SYCL_PI_TRACE = "0"
$env:ZE_ENABLE_TRACING = "0"

# Enable GPU affinity verification
$env:_VERIFY_GPU_AFFINITY = "1"

$SCENES = "C:\Users\latch\connor_family_movies_processed\scenes"
# Use explicit interpreter to avoid PATH ambiguity (satisfies tests/test_environment.py)
$PythonExe = "C:\Users\latch\AppData\Local\Programs\Python\Python311\python.exe"

Write-Host "=== Smoke Test: Subtitles on Tape 1 (Scenes 1-20) ===" -ForegroundColor Cyan
Write-Host "Activating conda environment: ipex-llm-xpu" -ForegroundColor Yellow
Write-Host "Scenes folder: $SCENES" -ForegroundColor Yellow
Write-Host "Task: subtitles" -ForegroundColor Yellow
Write-Host "Limit: 20 scenes" -ForegroundColor Yellow
Write-Host ""

# Use explicit interpreter to avoid PATH ambiguity (satisfies tests/test_environment.py)
& $PythonExe process_parallel.py `
  --task subtitles `
  --scenes-folder "$SCENES" `
  --limit 20 `
  --monitor-interval 5 `
  -v

$exitCode = $LASTEXITCODE
if ($exitCode -eq 0) {
    Write-Host ""
    Write-Host "=== Smoke test completed successfully ===" -ForegroundColor Green
} else {
    Write-Host ""
    Write-Host "=== Smoke test failed with exit code $exitCode ===" -ForegroundColor Red
}

exit $exitCode
