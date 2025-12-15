# Full pipeline for processing family movies
# Automatically targets the local inputs folder next to the repo.

$RepoRoot = Split-Path $PSScriptRoot -Parent
$InputFolder = Join-Path $RepoRoot "inputs"
$OutputFolder = "C:\Users\latch\connor_family_movies_processed"
$ScenesFolder = "$OutputFolder\scenes"
$ScriptDir = $PSScriptRoot
$DetectScenesPy = Join-Path $ScriptDir "detect_scenes.py"
$ProcessParallelPy = Join-Path $ScriptDir "..\process_parallel.py"
# Allow run_all.ps1 to override Python selection (e.g., conda run)
$PythonExe = if ($env:PYTHON_EXE) { $env:PYTHON_EXE } else { "python" }

Write-Host "=== Step 1: Scene Detection ===" -ForegroundColor Green
cmd /c "$PythonExe $DetectScenesPy --input `"$InputFolder`" --output `"$OutputFolder`" -v"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Scene detection failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Step 2: Face Export (with auto-cluster) ===" -ForegroundColor Green
cmd /c "$PythonExe $ProcessParallelPy `
    --task faces_export `
    --scenes-folder `"$ScenesFolder`" `
    --frame-stride 12 `
    --min-face-score 0.6 `
    --face-crop-size 256 `
    --face-max-per-track 3 `
    --auto-cluster `
    --cluster-method hierarchical `
    --cluster-similarity 0.90 `
    --cluster-min-samples 5 `
    --cluster-quality-threshold 0.3 `
    --cluster-n-passes 3 `
    -v"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Face export/clustering failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Step 3: Video Enhancement ===" -ForegroundColor Green
cmd /c "$PythonExe $ProcessParallelPy `
    --task video_enhance `
    --scenes-folder `"$ScenesFolder`" `
    --enhance-scale 2.0 `
    --enhance-denoise 1.0 `
    --enhance-crf 16 `
    --video-codec auto `
    -v"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Video enhancement failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Step 4: Audio Enhancement ===" -ForegroundColor Green
cmd /c "$PythonExe $ProcessParallelPy `
    --task audio_enhance `
    --scenes-folder `"$ScenesFolder`" `
    --audio-denoise -25 `
    --audio-bitrate 192k `
    -v"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Audio enhancement failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Step 5: Subtitles ===" -ForegroundColor Green
cmd /c "$PythonExe $ProcessParallelPy `
    --task subtitles `
    --scenes-folder `"$ScenesFolder`" `
    -v"
if ($LASTEXITCODE -ne 0) {
    Write-Host "Subtitle generation failed!" -ForegroundColor Red
    exit 1
}

Write-Host "`n=== Pipeline Complete! ===" -ForegroundColor Green
Write-Host "Results in: $OutputFolder" -ForegroundColor Cyan
Write-Host "Face clusters: $OutputFolder\faces_export" -ForegroundColor Cyan
Write-Host "Enhanced videos: $ScenesFolder (look for *_enhanced.mp4)" -ForegroundColor Cyan
Write-Host "Subtitles: $ScenesFolder (look for *.srt files)" -ForegroundColor Cyan

