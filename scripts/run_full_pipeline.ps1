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
$PythonParts = $PythonExe -split "\s+"

function Invoke-Py {
    param([string[]]$Args)
    $baseArgs = if ($PythonParts.Length -gt 1) { $PythonParts[1..($PythonParts.Length - 1)] } else { @() }
    $cmdPreview = @($PythonParts[0]) + $baseArgs + $Args
    Write-Host "-> $($cmdPreview -join ' ')" -ForegroundColor DarkGray
    & $PythonParts[0] @baseArgs @Args
    return $LASTEXITCODE
}

function Run-Step {
    param(
        [string]$Label,
        [string[]]$Args
    )
    Write-Host "=== $Label ===" -ForegroundColor Green
    $rc = Invoke-Py -Args $Args
    Write-Host "Exit code: $rc" -ForegroundColor DarkGray
    if ($rc -ne 0) {
        throw "$Label failed (exit code $rc)"
    }
}

Run-Step -Label "Step 1: Scene Detection" -Args @(
    $DetectScenesPy,
    "--input", $InputFolder,
    "--output", $OutputFolder,
    "-v"
)

Run-Step -Label "Step 2: Face Export (with auto-cluster)" -Args @(
    $ProcessParallelPy,
    "--task", "faces_export",
    "--scenes-folder", $ScenesFolder,
    "--frame-stride", "12",
    "--min-face-score", "0.6",
    "--face-crop-size", "256",
    "--face-max-per-track", "3",
    "--auto-cluster",
    "--cluster-method", "hierarchical",
    "--cluster-similarity", "0.90",
    "--cluster-min-samples", "5",
    "--cluster-quality-threshold", "0.3",
    "--cluster-n-passes", "3",
    "-v"
)

Run-Step -Label "Step 3: Video Enhancement" -Args @(
    $ProcessParallelPy,
    "--task", "video_enhance",
    "--scenes-folder", $ScenesFolder,
    "--enhance-scale", "2.0",
    "--enhance-denoise", "1.0",
    "--enhance-crf", "16",
    "--video-codec", "auto",
    "-v"
)

Run-Step -Label "Step 4: Audio Enhancement" -Args @(
    $ProcessParallelPy,
    "--task", "audio_enhance",
    "--scenes-folder", $ScenesFolder,
    "--audio-denoise", "-25",
    "--audio-bitrate", "192k",
    "-v"
)

Run-Step -Label "Step 5: Subtitles" -Args @(
    $ProcessParallelPy,
    "--task", "subtitles",
    "--scenes-folder", $ScenesFolder,
    "-v"
)

Write-Host "`n=== Pipeline Complete! ===" -ForegroundColor Green
Write-Host "Results in: $OutputFolder" -ForegroundColor Cyan
Write-Host "Face clusters: $OutputFolder\faces_export" -ForegroundColor Cyan
Write-Host "Enhanced videos: $ScenesFolder (look for *_enhanced.mp4)" -ForegroundColor Cyan
Write-Host "Subtitles: $ScenesFolder (look for *.srt files)" -ForegroundColor Cyan

