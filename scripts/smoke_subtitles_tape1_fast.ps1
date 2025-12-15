<#
Fast smoke test: subtitles for 2 canonical scenes (aim <60s).

Strategy:
- Find the "best" (smallest/most canonical) mp4 for Scene-001 and Scene-002
- Copy them into a temp folder
- Run process_parallel.py against that temp folder with --limit 2
- Exit non-zero on failure

Run:
  pwsh -File .\scripts\smoke_subtitles_tape1_fast.ps1
#>

$ErrorActionPreference = "Stop"

Write-Host "=== Fast Smoke Test: Subtitles (2 canonical scenes) ==="

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$VenvPython = Join-Path $RepoRoot ".venv_win_arcxpu\Scripts\python.exe"
$Python = if (Test-Path $VenvPython) { $VenvPython } else { "C:\Users\latch\AppData\Local\Programs\Python\Python311\python.exe" }

# Default scenes root matches your existing smoke script output.
$ScenesRootDefault = "C:\Users\latch\connor_family_movies_processed\scenes"
$ScenesRoot = if ($env:SCENES_ROOT -and (Test-Path $env:SCENES_ROOT)) { $env:SCENES_ROOT } else { $ScenesRootDefault }

if (-not (Test-Path $ScenesRoot)) {
  Write-Host "Scenes root not found: $ScenesRoot"
  exit 2
}

Write-Host "Repo:   $RepoRoot"
Write-Host "Python: $Python"
Write-Host "Scenes: $ScenesRoot"

function Get-BestSceneFile {
  param(
    [Parameter(Mandatory=$true)] [string] $ScenesRoot,
    [Parameter(Mandatory=$true)] [int]    $SceneNumber
  )

  $sceneNN = $SceneNumber.ToString("000")

  # Prefer exact canonical: *-Scene-001.mp4 (no extra suffixes)
  $exact = Get-ChildItem -Path $ScenesRoot -Recurse -File -Filter "*.mp4" -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -match "-Scene-$sceneNN\.mp4$" }

  if ($exact.Count -ge 1) {
    # If there are multiple, pick smallest (usually shortest) to keep runtime down.
    return $exact | Sort-Object Length | Select-Object -First 1
  }

  # Fallback: "mp4 that contains Scene-001" but exclude obvious non-canonical derivatives.
  $fallback = Get-ChildItem -Path $ScenesRoot -Recurse -File -Filter "*.mp4" -ErrorAction SilentlyContinue |
    Where-Object {
      $_.Name -match "-Scene-$sceneNN" -and
      $_.Name -notmatch "(?i)audiodenoise|enhanced|comparison|rife|realesrgan|libplacebo"
    }

  if ($fallback.Count -ge 1) {
    return $fallback | Sort-Object Length | Select-Object -First 1
  }

  return $null
}

$Scene1 = Get-BestSceneFile -ScenesRoot $ScenesRoot -SceneNumber 1
$Scene2 = Get-BestSceneFile -ScenesRoot $ScenesRoot -SceneNumber 2

if (-not $Scene1) { Write-Host "Could not find a suitable Scene-001 mp4 under $ScenesRoot"; exit 3 }
if (-not $Scene2) { Write-Host "Could not find a suitable Scene-002 mp4 under $ScenesRoot"; exit 3 }

Write-Host "Selected:"
Write-Host "  1) $($Scene1.FullName) ($([Math]::Round($Scene1.Length/1MB,2)) MB)"
Write-Host "  2) $($Scene2.FullName) ($([Math]::Round($Scene2.Length/1MB,2)) MB)"

# Create a temp scenes folder and copy exactly two files
$TempRoot = Join-Path $env:TEMP ("connor_smoke_fast_" + [Guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $TempRoot | Out-Null

Copy-Item -LiteralPath $Scene1.FullName -Destination (Join-Path $TempRoot $Scene1.Name) -Force
Copy-Item -LiteralPath $Scene2.FullName -Destination (Join-Path $TempRoot $Scene2.Name) -Force

Write-Host "Temp scenes folder: $TempRoot"
Write-Host "Task: subtitles | Limit: 2"

try {
  Push-Location $RepoRoot

  # Run the existing entrypoint against the temp folder.
  # NOTE: if your entrypoint differs, swap process_parallel.py path accordingly.
  & $Python "process_parallel.py" `
      --task "subtitles" `
      --scenes-folder $TempRoot `
      --limit 2 `
      -v

  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) {
    Write-Host "Fast smoke FAILED with exit code $exitCode"
    exit $exitCode
  }

  Write-Host "`n=== Fast smoke completed successfully ==="
  exit 0
}
finally {
  Pop-Location | Out-Null

  # Best-effort cleanup
  try { Remove-Item -LiteralPath $TempRoot -Recurse -Force -ErrorAction SilentlyContinue } catch {}
}
