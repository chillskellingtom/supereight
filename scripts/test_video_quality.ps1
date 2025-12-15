# Test script for video quality comparison
# Processes a single scene with maximum quality settings and creates side-by-side comparison

param(
    [string]$ScenePath = "",
    [string]$ScenesFolder = "C:\Users\latch\connor_family_movies_processed\scenes"
)

if (-not $ScenePath) {
    Write-Host "Finding first available scene..." -ForegroundColor Yellow
    $scenes = Get-ChildItem -Path $ScenesFolder -Recurse -Filter "*.mp4" | Where-Object { $_.Name -notlike "*_enhanced*" -and $_.Name -notlike "*_audiodenoise*" } | Select-Object -First 1
    if ($scenes) {
        $ScenePath = $scenes.FullName
        Write-Host "Using: $($scenes.Name)" -ForegroundColor Green
    } else {
        Write-Host "No scenes found in $ScenesFolder" -ForegroundColor Red
        exit 1
    }
}

$SceneFile = Get-Item $ScenePath
$OutputDir = $SceneFile.Directory
$BaseName = $SceneFile.BaseName
$EnhancedPath = Join-Path $OutputDir "${BaseName}_enhanced.mp4"
$ComparisonPath = Join-Path $OutputDir "${BaseName}_comparison.mp4"

Write-Host "`n=== Original Scene ===" -ForegroundColor Cyan
Write-Host "Path: $ScenePath" -ForegroundColor White

# Get original video info
Write-Host "`nGetting original video info..." -ForegroundColor Yellow
$origInfo = ffprobe -v error -select_streams v:0 -show_entries stream=width,height,bit_rate,codec_name -of json "$ScenePath" | ConvertFrom-Json
$origWidth = $origInfo.streams[0].width
$origHeight = $origInfo.streams[0].height
$origCodec = $origInfo.streams[0].codec_name
$origBitrate = if ($origInfo.streams[0].bit_rate) { [math]::Round($origInfo.streams[0].bit_rate / 1000000, 2) } else { "unknown" }

Write-Host "  Resolution: ${origWidth}x${origHeight}" -ForegroundColor White
Write-Host "  Codec: $origCodec" -ForegroundColor White
Write-Host "  Bitrate: ${origBitrate} Mbps" -ForegroundColor White

# Process with maximum quality settings
Write-Host "`n=== Processing with Maximum Quality Settings ===" -ForegroundColor Green
Write-Host "Settings:" -ForegroundColor Yellow
Write-Host "  Scale: 2.5x (ultra high quality upscale)" -ForegroundColor White
Write-Host "  CRF: 12 (very high quality, larger file)" -ForegroundColor White
Write-Host "  Preset: slow (best quality encoding)" -ForegroundColor White
Write-Host "  Denoise: 0.8 (light denoising to preserve detail)" -ForegroundColor White
Write-Host "  Sharpening: enabled (unsharp mask)" -ForegroundColor White

# Note: The script will automatically enumerate if enhanced video exists (e.g., _enhanced(1).mp4)

# Get the scenes folder (should be the parent folder containing video subfolders)
# Structure: scenes/tape 1/video.mp4, so scenes folder is the parent of the video's directory
$ScenesBaseFolder = $SceneFile.Directory.Parent.FullName
Write-Host "Using scenes folder: $ScenesBaseFolder" -ForegroundColor Yellow

python process_parallel.py `
    --task video_enhance `
    --scene-file "$ScenePath" `
    --enhance-scale 2.5 `
    --enhance-denoise 0.8 `
    --enhance-crf 12 `
    --video-codec auto `
    -v

if ($LASTEXITCODE -ne 0) {
    Write-Host "`nVideo enhancement failed!" -ForegroundColor Red
    exit 1
}

# Find the actual enhanced file (may be enumerated like _enhanced(1).mp4)
$EnhancedPath = $null
$enhancedPattern = Join-Path $OutputDir "${BaseName}_enhanced*.mp4"
$enhancedFiles = Get-ChildItem -Path $OutputDir -Filter "${BaseName}_enhanced*.mp4" | Sort-Object LastWriteTime -Descending
if ($enhancedFiles) {
    $EnhancedPath = $enhancedFiles[0].FullName
    Write-Host "`nFound enhanced video: $($enhancedFiles[0].Name)" -ForegroundColor Green
} else {
    Write-Host "`nEnhanced video not found matching pattern: ${BaseName}_enhanced*.mp4" -ForegroundColor Red
    exit 1
}

# Get enhanced video info
Write-Host "`nGetting enhanced video info..." -ForegroundColor Yellow
$enhInfo = ffprobe -v error -select_streams v:0 -show_entries stream=width,height,bit_rate,codec_name -of json "$EnhancedPath" | ConvertFrom-Json
$enhWidth = $enhInfo.streams[0].width
$enhHeight = $enhInfo.streams[0].height
$enhCodec = $enhInfo.streams[0].codec_name
$enhBitrate = if ($enhInfo.streams[0].bit_rate) { [math]::Round($enhInfo.streams[0].bit_rate / 1000000, 2) } else { "unknown" }

Write-Host "  Resolution: ${enhWidth}x${enhHeight}" -ForegroundColor White
Write-Host "  Codec: $enhCodec" -ForegroundColor White
Write-Host "  Bitrate: ${enhBitrate} Mbps" -ForegroundColor White

# Create side-by-side comparison
Write-Host "`n=== Creating Side-by-Side Comparison ===" -ForegroundColor Green

# Scale both videos to same height for comparison (use enhanced height)
$comparisonHeight = $enhHeight
$origScaleW = [math]::Round($origWidth * $comparisonHeight / $origHeight / 2) * 2  # Make even
$origScaleH = $comparisonHeight
$enhScaleW = [math]::Round($enhWidth * $comparisonHeight / $enhHeight / 2) * 2  # Make even
$enhScaleH = $comparisonHeight

$totalWidth = $origScaleW + $enhScaleW

Write-Host "  Comparison resolution: ${totalWidth}x${comparisonHeight}" -ForegroundColor White
Write-Host "  Left: Original (${origScaleW}x${origScaleH})" -ForegroundColor White
Write-Host "  Right: Enhanced (${enhScaleW}x${enhScaleH})" -ForegroundColor White

# Create side-by-side with labels
$vf = "scale=$origScaleW`:$origScaleH[orig];scale=$enhScaleW`:$enhScaleH[enh];[orig]pad=iw+$enhScaleW`:ih:0:0:black[left];[left][enh]overlay=$origScaleW`:0[out];[out]drawtext=text='Original':fontsize=24:fontcolor=white:x=10:y=10:box=1:boxcolor=black@0.5,drawtext=text='Enhanced (2.5x upscale, CRF 12)':fontsize=24:fontcolor=white:x=$($origScaleW+10):y=10:box=1:boxcolor=black@0.5"

ffmpeg -y `
    -i "$ScenePath" `
    -i "$EnhancedPath" `
    -filter_complex $vf `
    -c:v libx264 `
    -preset medium `
    -crf 18 `
    -c:a copy `
    -shortest `
    "$ComparisonPath"

if ($LASTEXITCODE -eq 0) {
    Write-Host "`n=== Comparison Complete! ===" -ForegroundColor Green
    Write-Host "`nFiles created:" -ForegroundColor Cyan
    Write-Host "  Original: $ScenePath" -ForegroundColor White
    Write-Host "  Enhanced: $EnhancedPath" -ForegroundColor White
    Write-Host "  Comparison: $ComparisonPath" -ForegroundColor White
    
    $origSize = [math]::Round((Get-Item $ScenePath).Length / 1MB, 2)
    $enhSize = [math]::Round((Get-Item $EnhancedPath).Length / 1MB, 2)
    $compSize = [math]::Round((Get-Item $ComparisonPath).Length / 1MB, 2)
    
    Write-Host "`nFile sizes:" -ForegroundColor Cyan
    Write-Host "  Original: ${origSize} MB" -ForegroundColor White
    Write-Host "  Enhanced: ${enhSize} MB ($([math]::Round($enhSize / $origSize, 2))x larger)" -ForegroundColor White
    Write-Host "  Comparison: ${compSize} MB" -ForegroundColor White
    
    Write-Host "`nOpen the comparison video to see side-by-side:" -ForegroundColor Yellow
    Write-Host "  $ComparisonPath" -ForegroundColor White
    
    # Try to open the comparison video
    Start-Process "$ComparisonPath"
} else {
    Write-Host "`nComparison creation failed!" -ForegroundColor Red
    exit 1
}

