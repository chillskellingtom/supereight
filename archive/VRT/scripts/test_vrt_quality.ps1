# Test script for VRT (Video Restoration Transformer) quality comparison
# Runs all VRT tasks on a single scene and creates side-by-side comparisons

param(
    [string]$ScenePath = "",
    [string]$ScenesFolder = "C:\Users\latch\connor_family_movies_processed\scenes",
    [switch]$DownloadModels = $false
)

$VRTModels = @(
    @{Key="videosr_reds_16frames"; Name="Video Super-Resolution (REDS 16-frames)"; Desc="Highest quality SR model"},
    @{Key="videosr_vimeo"; Name="Video Super-Resolution (Vimeo)"; Desc="High quality SR"},
    @{Key="videodeblur_reds"; Name="Video Deblurring (REDS)"; Desc="Motion blur removal"},
    @{Key="videodenoise"; Name="Video Denoising"; Desc="Noise reduction"},
    @{Key="videofi"; Name="Video Frame Interpolation"; Desc="Frame rate upscaling"}
)

if (-not $ScenePath) {
    Write-Host "Finding first available scene..." -ForegroundColor Yellow
    $scenes = Get-ChildItem -Path $ScenesFolder -Recurse -Filter "*.mp4" | 
        Where-Object { $_.Name -notlike "*_enhanced*" -and $_.Name -notlike "*_audiodenoise*" -and $_.Name -notlike "*_vrt*" } | 
        Select-Object -First 1
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

Write-Host "`n=== VRT Video Enhancement Test ===" -ForegroundColor Cyan
Write-Host "Scene: $($SceneFile.Name)" -ForegroundColor White
Write-Host "Output directory: $OutputDir" -ForegroundColor White

# Check if VRT is available
$vrtScript = Join-Path $PSScriptRoot "vrt_enhance.py"
if (-not (Test-Path $vrtScript)) {
    Write-Host "`nvrt_enhance.py not found at: $vrtScript" -ForegroundColor Red
    Write-Host "Please ensure vrt_enhance.py is in the same directory as this script." -ForegroundColor Yellow
    exit 1
}

# Download models if requested
if ($DownloadModels) {
    Write-Host "`n=== Downloading VRT Models ===" -ForegroundColor Green
    foreach ($model in $VRTModels) {
        Write-Host "`nDownloading: $($model.Name)" -ForegroundColor Yellow
        python $vrtScript --task $model.Key --input $ScenePath --output "$OutputDir\${BaseName}_vrt_download_test.mp4" 2>&1 | Out-Null
        if ($LASTEXITCODE -eq 0) {
            Remove-Item "$OutputDir\${BaseName}_vrt_download_test.mp4" -ErrorAction SilentlyContinue
            Write-Host "  ✓ Model downloaded" -ForegroundColor Green
        } else {
            Write-Host "  ✗ Failed to download model" -ForegroundColor Red
        }
    }
    Write-Host "`nModel download complete!" -ForegroundColor Green
    exit 0
}

# Get original video info
Write-Host "`n=== Original Video Info ===" -ForegroundColor Cyan
$origInfo = ffprobe -v error -select_streams v:0 -show_entries stream=width,height,bit_rate,codec_name,r_frame_rate -of json "$ScenePath" | ConvertFrom-Json
$origWidth = $origInfo.streams[0].width
$origHeight = $origInfo.streams[0].height
$origCodec = $origInfo.streams[0].codec_name
$origBitrate = if ($origInfo.streams[0].bit_rate) { [math]::Round($origInfo.streams[0].bit_rate / 1000000, 2) } else { "unknown" }
$fpsParts = $origInfo.streams[0].r_frame_rate.Split("/")
$origFPS = if ($fpsParts.Length -eq 2) { [math]::Round([double]$fpsParts[0] / [double]$fpsParts[1], 2) } else { "unknown" }

Write-Host "  Resolution: ${origWidth}x${origHeight}" -ForegroundColor White
Write-Host "  Codec: $origCodec" -ForegroundColor White
Write-Host "  Bitrate: ${origBitrate} Mbps" -ForegroundColor White
Write-Host "  Frame rate: ${origFPS} fps" -ForegroundColor White

# Process each VRT task
$Results = @()

foreach ($model in $VRTModels) {
    Write-Host "`n=== Processing: $($model.Name) ===" -ForegroundColor Green
    Write-Host "Description: $($model.Desc)" -ForegroundColor Yellow
    
    $outputFile = Join-Path $OutputDir "${BaseName}_vrt_$($model.Key).mp4"
    $comparisonFile = Join-Path $OutputDir "${BaseName}_vrt_$($model.Key)_comparison.mp4"
    
    # Run VRT processing
    # Use the same Python that has VRT dependencies installed
    $pythonExe = (Get-Command python).Source
    & $pythonExe $vrtScript `
        --task $model.Key `
        --input "$ScenePath" `
        --output "$outputFile" `
        --comparison "$comparisonFile" `
        -v
    
    if ($LASTEXITCODE -eq 0) {
        # Get enhanced video info
        $enhInfo = ffprobe -v error -select_streams v:0 -show_entries stream=width,height,bit_rate,codec_name -of json "$outputFile" | ConvertFrom-Json
        $enhWidth = $enhInfo.streams[0].width
        $enhHeight = $enhInfo.streams[0].height
        $enhBitrate = if ($enhInfo.streams[0].bit_rate) { [math]::Round($enhInfo.streams[0].bit_rate / 1000000, 2) } else { "unknown" }
        
        $origSize = [math]::Round((Get-Item $ScenePath).Length / 1MB, 2)
        $enhSize = [math]::Round((Get-Item $outputFile).Length / 1MB, 2)
        $compSize = [math]::Round((Get-Item $comparisonFile).Length / 1MB, 2)
        
        $Results += [PSCustomObject]@{
            Task = $model.Name
            Original = "${origWidth}x${origHeight}"
            Enhanced = "${enhWidth}x${enhHeight}"
            OriginalSize = "${origSize} MB"
            EnhancedSize = "${enhSize} MB"
            ComparisonSize = "${compSize} MB"
            OutputFile = $outputFile
            ComparisonFile = $comparisonFile
        }
        
        Write-Host "  ✓ Complete" -ForegroundColor Green
        Write-Host "    Resolution: ${origWidth}x${origHeight} -> ${enhWidth}x${enhHeight}" -ForegroundColor White
        Write-Host "    Size: ${origSize} MB -> ${enhSize} MB" -ForegroundColor White
    } else {
        Write-Host "  ✗ Failed" -ForegroundColor Red
        $Results += [PSCustomObject]@{
            Task = $model.Name
            Status = "Failed"
        }
    }
}

# Summary
Write-Host "`n=== Processing Summary ===" -ForegroundColor Cyan
$Results | Format-Table -AutoSize

Write-Host "`n=== Output Files ===" -ForegroundColor Cyan
foreach ($result in $Results) {
    if ($result.OutputFile) {
        Write-Host "`n$($result.Task):" -ForegroundColor Yellow
        Write-Host "  Enhanced: $($result.OutputFile)" -ForegroundColor White
        Write-Host "  Comparison: $($result.ComparisonFile)" -ForegroundColor White
    }
}

Write-Host "`n=== Opening Comparison Videos ===" -ForegroundColor Green
foreach ($result in $Results) {
    if ($result.ComparisonFile -and (Test-Path $result.ComparisonFile)) {
        Write-Host "Opening: $($result.Task)" -ForegroundColor Yellow
        Start-Process "$($result.ComparisonFile)"
        Start-Sleep -Seconds 2
    }
}

Write-Host "`n=== Test Complete! ===" -ForegroundColor Green
Write-Host "All comparison videos have been opened." -ForegroundColor White
Write-Host "Review the side-by-side comparisons to evaluate VRT quality." -ForegroundColor White

