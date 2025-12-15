# Download all VRT pretrained models
# This script downloads the high-quality models from the VRT releases

$ModelCache = "$env:USERPROFILE\.cache\vrt_models"
New-Item -ItemType Directory -Force -Path $ModelCache | Out-Null

$Models = @(
    @{
        File = "001_VRT_videosr_bi_REDS_6frames.pth"
        URL = "https://github.com/JingyunLiang/VRT/releases/download/v0.0/001_VRT_videosr_bi_REDS_6frames.pth"
        Size = "158 MB"
    },
    @{
        File = "002_VRT_videosr_bi_REDS_16frames.pth"
        URL = "https://github.com/JingyunLiang/VRT/releases/download/v0.0/002_VRT_videosr_bi_REDS_16frames.pth"
        Size = "201 MB"
        Priority = "HIGH"
    },
    @{
        File = "003_VRT_videosr_bi_Vimeo_7frames.pth"
        URL = "https://github.com/JingyunLiang/VRT/releases/download/v0.0/003_VRT_videosr_bi_Vimeo_7frames.pth"
        Size = "191 MB"
        Priority = "HIGH"
    },
    @{
        File = "004_VRT_videosr_bd_Vimeo_7frames.pth"
        URL = "https://github.com/JingyunLiang/VRT/releases/download/v0.0/004_VRT_videosr_bd_Vimeo_7frames.pth"
        Size = "191 MB"
    },
    @{
        File = "005_VRT_videodeblurring_DVD.pth"
        URL = "https://github.com/JingyunLiang/VRT/releases/download/v0.0/005_VRT_videodeblurring_DVD.pth"
        Size = "102 MB"
    },
    @{
        File = "006_VRT_videodeblurring_GoPro.pth"
        URL = "https://github.com/JingyunLiang/VRT/releases/download/v0.0/006_VRT_videodeblurring_GoPro.pth"
        Size = "102 MB"
    },
    @{
        File = "007_VRT_videodeblurring_REDS.pth"
        URL = "https://github.com/JingyunLiang/VRT/releases/download/v0.0/007_VRT_videodeblurring_REDS.pth"
        Size = "102 MB"
        Priority = "HIGH"
    },
    @{
        File = "008_VRT_videodenoising_DAVIS.pth"
        URL = "https://github.com/JingyunLiang/VRT/releases/download/v0.0/008_VRT_videodenoising_DAVIS.pth"
        Size = "102 MB"
        Priority = "HIGH"
    },
    @{
        File = "009_VRT_videofi_Vimeo_4frames.pth"
        URL = "https://github.com/JingyunLiang/VRT/releases/download/v0.0/009_VRT_videofi_Vimeo_4frames.pth"
        Size = "59.6 MB"
        Priority = "HIGH"
    },
    @{
        File = "spynet_sintel_final-3d2a1287.pth"
        URL = "https://github.com/JingyunLiang/VRT/releases/download/v0.0/spynet_sintel_final-3d2a1287.pth"
        Size = "5.5 MB"
    }
)

Write-Host "=== VRT Model Downloader ===" -ForegroundColor Cyan
Write-Host "Cache directory: $ModelCache" -ForegroundColor White
Write-Host ""

$HighPriority = $Models | Where-Object { $_.Priority -eq "HIGH" }
$OtherModels = $Models | Where-Object { $_.Priority -ne "HIGH" }

Write-Host "=== High Priority Models (Recommended) ===" -ForegroundColor Green
foreach ($model in $HighPriority) {
    $outputPath = Join-Path $ModelCache $model.File
    if (Test-Path $outputPath) {
        Write-Host "  ✓ $($model.File) (already downloaded)" -ForegroundColor Gray
    } else {
        Write-Host "  Downloading: $($model.File) ($($model.Size))..." -ForegroundColor Yellow
        try {
            Invoke-WebRequest -Uri $model.URL -OutFile $outputPath -UseBasicParsing
            Write-Host "    ✓ Downloaded" -ForegroundColor Green
        } catch {
            Write-Host "    ✗ Failed: $_" -ForegroundColor Red
        }
    }
}

Write-Host "`n=== Other Models ===" -ForegroundColor Yellow
foreach ($model in $OtherModels) {
    $outputPath = Join-Path $ModelCache $model.File
    if (Test-Path $outputPath) {
        Write-Host "  ✓ $($model.File) (already downloaded)" -ForegroundColor Gray
    } else {
        Write-Host "  Downloading: $($model.File) ($($model.Size))..." -ForegroundColor Yellow
        try {
            Invoke-WebRequest -Uri $model.URL -OutFile $outputPath -UseBasicParsing
            Write-Host "    ✓ Downloaded" -ForegroundColor Green
        } catch {
            Write-Host "    ✗ Failed: $_" -ForegroundColor Red
        }
    }
}

Write-Host "`n=== Download Complete ===" -ForegroundColor Green
$downloaded = (Get-ChildItem $ModelCache -Filter "*.pth").Count
Write-Host "Total models cached: $downloaded" -ForegroundColor White

