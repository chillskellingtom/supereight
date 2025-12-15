# PowerShell wrapper to run Intel GPU setup in WSL
# This script runs the installation script inside WSL Ubuntu-22.04

Write-Host "=== Intel GPU Setup for WSL ===" -ForegroundColor Cyan
Write-Host ""

# Check if WSL distro exists
$distro = "Ubuntu-22.04"
Write-Host "Checking for WSL distro: $distro" -ForegroundColor Yellow

$wslList = wsl.exe -l -q 2>&1
if ($wslList -notmatch $distro) {
    Write-Host "ERROR: WSL distro '$distro' not found!" -ForegroundColor Red
    Write-Host "Available distros:" -ForegroundColor Yellow
    wsl.exe -l
    exit 1
}

Write-Host "Found WSL distro: $distro" -ForegroundColor Green
Write-Host ""

# Copy the installation script to WSL
Write-Host "Copying installation script to WSL..." -ForegroundColor Yellow
$scriptPath = Join-Path $PSScriptRoot "install_intel_gpu_wsl.sh"
$wslScriptPath = "/tmp/install_intel_gpu_wsl.sh"

# Convert Windows path to WSL path
$wslPath = $scriptPath -replace '\\', '/' -replace '^C:', '/mnt/c' -replace '^([A-Z]):', '/mnt/$1'
$wslPath = $wslPath.ToLower()

Write-Host "Windows path: $scriptPath" -ForegroundColor Gray
Write-Host "WSL path: $wslPath" -ForegroundColor Gray

# Copy script to WSL
wsl.exe -d $distro -- bash -c "cp '$wslPath' '$wslScriptPath' && chmod +x '$wslScriptPath'"

if ($LASTEXITCODE -ne 0) {
    Write-Host "ERROR: Failed to copy script to WSL" -ForegroundColor Red
    Write-Host "Trying alternative method..." -ForegroundColor Yellow
    
    # Alternative: create script directly in WSL
    $scriptContent = Get-Content $scriptPath -Raw
    wsl.exe -d $distro -- bash -c "cat > '$wslScriptPath' << 'EOFMARKER'
$scriptContent
EOFMARKER
chmod +x '$wslScriptPath'"
}

Write-Host "Running installation script in WSL..." -ForegroundColor Yellow
Write-Host "NOTE: You may be prompted for sudo password" -ForegroundColor Cyan
Write-Host ""

# Run the installation script
wsl.exe -d $distro -- bash -c "cd ~ && '$wslScriptPath'"

if ($LASTEXITCODE -eq 0) {
    Write-Host ""
    Write-Host "=== Setup Complete ===" -ForegroundColor Green
    Write-Host ""
    Write-Host "To test GPU detection, run:" -ForegroundColor Cyan
    Write-Host "  wsl.exe -d $distro -- bash -c 'source .venv/bin/activate && python3 -c \"import torch; import intel_extension_for_pytorch as ipex; print(''PyTorch:'', torch.__version__); print(''IPEX:'', ipex.__version__); print(''XPU available:'', torch.xpu.is_available() if hasattr(torch, ''xpu'') else False)\"'" -ForegroundColor White
} else {
    Write-Host ""
    Write-Host "=== Setup Failed ===" -ForegroundColor Red
    Write-Host "Check the error messages above for details." -ForegroundColor Yellow
    exit 1
}

