# Monitor process_parallel.py execution
# Run this script to check status every few minutes

$LogFile = "process_parallel_output.log"
$MonitorLog = "process_parallel_monitor.log"
$CheckInterval = 180  # 3 minutes

Write-Host "=== Monitoring process_parallel.py ===" -ForegroundColor Cyan
Write-Host "Log file: $LogFile" -ForegroundColor Gray
Write-Host "Check interval: $CheckInterval seconds" -ForegroundColor Gray
Write-Host ""

$iteration = 0
while ($true) {
    $iteration++
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    Write-Host "`n[$timestamp] Check #$iteration" -ForegroundColor Yellow
    
    # Check if process is running
    $pythonProcs = Get-Process python -ErrorAction SilentlyContinue | Where-Object {$_.Path -like "*anaconda*"}
    if ($pythonProcs) {
        Write-Host "  ✓ Python process(es) running:" -ForegroundColor Green
        $pythonProcs | ForEach-Object {
            $cpuTime = if ($_.CPU) { "{0:N2}" -f $_.CPU } else { "N/A" }
            $memMB = [math]::Round($_.WorkingSet64/1MB, 2)
            Write-Host "    PID: $($_.Id) | CPU: $cpuTime | Memory: $memMB MB | Runtime: $((Get-Date) - $_.StartTime)" -ForegroundColor Gray
        }
    } else {
        Write-Host "  ✗ No Python processes found" -ForegroundColor Red
    }
    
    # Check log file for recent activity
    if (Test-Path $LogFile) {
        $logSize = (Get-Item $LogFile).Length / 1KB
        $lastModified = (Get-Item $LogFile).LastWriteTime
        $timeSinceUpdate = (Get-Date) - $lastModified
        
        Write-Host "  Log file: $([math]::Round($logSize, 2)) KB | Last updated: $timeSinceUpdate ago" -ForegroundColor Gray
        
        # Get recent log entries
        $recentLogs = Get-Content $LogFile -Tail 20 -ErrorAction SilentlyContinue
        if ($recentLogs) {
            Write-Host "  Recent log entries:" -ForegroundColor Cyan
            $recentLogs | Select-Object -Last 10 | ForEach-Object {
                if ($_ -match "ERROR|Failed|FAILED") {
                    Write-Host "    $_" -ForegroundColor Red
                } elseif ($_ -match "WARNING|Warning") {
                    Write-Host "    $_" -ForegroundColor Yellow
                } elseif ($_ -match "INFO.*worker|INFO.*main|INFO.*progress|INFO.*complete") {
                    Write-Host "    $_" -ForegroundColor Green
                } else {
                    Write-Host "    $_" -ForegroundColor Gray
                }
            }
        }
        
        # Check for errors
        $errorCount = (Get-Content $LogFile -ErrorAction SilentlyContinue | Select-String -Pattern "ERROR|Failed|FAILED" | Measure-Object).Count
        if ($errorCount -gt 0) {
            Write-Host "  ⚠ Total errors in log: $errorCount" -ForegroundColor Yellow
        }
        
        # Check for completion
        $completed = Get-Content $LogFile -ErrorAction SilentlyContinue | Select-String -Pattern "Processing complete|All workers completed"
        if ($completed) {
            Write-Host "  ✓ Process appears to have completed!" -ForegroundColor Green
            $completed | Select-Object -Last 1 | ForEach-Object { Write-Host "    $_" -ForegroundColor Gray }
        }
    } else {
        Write-Host "  ⚠ Log file not found: $LogFile" -ForegroundColor Yellow
    }
    
    # Check monitor log if it exists
    if (Test-Path $MonitorLog) {
        $monitorSize = (Get-Item $MonitorLog).Length / 1KB
        Write-Host "  Monitor log: $([math]::Round($monitorSize, 2)) KB" -ForegroundColor Gray
    }
    
    Write-Host "  Next check in $($CheckInterval) seconds..." -ForegroundColor Gray
    Start-Sleep -Seconds $CheckInterval
}



