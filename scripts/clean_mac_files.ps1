#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Removes macOS AppleDouble metadata files (starting with .-) from the dataset.

.DESCRIPTION
    Recursively searches for and removes files starting with ".-" which are created 
    by macOS to store resource forks on non-HFS+ file systems. These files can 
    interfere with dataset processing.

.PARAMETER DatasetPath
    Path to the dataset folder to clean (default: "dataset")

.PARAMETER DryRun
    Show what would be deleted without actually deleting

.EXAMPLE
    .\clean_mac_files.ps1 -DryRun
    
.EXAMPLE
    .\clean_mac_files.ps1 -DatasetPath "dataset/d1"
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$DatasetPath = "dataset",
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun
)

if (-not (Test-Path $DatasetPath)) {
    Write-Host "ERROR: Path '$DatasetPath' does not exist!" -ForegroundColor Red
    exit 1
}

$DatasetPath = Resolve-Path $DatasetPath

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "MACOS METADATA CLEANER" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Target: $DatasetPath" -ForegroundColor White
Write-Host "DryRun: $($DryRun.IsPresent)" -ForegroundColor White
Write-Host ""

# Find files starting with .-
# We use -Force to ensure we find hidden files
$macFiles = Get-ChildItem -Path $DatasetPath -Recurse -Force -File | Where-Object { $_.Name -like ".-*" }

if ($macFiles.Count -eq 0) {
    Write-Host "✓ No macOS metadata files (.-*) found." -ForegroundColor Green
    exit 0
}

Write-Host "Found $($macFiles.Count) metadata files." -ForegroundColor Yellow
Write-Host ""

$deletedCount = 0
$errorCount = 0

foreach ($file in $macFiles) {
    $relPath = $file.FullName.Replace($DatasetPath, '').TrimStart('\')
    
    if ($DryRun) {
        Write-Host "  [DRY] Would delete: $relPath" -ForegroundColor Yellow
        $deletedCount++
    }
    else {
        try {
            Remove-Item -Path $file.FullName -Force
            Write-Host "  [DEL] Removed: $relPath" -ForegroundColor Red
            $deletedCount++
        }
        catch {
            Write-Host "  [ERR] Failed to remove $relPath`: $($_.Exception.Message)" -ForegroundColor DarkRed
            $errorCount++
        }
    }
}

Write-Host ""
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
if ($DryRun) {
    Write-Host "  Found (to delete): $deletedCount" -ForegroundColor Yellow
    Write-Host "  ⚠ DRY RUN - No files were actually deleted." -ForegroundColor Yellow
} else {
    Write-Host "  Deleted: $deletedCount" -ForegroundColor Green
    Write-Host "  Errors: $errorCount" -ForegroundColor Red
}
Write-Host "=" * 70 -ForegroundColor Cyan
