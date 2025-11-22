#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Organize dataset images into expression folders based on naming convention.

.DESCRIPTION
    This script processes a dataset folder and organizes images into expression subfolders.
    If images follow the naming convention: <USN>-<PersonNumber>-<Emotion>-<ImageNumber>
    they will be moved into their respective emotion folders.

.PARAMETER DatasetPath
    Path to the dataset folder to organize (e.g., "dataset/d3")

.EXAMPLE
    .\organize_dataset.ps1 -DatasetPath "dataset/d3"
    
.EXAMPLE
    .\organize_dataset.ps1 -DatasetPath "dataset/d1"
#>

param(
    [Parameter(Mandatory=$false, HelpMessage="Path to the dataset folder to organize")]
    [string]$DatasetPath = "dataset",
    
    [Parameter(Mandatory=$false)]
    [switch]$Recurse
)

# Define emotion mappings (code -> folder name)
$EmotionMap = @{
    'AN' = 'angry'
    'DI' = 'disgust'
    'FE' = 'fear'
    'HA' = 'happy'
    'NE' = 'neutral'
    'SA' = 'sad'
    'SU' = 'surprised'
}

# All valid emotion folders
$ValidEmotions = @('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised')

# Check if dataset path exists
if (-not (Test-Path $DatasetPath)) {
    Write-Host "ERROR: Dataset path '$DatasetPath' does not exist!" -ForegroundColor Red
    exit 1
}

$DatasetPath = Resolve-Path $DatasetPath

# Auto-enable recursion if targeting the root dataset folder
if (-not $Recurse.IsPresent -and (Split-Path -Leaf $DatasetPath) -eq "dataset") {
    Write-Host "Targeting root 'dataset' folder - Enabling recursive mode." -ForegroundColor Cyan
    $Recurse = $true
}

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "DATASET ORGANIZER" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Target folder: $DatasetPath" -ForegroundColor White
Write-Host "Recurse: $($Recurse.IsPresent)" -ForegroundColor White
Write-Host ""

# Collect folders to process
$foldersToProcess = @()
if ($Recurse) {
    # Get all subdirectories (d1, d2, etc.)
    $foldersToProcess = Get-ChildItem -Path $DatasetPath -Directory
} else {
    $foldersToProcess += Get-Item $DatasetPath
}

foreach ($currentFolder in $foldersToProcess) {
    $folderPath = $currentFolder.FullName
    $folderName = $currentFolder.Name
    
    Write-Host "Processing: $folderName" -ForegroundColor Yellow
    Write-Host "-" * 70 -ForegroundColor Cyan

    # Check current structure
    $subdirs = Get-ChildItem -Path $folderPath -Directory -ErrorAction SilentlyContinue
    $images = Get-ChildItem -Path "$folderPath\*" -File -Include *.jpg,*.jpeg,*.png,*.bmp -ErrorAction SilentlyContinue

    # Check if already organized
    $hasEmotionFolders = $false
    foreach ($dir in $subdirs) {
        if ($ValidEmotions -contains $dir.Name.ToLower()) {
            $hasEmotionFolders = $true
            break
        }
    }

    if ($hasEmotionFolders -and $images.Count -eq 0) {
        Write-Host "  [✓] Already organized." -ForegroundColor Green
        continue
    }

    # If images found in root, organize them
    if ($images.Count -gt 0) {
        # Create emotion folders if they don't exist
        foreach ($emotion in $ValidEmotions) {
            $emotionPath = Join-Path $folderPath $emotion
            if (-not (Test-Path $emotionPath)) {
                New-Item -Path $emotionPath -ItemType Directory -Force | Out-Null
            }
        }
        
        $processedCount = 0
        $skippedCount = 0
        $errorCount = 0
        
        foreach ($image in $images) {
            $filename = $image.Name
            
            # Parse filename: <USN>-<PersonNumber>-<Emotion>-<ImageNumber>
            # Example: 23BTRCL003-01-HA-01.jpg
            if ($filename -match '^(.+?)-(\d+)-([A-Z]{2})-(\d+)\.(jpg|jpeg|png|bmp)$') {
                $usn = $matches[1]
                $personNum = $matches[2]
                $emotionCode = $matches[3]
                $imageNum = $matches[4]
                $extension = $matches[5]
                
                # Map emotion code to folder name
                if ($EmotionMap.ContainsKey($emotionCode)) {
                    $emotionFolder = $EmotionMap[$emotionCode]
                    $destPath = Join-Path $folderPath $emotionFolder
                    
                    # Enforce .jpg extension
                    $baseName = [System.IO.Path]::GetFileNameWithoutExtension($filename)
                    $destFilename = "${baseName}.jpg"
                    $destFile = Join-Path $destPath $destFilename
                    
                    try {
                        Move-Item -Path $image.FullName -Destination $destFile -Force
                        $processedCount++
                    }
                    catch {
                        Write-Host "  [ERROR] Failed to move $filename`: $($_.Exception.Message)" -ForegroundColor Red
                        $errorCount++
                    }
                }
                else {
                    Write-Host "  [SKIP] Unknown emotion code '$emotionCode' in: $filename" -ForegroundColor Yellow
                    $skippedCount++
                }
            }
            else {
                Write-Host "  [SKIP] Invalid naming format: $filename" -ForegroundColor Yellow
                $skippedCount++
            }
        }
        
        Write-Host "  Processed: $processedCount | Skipped: $skippedCount | Errors: $errorCount" -ForegroundColor White
    }
    else {
        Write-Host "  [!] No images found in root and no emotion folders." -ForegroundColor Yellow
    }
    Write-Host ""
}

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Organization complete!" -ForegroundColor Green
Write-Host "=" * 70 -ForegroundColor Cyan
exit 0

# ...existing code...
