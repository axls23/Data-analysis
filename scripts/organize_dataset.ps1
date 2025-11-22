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
    [Parameter(Mandatory=$true, HelpMessage="Path to the dataset folder to organize")]
    [string]$DatasetPath
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

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "DATASET ORGANIZER" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Target folder: $DatasetPath" -ForegroundColor White
Write-Host ""

# Get absolute path
$DatasetPath = Resolve-Path $DatasetPath

# Check current structure
$subdirs = Get-ChildItem -Path $DatasetPath -Directory -ErrorAction SilentlyContinue
$images = Get-ChildItem -Path "$DatasetPath\*" -File -Include *.jpg,*.jpeg,*.png,*.bmp -ErrorAction SilentlyContinue

Write-Host "Analysis:" -ForegroundColor Yellow
Write-Host "  Subdirectories found: $($subdirs.Count)" -ForegroundColor White
Write-Host "  Images in root: $($images.Count)" -ForegroundColor White
Write-Host ""

# Check if already organized
$hasEmotionFolders = $false
foreach ($dir in $subdirs) {
    if ($ValidEmotions -contains $dir.Name.ToLower()) {
        $hasEmotionFolders = $true
        break
    }
}

if ($hasEmotionFolders -and $images.Count -eq 0) {
    Write-Host "✓ Dataset appears to be already organized into emotion folders." -ForegroundColor Green
    Write-Host ""
    Write-Host "Folder structure:" -ForegroundColor Yellow
    foreach ($emotion in $ValidEmotions) {
        $emotionPath = Join-Path $DatasetPath $emotion
        if (Test-Path $emotionPath) {
            $count = (Get-ChildItem -Path "$emotionPath\*" -File -Include *.jpg,*.jpeg,*.png,*.bmp, *.JPG).Count
            Write-Host "  $emotion/: $count images" -ForegroundColor White
        }
    }
    Write-Host ""
    Write-Host "No action needed." -ForegroundColor Green
    exit 0
}

# If images found in root, organize them
if ($images.Count -gt 0) {
    Write-Host "✓ Found $($images.Count) images in root folder. Organizing..." -ForegroundColor Yellow
    Write-Host ""
    
    # Create emotion folders if they don't exist
    foreach ($emotion in $ValidEmotions) {
        $emotionPath = Join-Path $DatasetPath $emotion
        if (-not (Test-Path $emotionPath)) {
            New-Item -Path $emotionPath -ItemType Directory -Force | Out-Null
            Write-Host "  [CREATED] $emotion/" -ForegroundColor Green
        }
    }
    
    Write-Host ""
    Write-Host "Processing images..." -ForegroundColor Yellow
    Write-Host "-" * 70 -ForegroundColor Cyan
    
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
                $destPath = Join-Path $DatasetPath $emotionFolder
                $destFile = Join-Path $destPath $filename
                
                try {
                    Move-Item -Path $image.FullName -Destination $destFile -Force
                    Write-Host "  [OK] $filename -> $emotionFolder/" -ForegroundColor Green
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
    
    Write-Host "-" * 70 -ForegroundColor Cyan
    Write-Host ""
    Write-Host "SUMMARY" -ForegroundColor Cyan
    Write-Host "=" * 70 -ForegroundColor Cyan
    Write-Host "  Images processed: $processedCount" -ForegroundColor Green
    Write-Host "  Images skipped: $skippedCount" -ForegroundColor Yellow
    Write-Host "  Errors: $errorCount" -ForegroundColor Red
    Write-Host ""
    
    # Show final folder structure
    Write-Host "Final folder structure:" -ForegroundColor Yellow
    foreach ($emotion in $ValidEmotions) {
        $emotionPath = Join-Path $DatasetPath $emotion
        if (Test-Path $emotionPath) {
            $count = (Get-ChildItem -Path "$emotionPath\*" -File -Include *.jpg,*.jpeg,*.png,*.bmp).Count
            Write-Host "  $emotion/: $count images" -ForegroundColor White
        }
    }
    
    Write-Host ""
    Write-Host "✓ Organization complete!" -ForegroundColor Green
}
else {
    Write-Host "⚠ No images found in root folder and no emotion folders detected." -ForegroundColor Yellow
    Write-Host "  This folder may already be organized or empty." -ForegroundColor White
}

Write-Host "=" * 70 -ForegroundColor Cyan
