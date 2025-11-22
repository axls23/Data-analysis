#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Process all zip files from datasets_zips and organize them into dataset folders.

.DESCRIPTION
    This script:
    1. Extracts all zip files from datasets_zips/ into dataset/d<num>/ folders
    2. Validates and fixes filenames to match strict naming convention
    3. Creates emotion folders (angry, disgust, fear, happy, neutral, sad, surprised)
    4. Organizes files into emotion folders based on emotion codes
    5. Removes any non-standard folders

.PARAMETER ZipsFolder
    Path to folder containing zip files (default: "datasets_zips")

.PARAMETER DatasetFolder
    Path to dataset output folder (default: "dataset")

.PARAMETER StartNumber
    Starting number for d<num> folders (auto-detected if not specified)

.EXAMPLE
    .\process_zip_datasets.ps1

.EXAMPLE
    .\process_zip_datasets.ps1 -ZipsFolder "my_zips" -DatasetFolder "my_dataset"
#>

param(
    [Parameter(Mandatory=$false)]
    [string]$ZipsFolder = "datasets_zips",
    
    [Parameter(Mandatory=$false)]
    [string]$DatasetFolder = "dataset",
    
    [Parameter(Mandatory=$false)]
    [int]$StartNumber = -1
)

# Emotion mappings and valid values
$EmotionMap = @{
    'angry' = 'AN'; 'AN' = 'AN'
    'disgust' = 'DI'; 'DI' = 'DI'
    'fear' = 'FE'; 'FE' = 'FE'
    'happy' = 'HA'; 'HA' = 'HA'; 'happiness' = 'HA'
    'neutral' = 'NE'; 'NE' = 'NE'; 'NU' = 'NE'
    'sad' = 'SA'; 'SA' = 'SA'
    'surprised' = 'SU'; 'surprise' = 'SU'; 'SU' = 'SU'
}

$ValidEmotions = @('AN', 'DI', 'FE', 'HA', 'NE', 'SA', 'SU')
$EmotionFolders = @('angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprised')
$EmotionToFolder = @{
    'AN' = 'angry'; 'DI' = 'disgust'; 'FE' = 'fear'
    'HA' = 'happy'; 'NE' = 'neutral'; 'SA' = 'sad'; 'SU' = 'surprised'
}

function Normalize-Emotion {
    param([string]$emotion)
    
    if ($EmotionMap.ContainsKey($emotion.ToLower())) {
        return $EmotionMap[$emotion.ToLower()]
    }
    if ($EmotionMap.ContainsKey($emotion.ToUpper())) {
        return $EmotionMap[$emotion.ToUpper()]
    }
    if ($EmotionMap.ContainsKey($emotion)) {
        return $EmotionMap[$emotion]
    }
    
    if ($emotion.Length -ge 2) {
        $twoLetter = $emotion.Substring(0, 2).ToUpper()
        if ($ValidEmotions -contains $twoLetter) {
            return $twoLetter
        }
    }
    
    return $null
}

function Fix-Filename {
    param([string]$filename)
    
    $base = [System.IO.Path]::GetFileNameWithoutExtension($filename)
    
    # Remove spaces
    $normalized = $base -replace '\s', ''
    
    # Convert underscores to hyphens
    $normalized = $normalized -replace '_', '-'
    $normalized = $normalized -replace '--+', '-'
    
    # Robust regex matching (searches within string)
    if ($normalized -match '(23[A-Za-z0-9]+)-(\d+)-([A-Za-z]+)-(\d+)') {
        $usn = $matches[1]
        $personNum = $matches[2].PadLeft(2, '0')
        $emotionInput = $matches[3]
        $imageNum = $matches[4].PadLeft(2, '0')
        
        $emotion = Normalize-Emotion $emotionInput
        if (-not $emotion) {
            return $null
        }
        
        # Always enforce .jpg extension
        return "${usn}-${personNum}-${emotion}-${imageNum}.jpg"
    }
    
    return $null
}

function Get-NextDatasetNumber {
    param([string]$datasetPath)
    
    if (-not (Test-Path $datasetPath)) {
        return 1
    }
    
    $existing = Get-ChildItem -Path $datasetPath -Directory | 
                Where-Object { $_.Name -match '^d(\d+)$' } |
                ForEach-Object { [int]$matches[1] } |
                Sort-Object -Descending |
                Select-Object -First 1
    
    if ($existing) {
        return $existing + 1
    }
    return 1
}

# Validate paths
if (-not (Test-Path $ZipsFolder)) {
    Write-Host "ERROR: Zips folder '$ZipsFolder' does not exist!" -ForegroundColor Red
    exit 1
}

# Create dataset folder if needed
if (-not (Test-Path $DatasetFolder)) {
    New-Item -Path $DatasetFolder -ItemType Directory -Force | Out-Null
    Write-Host "[OK] Created dataset folder: $DatasetFolder" -ForegroundColor Green
}

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "ZIP DATASET PROCESSOR" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Zips folder: $ZipsFolder" -ForegroundColor White
Write-Host "Dataset folder: $DatasetFolder" -ForegroundColor White
Write-Host ""

# Get all zip files
$zipFiles = Get-ChildItem -Path $ZipsFolder -Filter "*.zip" | Sort-Object Name

if ($zipFiles.Count -eq 0) {
    Write-Host "No zip files found in $ZipsFolder" -ForegroundColor Yellow
    exit 0
}

Write-Host "Found $($zipFiles.Count) zip file(s)" -ForegroundColor Yellow
Write-Host ""

# Determine starting number
if ($StartNumber -lt 0) {
    $StartNumber = Get-NextDatasetNumber $DatasetFolder
}

Write-Host "Starting from: d$StartNumber" -ForegroundColor Cyan
Write-Host ""

$currentNum = $StartNumber
$totalProcessed = 0
$totalErrors = 0

foreach ($zipFile in $zipFiles) {
    $zipName = $zipFile.Name
    $targetFolder = "d$currentNum"
    $targetPath = Join-Path $DatasetFolder $targetFolder
    
    Write-Host "=" * 70 -ForegroundColor DarkCyan
    Write-Host "Processing: $zipName -> $targetFolder" -ForegroundColor Yellow
    Write-Host "=" * 70 -ForegroundColor DarkCyan
    
    # Create target folder
    if (Test-Path $targetPath) {
        Write-Host "  [WARN] Folder $targetFolder already exists, skipping..." -ForegroundColor Yellow
        $currentNum++
        continue
    }
    
    try {
        New-Item -Path $targetPath -ItemType Directory -Force | Out-Null
        Write-Host "  [OK] Created folder: $targetFolder" -ForegroundColor Green
        
        # Extract zip
        Write-Host "  [INFO] Extracting zip..." -ForegroundColor Cyan
        Expand-Archive -Path $zipFile.FullName -DestinationPath $targetPath -Force
        Write-Host "  [OK] Extraction complete" -ForegroundColor Green
        
        # Get all image files (may be in subdirectories or root)
        $allFiles = Get-ChildItem -Path $targetPath -Recurse -File -Include *.jpg,*.jpeg,*.png,*.bmp,*.JPG,*.JPEG,*.PNG,*.BMP
        
        Write-Host "  [INFO] Found $($allFiles.Count) image files" -ForegroundColor Cyan
        
        # Move files from subdirectories to root first
        foreach ($file in $allFiles) {
            if ($file.DirectoryName -ne $targetPath) {
                $newPath = Join-Path $targetPath $file.Name
                Move-Item -Path $file.FullName -Destination $newPath -Force -ErrorAction SilentlyContinue
            }
        }
        
        # Get files again from root
        $rootFiles = Get-ChildItem -Path "$targetPath\*" -File -Include *.jpg,*.jpeg,*.png,*.bmp,*.JPG,*.JPEG,*.PNG,*.BMP
        
        # Fix filenames
        Write-Host "  [INFO] Fixing filenames..." -ForegroundColor Cyan
        $fixedCount = 0
        $skippedCount = 0
        
        foreach ($file in $rootFiles) {
            $oldName = $file.Name
            $newName = Fix-Filename $oldName
            
            if ($newName -and $newName -ne $oldName) {
                $newPath = Join-Path $file.DirectoryName $newName
                if (Test-Path $newPath) {
                    Write-Host "    [SKIP] $oldName (target exists)" -ForegroundColor DarkYellow
                    $skippedCount++
                } else {
                    Rename-Item -Path $file.FullName -NewName $newName -Force
                    Write-Host "    [FIX] $oldName -> $newName" -ForegroundColor Green
                    $fixedCount++
                }
            }
        }
        
        Write-Host "  [OK] Fixed: $fixedCount, Skipped: $skippedCount" -ForegroundColor Green
        
        # Create emotion folders
        Write-Host "  [INFO] Creating emotion folders..." -ForegroundColor Cyan
        foreach ($emotion in $EmotionFolders) {
            $emotionPath = Join-Path $targetPath $emotion
            if (-not (Test-Path $emotionPath)) {
                New-Item -Path $emotionPath -ItemType Directory -Force | Out-Null
            }
        }
        Write-Host "  [OK] Emotion folders ready" -ForegroundColor Green
        
        # Organize files into emotion folders
        Write-Host "  [INFO] Organizing files into emotion folders..." -ForegroundColor Cyan
        $rootFiles = Get-ChildItem -Path "$targetPath\*" -File -Include *.jpg,*.jpeg,*.png,*.bmp,*.JPG,*.JPEG,*.PNG,*.BMP
        $organizedCount = 0
        
        foreach ($file in $rootFiles) {
            if ($file.Name -match '^.+-\d+-([A-Z]{2})-\d+\.(jpg|jpeg|png|bmp|JPG)$') {
                $emotionCode = $matches[1]
                if ($EmotionToFolder.ContainsKey($emotionCode)) {
                    $emotionFolder = $EmotionToFolder[$emotionCode]
                    $destPath = Join-Path $targetPath $emotionFolder
                    $destFile = Join-Path $destPath $file.Name
                    
                    if (-not (Test-Path $destFile)) {
                        Move-Item -Path $file.FullName -Destination $destFile -Force
                        $organizedCount++
                    }
                }
            }
        }
        
        Write-Host "  [OK] Organized $organizedCount files" -ForegroundColor Green
        
        # Remove non-emotion folders
        Write-Host "  [INFO] Cleaning up extra folders..." -ForegroundColor Cyan
        $allDirs = Get-ChildItem -Path $targetPath -Directory
        $removedCount = 0
        
        foreach ($dir in $allDirs) {
            if ($EmotionFolders -notcontains $dir.Name.ToLower()) {
                Remove-Item -Path $dir.FullName -Recurse -Force
                Write-Host "    [DEL] $($dir.Name)" -ForegroundColor Red
                $removedCount++
            }
        }
        
        if ($removedCount -gt 0) {
            Write-Host "  [OK] Removed $removedCount extra folder(s)" -ForegroundColor Green
        } else {
            Write-Host "  [OK] No extra folders to remove" -ForegroundColor Green
        }
        
        Write-Host "  [SUCCESS] $zipName processed successfully" -ForegroundColor Green
        $totalProcessed++
        
    } catch {
        Write-Host "  [ERROR] Failed to process $zipName : $($_.Exception.Message)" -ForegroundColor Red
        $totalErrors++
    }
    
    Write-Host ""
    $currentNum++
}

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "PROCESSING COMPLETE" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Total zip files processed: $totalProcessed" -ForegroundColor Green
Write-Host "Total errors: $totalErrors" -ForegroundColor Red
Write-Host "Dataset folders created: d$StartNumber to d$($currentNum-1)" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
