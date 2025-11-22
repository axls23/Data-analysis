#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Validate and fix filenames to match strict naming convention.

.DESCRIPTION
    Ensures all image files follow the exact convention: <USN>-<PersonNumber>-<Emotion>-<ImageNumber>
    Where:
    - USN starts with "23"
    - PersonNumber is "01", "02", or "03"
    - Emotion is 2 uppercase letters (AN, DI, FE, HA, NE, SA, SU)
    - ImageNumber is "01", "02", or "03"
    
    Corrects:
    - Underscores to hyphens
    - Missing zero-padding
    - Lowercase emotion codes
    - Wrong emotion codes (NU->NE, surprise->SU, etc.)
    - Emotion words to 2-letter codes
    - Non-standard USN format

.PARAMETER DatasetPath
    Path to the dataset folder to validate/fix

.PARAMETER DryRun
    Show what would be changed without making changes

.PARAMETER Recurse
    Process subdirectories recursively

.EXAMPLE
    .\validate_naming.ps1 -DatasetPath "dataset/d22" -DryRun -Recurse

.EXAMPLE
    .\validate_naming.ps1 -DatasetPath "dataset/d22" -Recurse
#>

param(
    [Parameter(Mandatory=$false, HelpMessage="Path to the dataset folder")]
    [string]$DatasetPath = "dataset",
    
    [Parameter(Mandatory=$false)]
    [switch]$DryRun,
    
    [Parameter(Mandatory=$false)]
    [switch]$Recurse
)

# Emotion mappings (case-insensitive)
$EmotionMap = @{
    'angry' = 'AN'
    'AN' = 'AN'
    'disgust' = 'DI'
    'DI' = 'DI'
    'fear' = 'FE'
    'FE' = 'FE'
    'happy' = 'HA'
    'HA' = 'HA'
    'happiness' = 'HA'
    'neutral' = 'NE'
    'NE' = 'NE'
    'NU' = 'NE'
    'sad' = 'SA'
    'SA' = 'SA'
    'surprised' = 'SU'
    'surprise' = 'SU'
    'SU' = 'SU'
}

$ValidEmotions = @('AN', 'DI', 'FE', 'HA', 'NE', 'SA', 'SU')
$ValidPersonNumbers = @('01', '02', '03')
$ValidImageNumbers = @('01', '02', '03')

function Normalize-Emotion {
    param([string]$emotion)
    
    # Try exact match first (case-insensitive)
    if ($EmotionMap.ContainsKey($emotion.ToLower())) {
        return $EmotionMap[$emotion.ToLower()]
    }
    if ($EmotionMap.ContainsKey($emotion.ToUpper())) {
        return $EmotionMap[$emotion.ToUpper()]
    }
    if ($EmotionMap.ContainsKey($emotion)) {
        return $EmotionMap[$emotion]
    }
    
    # Try first 2 letters uppercase
    if ($emotion.Length -ge 2) {
        $twoLetter = $emotion.Substring(0, 2).ToUpper()
        if ($ValidEmotions -contains $twoLetter) {
            return $twoLetter
        }
    }

    # Fallback: First letter match
    if ($emotion.Length -ge 1) {
        $first = $emotion.Substring(0, 1).ToUpper()
        switch ($first) {
            'A' { return 'AN' }
            'D' { return 'DI' }
            'F' { return 'FE' }
            'H' { return 'HA' }
            'N' { return 'NE' }
            'S' { 
                # Check for SU (Surprised), otherwise default to SA (Sad)
                if ($emotion.Length -ge 2 -and $emotion.Substring(0, 2).ToUpper() -eq 'SU') {
                    return 'SU'
                }
                return 'SA'
            }
        }
    }
    
    return $null
}

function Get-CorrectedFilename {
    param(
        [string]$filename,
        [ref]$issues
    )
    
    $issues.Value = @()
    $base = [System.IO.Path]::GetFileNameWithoutExtension($filename)
    $ext = [System.IO.Path]::GetExtension($filename)
    $originalExt = $ext
    
    # Normalize the extension to .jpg
    if ($ext -match '\.(jpeg|png|bmp|JPEG|PNG|BMP)$') {
        $ext = '.jpg'
    }
    
    # Remove spaces
    $normalized = $base -replace '\s', ''

    # First, convert underscores to hyphens
    $normalized = $normalized -replace '_', '-'
    
    # Normalize multiple hyphens to single hyphen
    $normalized = $normalized -replace '--+', '-'
    
    # Try to extract components using a pattern that searches within the string
    # Look for USN ending with 3 digits (or O's), followed by 3 groups separated by hyphens
    # We allow 'O' in place of digits for PersonNum and ImageNum too
    if ($normalized -match '([A-Za-z0-9]*[\dO]{3})-([\dO]+)-([A-Za-z]+)-([\dO]+)') {
        $usn = $matches[1]
        # Fix O -> 0 in USN suffix (last 3 chars)
        if ($usn.Length -ge 3) {
            $suffix = $usn.Substring($usn.Length - 3).Replace('O', '0')
            $prefix = $usn.Substring(0, $usn.Length - 3)
            $usn = $prefix + $suffix
        }

        $personNum = $matches[2].Replace('O', '0')
        $emotionInput = $matches[3]
        $imageNum = $matches[4].Replace('O', '0')
        
        $hasIssues = $false
        
        # Check USN starts with 23
        if (-not $usn.StartsWith('23')) {
            $usn = "23$usn"
            $issues.Value += "Prepended '23' to USN"
            $hasIssues = $true
        }
        
        # Check person number
        $personNumPadded = $personNum.PadLeft(2, '0')
        if ($ValidPersonNumbers -notcontains $personNumPadded) {
            $issues.Value += "Invalid person number: $personNum (must be 01-03)"
            return $null
        }
        if ($personNum -ne $personNumPadded) {
            $issues.Value += "Person number needs padding: $personNum -> $personNumPadded"
            $hasIssues = $true
        }
        
        # Check emotion
        $emotionNormalized = Normalize-Emotion $emotionInput
        if (-not $emotionNormalized) {
            $issues.Value += "Unknown emotion: $emotionInput"
            return $null
        }
        if ($emotionInput -ne $emotionNormalized) {
            $issues.Value += "Emotion needs normalization: $emotionInput -> $emotionNormalized"
            $hasIssues = $true
        }
        
        # Check image number
        $imageNumPadded = $imageNum.PadLeft(2, '0')
        if ($ValidImageNumbers -notcontains $imageNumPadded) {
            $issues.Value += "Invalid image number: $imageNum (must be 01-03)"
            return $null
        }
        if ($imageNum -ne $imageNumPadded) {
            $issues.Value += "Image number needs padding: $imageNum -> $imageNumPadded"
            $hasIssues = $true
        }
        
        # Check if original had underscores or spaces or extra text
        $correctedName = "${usn}-${personNumPadded}-${emotionNormalized}-${imageNumPadded}.jpg"
        
        if ($filename -ne $correctedName) {
             if ($base -match '_') { $issues.Value += "Uses underscores" }
             if ($base -match '\s') { $issues.Value += "Contains spaces" }
             if ($originalExt -ne '.jpg') { $issues.Value += "Extension normalized to .jpg" }
             
             # If we are here, it means the filename is different, so we are renaming/fixing
             $issues.Value += "Enforcing strict format (removing extra text/formatting)"
             return $correctedName
        }
        
        return $null  # Already correct
    }
    
    # Doesn't match pattern - try to parse it anyway
    $issues.Value += "Doesn't match expected pattern <USN>-<PersonNum>-<Emotion>-<ImageNum>"
    return $null
}

# Validate path
if (-not (Test-Path $DatasetPath)) {
    Write-Host "ERROR: Path '$DatasetPath' does not exist!" -ForegroundColor Red
    exit 1
}

$DatasetPath = Resolve-Path $DatasetPath

# Auto-enable recursion if targeting the root dataset folder
if (-not $Recurse.IsPresent -and (Split-Path -Leaf $DatasetPath) -eq "dataset") {
    Write-Host "Targeting root 'dataset' folder - Enabling recursive mode." -ForegroundColor Cyan
    $Recurse = $true
}

Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "NAMING CONVENTION VALIDATOR & FIXER" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "Target: $DatasetPath" -ForegroundColor White
Write-Host "Recurse: $($Recurse.IsPresent) | DryRun: $($DryRun.IsPresent)" -ForegroundColor White
Write-Host ""
Write-Host "Strict format (hyphen-separated only): ^23[A-Z0-9]+-0[1-3]-[A-Z]{2}-0[1-3]\.jpg$" -ForegroundColor Yellow
Write-Host "  USN: Starts with '23' (e.g., 23BTRCL202)" -ForegroundColor White
Write-Host "  PersonNumber: 01, 02, or 03" -ForegroundColor White
Write-Host "  Emotion: AN, DI, FE, HA, NE, SA, SU (uppercase)" -ForegroundColor White
Write-Host "  ImageNumber: 01, 02, or 03" -ForegroundColor White
Write-Host "  Extension: .jpg (normalized from any image format)" -ForegroundColor White
Write-Host ""

# Collect folders
$folders = @()
if ($Recurse) {
    $folders = Get-ChildItem -Path $DatasetPath -Directory -Recurse
    $folders += Get-Item $DatasetPath
} else {
    # Check root
    $rootImages = Get-ChildItem -Path "$DatasetPath\*" -File -Include *.jpg,*.jpeg,*.png,*.bmp,*.JPG,*.JPEG,*.PNG,*.BMP -ErrorAction SilentlyContinue
    if ($rootImages.Count -gt 0) {
        $folders += Get-Item $DatasetPath
    }
    # Check subdirectories
    $folders += Get-ChildItem -Path $DatasetPath -Directory
}

$totalFiles = 0
$totalCorrect = 0
$totalFixed = 0
$totalSkipped = 0
$totalErrors = 0
$totalInvalid = 0

foreach ($folder in $folders) {
    $images = Get-ChildItem -Path "$($folder.FullName)\*" -File -Include *.jpg,*.jpeg,*.png,*.bmp,*.JPG,*.JPEG,*.PNG,*.BMP -ErrorAction SilentlyContinue
    
    if ($images.Count -eq 0) { continue }
    
    $folderName = $folder.FullName.Replace($DatasetPath, '').TrimStart('\')
    if ([string]::IsNullOrEmpty($folderName)) { $folderName = "root" }
    
    # First pass: check if there are any issues
    $hasIssues = $false
    foreach ($image in $images) {
        $issues = $null
        $correctedName = Get-CorrectedFilename $image.Name ([ref]$issues)
        if ($null -ne $correctedName -or $issues.Count -gt 0) {
            $hasIssues = $true
            break
        }
    }
    
    if (-not $hasIssues) { continue }
    
    Write-Host "Processing: $folderName/ ($($images.Count) files)" -ForegroundColor Yellow
    Write-Host "-" * 70 -ForegroundColor Cyan
    
    foreach ($image in $images) {
        $totalFiles++
        $oldName = $image.Name
        
        $issues = $null
        $correctedName = Get-CorrectedFilename $oldName ([ref]$issues)
        
        if ($null -eq $correctedName) {
            if ($issues.Count -eq 0) {
                # Already correct - don't print
                $totalCorrect++
            } else {
                # Invalid - can't be fixed
                Write-Host "  [✗] $oldName" -ForegroundColor Red
                foreach ($issue in $issues) {
                    Write-Host "      └─ $issue" -ForegroundColor Red
                }
                $totalInvalid++
            }
            continue
        }
        
        # Needs fixing
        if ($DryRun) {
            Write-Host "  [DRY] $oldName -> $correctedName" -ForegroundColor Yellow
            foreach ($issue in $issues) {
                Write-Host "        └─ $issue" -ForegroundColor DarkYellow
            }
            $totalFixed++
            continue
        }
        
        $newPath = Join-Path $image.DirectoryName $correctedName
        if (Test-Path $newPath) {
            Write-Host "  [SKIP] $oldName -> $correctedName (exists)" -ForegroundColor DarkYellow
            $totalSkipped++
            continue
        }
        
        try {
            Rename-Item -Path $image.FullName -NewName $correctedName -Force
            Write-Host "  [FIX] $oldName -> $correctedName" -ForegroundColor Green
            foreach ($issue in $issues) {
                Write-Host "        └─ Fixed: $issue" -ForegroundColor DarkGreen
            }
            $totalFixed++
        } catch {
            Write-Host "  [ERR] $oldName : $($_.Exception.Message)" -ForegroundColor Red
            $totalErrors++
        }
    }
    Write-Host ""
}


Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "SUMMARY" -ForegroundColor Cyan
Write-Host "=" * 70 -ForegroundColor Cyan
Write-Host "  Total files scanned: $totalFiles" -ForegroundColor White
Write-Host "  Already correct: $totalCorrect" -ForegroundColor Green
Write-Host "  Fixed: $totalFixed" -ForegroundColor Green
Write-Host "  Skipped (exists): $totalSkipped" -ForegroundColor Yellow
Write-Host "  Invalid (can't fix): $totalInvalid" -ForegroundColor Red
Write-Host "  Errors: $totalErrors" -ForegroundColor Red
if ($DryRun) {
    Write-Host ""
    Write-Host "  ⚠ DRY RUN - No changes made" -ForegroundColor Yellow
    Write-Host "  Remove -DryRun flag to apply fixes" -ForegroundColor Yellow
}
Write-Host "=" * 70 -ForegroundColor Cyan
