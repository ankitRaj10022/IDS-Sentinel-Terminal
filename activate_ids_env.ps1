# IDS Project Environment Activation Script
# This script loads Conda and activates the ids-env environment.
$ErrorActionPreference = "Stop"

function Get-CondaHook {
    $candidates = @(
        "$env:USERPROFILE\conda-wrapper.ps1",
        "$env:USERPROFILE\anaconda3\shell\condabin\conda-hook.ps1",
        "$env:USERPROFILE\miniconda3\shell\condabin\conda-hook.ps1",
        "$env:LOCALAPPDATA\miniconda3\shell\condabin\conda-hook.ps1",
        "C:\ProgramData\anaconda3\shell\condabin\conda-hook.ps1",
        "C:\ProgramData\miniconda3\shell\condabin\conda-hook.ps1"
    )

    foreach ($candidate in $candidates) {
        if (Test-Path -LiteralPath $candidate) {
            return $candidate
        }
    }

    return $null
}

Write-Host "Loading Conda wrapper..." -ForegroundColor Yellow
$condaHook = Get-CondaHook
if (-not $condaHook) {
    throw "Could not find Conda. Install Miniconda or Anaconda, then reopen PowerShell and run this script again."
}

. $condaHook

if (-not (Get-Command conda -ErrorAction SilentlyContinue)) {
    throw "Conda was found at '$condaHook', but the 'conda' command did not load correctly."
}

Write-Host ""
Write-Host "Activating IDS environment..." -ForegroundColor Yellow
conda activate ids-env

Write-Host ""
Write-Host "IDS environment activated!" -ForegroundColor Green
Write-Host ""
Write-Host "You can now run your Intrusion Detection System scripts:" -ForegroundColor Cyan
Write-Host "  - For DNN training: python dnn1.py" -ForegroundColor White
Write-Host "  - For accuracy testing: python dnn1acc.py" -ForegroundColor White
Write-Host "  - For classical ML: python ../all.py" -ForegroundColor White
Write-Host ""
Write-Host "Installed packages:" -ForegroundColor Yellow
Write-Host "  - TensorFlow 2.18.1" -ForegroundColor Green
Write-Host "  - Keras 3.10.0" -ForegroundColor Green
Write-Host "  - scikit-learn 1.5.1" -ForegroundColor Green
Write-Host "  - pandas 2.3.1" -ForegroundColor Green
Write-Host "  - numpy 1.26.4" -ForegroundColor Green
Write-Host "  - h5py 3.14.0" -ForegroundColor Green
Write-Host "  - matplotlib 3.9.2" -ForegroundColor Green
Write-Host "  - seaborn 0.13.2" -ForegroundColor Green
