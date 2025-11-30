# IDS Project Environment Activation Script
# This script loads the conda wrapper and activates the ids-env environment

Write-Host "Loading Conda wrapper..." -ForegroundColor Yellow
. "$env:USERPROFILE\conda-wrapper.ps1"

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
