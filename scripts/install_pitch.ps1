param(
    [switch]$Rebuild
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Wheel = Join-Path $Root "dist\ids_sentinel_terminal-0.2.1-py3-none-any.whl"
$Python = "python"
$PythonArgs = @()

if (Get-Command py -ErrorAction SilentlyContinue) {
    $Python = "py"
    $PythonArgs = @("-3")
}

if ($Rebuild -or -not (Test-Path -LiteralPath $Wheel)) {
    Push-Location $Root
    try {
        & $Python @PythonArgs scripts\build_python_package.py
        & $Python @PythonArgs scripts\build_distributions.py
    }
    finally {
        Pop-Location
    }
}

if (-not (Test-Path -LiteralPath $Wheel)) {
    throw "Wheel not found: $Wheel"
}

& $Python @PythonArgs -m pip install --user --force-reinstall $Wheel
Write-Host ""
Write-Host "IDS Sentinel Terminal installed." -ForegroundColor Green
Write-Host "Run these commands:" -ForegroundColor Cyan
Write-Host "  ids-sentinel --version"
Write-Host "  ids-sentinel status"
Write-Host "  ids-sentinel gui"
