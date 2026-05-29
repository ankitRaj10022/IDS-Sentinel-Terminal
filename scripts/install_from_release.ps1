param(
    [string]$Repo = "ankitRaj10022/IDS-Sentinel-Terminal",
    [string]$Version = "latest",
    [string]$InstallDir = "$env:LOCALAPPDATA\IDS-Sentinel-Terminal"
)

$ErrorActionPreference = "Stop"
$AssetName = "ids-sentinel-terminal-windows.zip"

if ($Version -eq "latest") {
    $Url = "https://github.com/$Repo/releases/latest/download/$AssetName"
}
else {
    $Tag = $Version
    if (-not $Tag.StartsWith("v")) {
        $Tag = "v$Tag"
    }
    $Url = "https://github.com/$Repo/releases/download/$Tag/$AssetName"
}

$TempDir = Join-Path ([System.IO.Path]::GetTempPath()) ("ids-sentinel-release-" + [System.Guid]::NewGuid().ToString("N"))
$ZipPath = Join-Path $TempDir $AssetName
New-Item -ItemType Directory -Path $TempDir | Out-Null

try {
    Write-Host "Downloading IDS Sentinel Terminal from:" -ForegroundColor Cyan
    Write-Host "  $Url"
    Invoke-WebRequest -Uri $Url -OutFile $ZipPath

    if (Test-Path -LiteralPath $InstallDir) {
        Remove-Item -LiteralPath $InstallDir -Recurse -Force
    }
    New-Item -ItemType Directory -Path $InstallDir | Out-Null
    Expand-Archive -LiteralPath $ZipPath -DestinationPath $InstallDir -Force

    $Launcher = Join-Path $InstallDir "ids-sentinel-terminal\ids-sentinel-terminal.cmd"
    if (-not (Test-Path -LiteralPath $Launcher)) {
        throw "Launcher not found after extraction: $Launcher"
    }

    Write-Host ""
    Write-Host "IDS Sentinel Terminal installed." -ForegroundColor Green
    Write-Host "Run it with:" -ForegroundColor Cyan
    Write-Host "  `"$Launcher`" status"
    Write-Host "  `"$Launcher`" gui"
    Write-Host ""
    & $Launcher --version
    & $Launcher status
}
finally {
    if (Test-Path -LiteralPath $TempDir) {
        Remove-Item -LiteralPath $TempDir -Recurse -Force -ErrorAction SilentlyContinue
    }
}
