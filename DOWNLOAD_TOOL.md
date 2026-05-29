# Download IDS Sentinel Terminal Without Cloning The Repo

`pip install ids-sentinel-terminal` will work only after the package is published to PyPI. Until then, the clean public download path is GitHub Releases.

The release assets are tool-only downloads:

```text
ids-sentinel-terminal-windows.zip
ids-sentinel-terminal-linux.tar.gz
ids-sentinel-terminal-macos.tar.gz
ids_sentinel_terminal-0.2.1-py3-none-any.whl
```

## One-Line Install From GitHub Releases

These commands download only the released tool archive. They do not clone the repository.

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/ankitRaj10022/IDS-Sentinel-Terminal/main/scripts/install_from_release.ps1 | iex"
```

Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/ankitRaj10022/IDS-Sentinel-Terminal/main/scripts/install_from_release.sh | sh
```

macOS:

```bash
curl -fsSL https://raw.githubusercontent.com/ankitRaj10022/IDS-Sentinel-Terminal/main/scripts/install_from_release.sh | sh -s -- macos
```

## Direct Download Links

After publishing a GitHub Release, these URLs always point to the latest tool artifacts:

```text
https://github.com/ankitRaj10022/IDS-Sentinel-Terminal/releases/latest/download/ids-sentinel-terminal-windows.zip
https://github.com/ankitRaj10022/IDS-Sentinel-Terminal/releases/latest/download/ids-sentinel-terminal-linux.tar.gz
https://github.com/ankitRaj10022/IDS-Sentinel-Terminal/releases/latest/download/ids-sentinel-terminal-macos.tar.gz
https://github.com/ankitRaj10022/IDS-Sentinel-Terminal/releases/latest/download/ids_sentinel_terminal-0.2.1-py3-none-any.whl
```

## Manual Windows Install

```powershell
$url = "https://github.com/ankitRaj10022/IDS-Sentinel-Terminal/releases/latest/download/ids-sentinel-terminal-windows.zip"
$zip = "$env:TEMP\ids-sentinel-terminal-windows.zip"
$target = "$env:LOCALAPPDATA\IDS-Sentinel-Terminal"
Invoke-WebRequest $url -OutFile $zip
Remove-Item $target -Recurse -Force -ErrorAction SilentlyContinue
Expand-Archive $zip -DestinationPath $target -Force
& "$target\ids-sentinel-terminal\ids-sentinel-terminal.cmd" status
```

## Manual Linux Install

```bash
url="https://github.com/ankitRaj10022/IDS-Sentinel-Terminal/releases/latest/download/ids-sentinel-terminal-linux.tar.gz"
target="$HOME/.local/ids-sentinel-terminal"
tmp="$(mktemp -d)"
mkdir -p "$target"
curl -L "$url" -o "$tmp/ids-sentinel-terminal-linux.tar.gz"
rm -rf "$target/ids-sentinel-terminal"
tar -xzf "$tmp/ids-sentinel-terminal-linux.tar.gz" -C "$target"
"$target/ids-sentinel-terminal/ids-sentinel-terminal" status
```

## Requirement

The portable tool requires Python 3 on the target machine. No Conda, Docker, repo clone, or full source checkout is required.
