# IDS Sentinel Terminal Pitch Install Guide

`pip install ids-sentinel-terminal` only works after the package is published on PyPI. As of this build, PyPI has no public `ids-sentinel-terminal` package, so pip correctly reports:

```text
ERROR: No matching distribution found for ids-sentinel-terminal
```

Use one of the install paths below for your jury demo.

## Best Option: Download Only The Released Tool

After publishing a GitHub Release, another machine can install only the tool archive without cloning the repository.

Windows PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -Command "irm https://raw.githubusercontent.com/ankitRaj10022/IDS-Sentinel-Terminal/main/scripts/install_from_release.ps1 | iex"
```

Linux:

```bash
curl -fsSL https://raw.githubusercontent.com/ankitRaj10022/IDS-Sentinel-Terminal/main/scripts/install_from_release.sh | sh
```

These commands download the latest release ZIP/TAR asset only.

## Option 1: Install From GitHub

This does not require manually cloning the repository. `pip` downloads the package source into a temporary build folder and installs the `ids-sentinel` commands.

Windows PowerShell:

```powershell
py -3 -m pip install --user --upgrade git+https://github.com/ankitRaj10022/IDS-Sentinel-Terminal.git
ids-sentinel --version
ids-sentinel status
```

Linux/macOS with `pipx`:

```bash
pipx install --force git+https://github.com/ankitRaj10022/IDS-Sentinel-Terminal.git
ids-sentinel status
```

Linux/macOS with a normal Python environment:

```bash
python3 -m pip install --user --upgrade git+https://github.com/ankitRaj10022/IDS-Sentinel-Terminal.git
export PATH="$HOME/.local/bin:$PATH"
ids-sentinel --version
ids-sentinel status
```

## Option 2: Install The Local Wheel

Use this when you are presenting from this project folder or carrying the `dist` folder on a USB drive.

Windows PowerShell:

```powershell
py -3 -m pip install --user --force-reinstall .\dist\ids_sentinel_terminal-0.2.1-py3-none-any.whl
ids-sentinel --version
ids-sentinel status
```

Linux/macOS from the repo folder with `pipx`:

```bash
pipx install --force ./dist/ids_sentinel_terminal-0.2.1-py3-none-any.whl
ids-sentinel status
```

Linux/macOS from the repo folder with `pip`:

```bash
python3 -m pip install --user --force-reinstall ./dist/ids_sentinel_terminal-0.2.1-py3-none-any.whl
export PATH="$HOME/.local/bin:$PATH"
ids-sentinel --version
ids-sentinel status
```

WSL from the Windows repo path with `pipx`:

```bash
pipx install --force /mnt/c/Users/danny/Desktop/Intrusion-Detection-Systems-master/IDS-MachineLearning/dist/ids_sentinel_terminal-0.2.1-py3-none-any.whl
ids-sentinel status
```

WSL from the Windows repo path with `pip`:

```bash
python3 -m pip install --user --force-reinstall /mnt/c/Users/danny/Desktop/Intrusion-Detection-Systems-master/IDS-MachineLearning/dist/ids_sentinel_terminal-0.2.1-py3-none-any.whl
export PATH="$HOME/.local/bin:$PATH"
ids-sentinel --version
ids-sentinel status
```

If the command is installed but not found, add the Python user scripts folder to `PATH`.

Windows:

```powershell
$scripts = py -3 -c "import pathlib, site; print(pathlib.Path(site.USER_BASE) / 'Scripts')"
$env:Path += ";$scripts"
```

Linux/macOS:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

## Option 3: Portable Tool, No Install

Use this for the safest live demo. Extract the archive and run the included launcher.

Windows:

```powershell
Expand-Archive .\dist\ids-sentinel-terminal-windows.zip -DestinationPath .\ids-sentinel-demo -Force
.\ids-sentinel-demo\ids-sentinel-terminal\ids-sentinel-terminal.cmd status
.\ids-sentinel-demo\ids-sentinel-terminal\ids-sentinel-terminal.cmd scan kddtest.csv --limit 5000
```

Linux/macOS:

```bash
mkdir -p ids-sentinel-demo
tar -xzf dist/ids-sentinel-terminal-linux.tar.gz -C ids-sentinel-demo
./ids-sentinel-demo/ids-sentinel-terminal/ids-sentinel-terminal status
./ids-sentinel-demo/ids-sentinel-terminal/ids-sentinel-terminal scan kddtest.csv --limit 5000
```

## Demo Commands

```bash
ids-sentinel status
ids-sentinel traffic
ids-sentinel attacks
ids-sentinel malware --limit 5000
ids-sentinel scan kddtest.csv --limit 5000
ids-sentinel reports
ids-sentinel ports --limit 20
ids-sentinel gui
```

## PyPI Publication

After the package is published to PyPI, this command will work:

```bash
pip install ids-sentinel-terminal
```

Until then, use the GitHub URL, wheel file, or portable archive above.
