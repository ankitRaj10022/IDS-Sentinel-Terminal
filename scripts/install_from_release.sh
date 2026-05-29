#!/usr/bin/env sh
set -eu

platform="${1:-linux}"
repo="${IDS_SENTINEL_REPO:-ankitRaj10022/IDS-Sentinel-Terminal}"
version="${IDS_SENTINEL_VERSION:-latest}"
install_dir="${IDS_SENTINEL_INSTALL_DIR:-$HOME/.local/ids-sentinel-terminal}"

case "$platform" in
  linux)
    asset="ids-sentinel-terminal-linux.tar.gz"
    ;;
  macos|darwin)
    asset="ids-sentinel-terminal-macos.tar.gz"
    ;;
  *)
    echo "Unsupported platform: $platform" >&2
    echo "Use: install_from_release.sh [linux|macos]" >&2
    exit 1
    ;;
esac

if [ "$version" = "latest" ]; then
  url="https://github.com/$repo/releases/latest/download/$asset"
else
  case "$version" in
    v*) tag="$version" ;;
    *) tag="v$version" ;;
  esac
  url="https://github.com/$repo/releases/download/$tag/$asset"
fi

tmp_dir="$(mktemp -d)"
archive="$tmp_dir/$asset"

cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

echo "Downloading IDS Sentinel Terminal from:"
echo "  $url"

if command -v curl >/dev/null 2>&1; then
  curl -fsSL "$url" -o "$archive"
elif command -v wget >/dev/null 2>&1; then
  wget -O "$archive" "$url"
else
  echo "curl or wget is required to download the tool." >&2
  exit 1
fi

mkdir -p "$install_dir"
rm -rf "$install_dir/ids-sentinel-terminal"
tar -xzf "$archive" -C "$install_dir"

launcher="$install_dir/ids-sentinel-terminal/ids-sentinel-terminal"
if [ ! -x "$launcher" ]; then
  chmod +x "$launcher"
fi

echo
echo "IDS Sentinel Terminal installed."
echo "Run it with:"
echo "  $launcher status"
echo "  $launcher gui"
echo
"$launcher" --version
"$launcher" status
