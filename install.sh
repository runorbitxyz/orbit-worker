#!/bin/sh
# ORBIT operator installer: downloads the worker (one readable Python file), pairs it with your account in the
# browser, and keeps it running in the background (systemd user service on Linux, launchd agent on macOS).
#   curl -fsSL https://api.runorbit.xyz/install.sh | sh
#   curl -fsSL https://api.runorbit.xyz/install.sh | sh -s -- --private      # a rig that serves only you, free
# Remove it later with: python3 ~/.orbit/orbit_worker.py --uninstall-service
set -eu

main() {
  ORBIT_URL="${ORBIT_URL:-https://api.runorbit.xyz}"
  DIR="$HOME/.orbit"
  PY="$(command -v python3 || command -v python || true)"
  if [ -z "$PY" ]; then
    echo "ORBIT needs Python 3.9+ (preinstalled on macOS and most Linux distributions)." >&2
    exit 1
  fi
  mkdir -p "$DIR"
  chmod 700 "$DIR"
  echo "-> Downloading the ORBIT worker to $DIR/orbit_worker.py"
  curl -fsSL --proto '=https,http' "$ORBIT_URL/worker/orbit_worker.py" -o "$DIR/orbit_worker.py.part"
  mv "$DIR/orbit_worker.py.part" "$DIR/orbit_worker.py"
  exec "$PY" "$DIR/orbit_worker.py" --orbit-url "$ORBIT_URL" --setup "$@"
}

# Everything runs from main(), so a download cut off midway runs nothing.
main "$@"
