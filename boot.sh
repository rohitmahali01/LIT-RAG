#!/bin/bash
# This script ensures that required shared libraries are linked correctly at runtime.

echo "[boot.sh] Linking missing shared libraries for OpenCV..."

# Find the actual location of the required libraries within the Nix store
# and create a symbolic link to them in a standard system path.
# The '|| true' part ensures the script doesn't fail if a link already exists.

ln -sf "$(find /nix/store -name 'libGL.so.1' | head -n 1)" /usr/lib/libGL.so.1 || true
ln -sf "$(find /nix/store -name 'libglib-2.0.so.0' | head -n 1)" /usr/lib/libglib-2.0.so.0 || true
ln -sf "$(find /nix/store -name 'libgthread-2.0.so.0' | head -n 1)" /usr/lib/libgthread-2.0.so.0 || true

echo "[boot.sh] Links created. Starting application..."

# 'exec "$@"' runs the command that was passed as arguments to this script.
# In this case, it will be "uvicorn main:app --host 0.0.0.0 --port $PORT"
exec "$@"
