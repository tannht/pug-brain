#!/bin/bash
set -e

DASHBOARD_DIR="$(cd "$(dirname "$0")" && pwd)"
DIST_DIR="$DASHBOARD_DIR/../src/neural_memory/server/static/dist"

echo "Dashboard dir: $DASHBOARD_DIR"
echo "Output dir: $DIST_DIR"

cd "$DASHBOARD_DIR"

# Clear output directory
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR"

# Run vite build directly
echo "Starting vite build..."
node ./node_modules/.bin/vite build

echo "Build complete!"
ls -la "$DIST_DIR"
