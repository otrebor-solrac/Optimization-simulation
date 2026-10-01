#!/usr/bin/env bash
set -e

echo "=== Configurando permisos X11 para Docker ==="
xhost +local:root 2>/dev/null || xhost +local:docker 2>/dev/null || true

export DISPLAY=${DISPLAY:-:1}
echo "Usando DISPLAY=$DISPLAY"

echo "=== Levantando contenedor Sudoku GA Visualizer ==="
docker compose up --build
