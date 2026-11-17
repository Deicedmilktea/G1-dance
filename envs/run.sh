#!/usr/bin/env bash
set -euo pipefail
project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)"
if (( $# < 2 )); then
    echo '用法: ./envs/run.sh motion|training <Python 参数>' >&2
    exit 2
fi
mode="$1"
shift
case "$mode" in
    motion)
        interpreter="$project_dir/.venv-motion/bin/python"
        export PYTHONPATH="$project_dir/third_party/GVHMR${PYTHONPATH:+:$PYTHONPATH}"
        ;;
    training)
        interpreter="$project_dir/envs/training/.venv/bin/python"
        ;;
    *)
        echo '环境必须是 motion 或 training' >&2
        exit 2
        ;;
esac
export G1_PROJECT_ROOT="$project_dir"
export G1_SUPERMIMIC_SOURCE=nju
export UV_CACHE_DIR="$project_dir/.runtime/uv-cache"
export UV_PYTHON_INSTALL_DIR="$project_dir/.runtime/python"
export XDG_CACHE_HOME="$project_dir/.runtime/cache"
export XDG_CONFIG_HOME="$project_dir/.runtime/config"
export MPLCONFIGDIR="$project_dir/.runtime/matplotlib"
export YOLO_CONFIG_DIR="$project_dir/.runtime/config/ultralytics"
export PYTHONDONTWRITEBYTECODE=1
export PATH="$project_dir/.runtime/bin:$PATH"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$MPLCONFIGDIR" "$YOLO_CONFIG_DIR"
exec "$interpreter" "$@"
