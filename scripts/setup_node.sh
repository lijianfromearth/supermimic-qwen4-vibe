#!/usr/bin/env bash
# Build the training venv on a GPU node.
#
# pypi.org and github.com are unreachable there, so the interpreter must already
# be installed (uv never downloads one) and every index comes from pyproject.toml:
# tuna mirrors PyPI, download.pytorch.org serves torch cu130, pypi.nvidia.com
# serves warp-lang.  mjlab is vendored because its GitHub remote is blocked.
set -euo pipefail
cd "$(dirname "$0")/.."

export UV_PYTHON_DOWNLOADS=never
export UV_HTTP_TIMEOUT=180

uv venv --python 3.11 .venv
uv sync --python 3.11 --no-progress
uv run --no-sync python -c '
import sys, torch, mujoco, mujoco_warp, warp, mjlab, supermimic
print("python", sys.version.split()[0])
print("torch", torch.__version__, "cuda", torch.version.cuda)
print("mujoco", mujoco.__version__, "mujoco_warp", mujoco_warp.__version__)
print("warp", warp.__version__)
print("mjlab", mjlab.__file__)
print("supermimic", supermimic.__file__)
print("gpu count", torch.cuda.device_count())
assert torch.cuda.is_available(), "no CUDA device"
print("gpu0", torch.cuda.get_device_name(0), "sm", torch.cuda.get_device_capability(0))
'