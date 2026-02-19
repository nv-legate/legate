#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

pwd

export RAPIDS_SCRIPT_NAME="test_wheel_linux.bash"

rapids-logger "Are my wheels there???"

ls -lh

ls -lh final-dist

rapids-pip-retry install final-dist/*.whl

rapids-logger "Lamest of proof of life tests for legate"
export LEGATE_SHOW_CONFIG=1
export LEGATE_CONFIG="--fbmem 512"
export LEGION_DEFAULT_ARGS="-ll:show_rsrv"
python -c 'import legate.core'
rapids-logger "Maybe that worked"


rapids-logger "legate-issue"
python -c "from legate.issue import main; main()"

# Attempt to run the tests...
# Temporary workaround for CuPy 14.0.0 regression; see
# https://github.com/nv-legate/legate.internal/issues/3579.
rapids-pip-retry install psutil pytest pytest-mock ipython jupyter_client "cupy!=14.0.0" openmpi h5py

# pytest doesn't truncate output if "CI" is defined in the env:
# https://doc.pytest.org/en/latest/explanation/ci.html
export CI=1

rapids-logger "Running python tests on 1 GPU machine ..."
LEGATE_CONFIG="--fbmem 4000 --gpus 1 --auto-config 0" \
    python -m pytest \
    --color=yes \
    --ignore tests/python/wo_runtime \
    tests/python \
    -s
LEGATE_CONFIG="--fbmem 4000 --gpus 1 --auto-config 0" \
    python -m pytest \
    --color=yes \
    tests/python/wo_runtime \
    -s
