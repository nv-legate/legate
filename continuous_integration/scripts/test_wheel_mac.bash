#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export RAPIDS_SCRIPT_NAME="test_wheel_mac.bash"

echo "Are my wheels there???"

ls -lh

python -m pip install wheelhouse/*.whl

echo "Lamest of proof of life tests for legate"
export LEGATE_SHOW_CONFIG=1
export LEGATE_CONFIG="--fbmem 512"
export LEGION_DEFAULT_ARGS="-ll:show_rsrv"
python -c 'import legate.core'
echo "Maybe that worked"

echo "legate-issue"
python -c "from legate.issue import main; main()"

# Disable tests for now - no MPI, no HDF5 support means they fail.
exit 0

# Attempt to run the tests...
#python -m pip install psutil pytest pytest-mock h5py ipython jupyter_client

# pytest doesn't truncate output if "CI" is defined in the env:
# https://doc.pytest.org/en/latest/explanation/ci.html
#export CI=1

#echo "Running python tests on CPU..."
#LEGATE_CONFIG="--fbmem 4000 --auto-config 0" \
#    python -m pytest \
#    --color=yes \
#    tests/python \
#    -s
