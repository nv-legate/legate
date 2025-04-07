#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export RAPIDS_SCRIPT_NAME="test_sheel_linux.bash"

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
