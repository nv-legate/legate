#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

echo "Are my wheels there???"

ls -lh

ls -lh final-dist

pip install final-dist/*.whl

echo "Lamest of proof of life tests for legate"
export LEGATE_SHOW_CONFIG=1
export LEGATE_CONFIG="--fbmem 512"
export LEGION_DEFAULT_ARGS="-ll:show_rsrv"
python -c 'import legate.core'
echo "Maybe that worked"
