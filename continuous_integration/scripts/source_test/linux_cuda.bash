#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export RAPIDS_SCRIPT_NAME="linux_cuda.bash"

# Attempt to extract out important environment variables for 1 GPU testing.
LEGATE_CONFIG="--sysmem 4000 --fbmem 4000 --gpus 1 --auto-config 0"
export LEGATE_CONFIG

# Exclude the tests that did not run cleanly in the same process.
ctest \
  --test-dir build/cpp/tests \
  -E "tests_non_reentrant_with_runtime|tests_non_reentrant_wo_runtime" \
  --output-on-failure \
  --verbose
