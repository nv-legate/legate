#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

export RAPIDS_SCRIPT_NAME="test_source_linux.bash"

# Untar the build tree at the same location as it was on the build machine.
tar -zxvf artifacts/build.tgz
ls -lh build

if [[ "${CI:-false}" == "true" ]]; then
  echo "Installing extra packages"
  dnf install -y libatomic python3.11 python3.11-pip

  # Quick and hackish way to get a new cmake version.
  python3 -m venv ~/env
  source ~/env/bin/activate
  pip install cmake
fi

# Attempt to extract out important environment variables for 1 GPU testing.
LEGATE_CONFIG="--sysmem 4000 --fbmem 4000 --gpus 1 --auto-config 0"
export LEGATE_CONFIG

# Exclude the tests that did not run cleanly in the same process.
ctest \
  --test-dir build/cpp/tests \
  -E "tests_non_reentrant_with_runtime|tests_non_reentrant_wo_runtime" \
  --output-on-failure \
  --verbose
