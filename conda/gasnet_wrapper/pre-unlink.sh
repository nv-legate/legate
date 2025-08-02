#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

echo -e "\n\n--------------------- CONDA/GASNET_WRAPPER/PREUNLINK.SH -----------------------\n"

set -eo pipefail
echo "Remove the built artifacts"
# shellcheck disable=SC2154
rm -rf "${CONDA_PREFIX}/gex-wrapper/lib*"
rm -rf "${CONDA_PREFIX}/gex-wrapper/src/build"
