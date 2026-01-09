#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
echo -e "\n\n--------------------- CONDA/MPI_WRAPPER/PREUNLINK.SH -----------------------\n"

set -eo pipefail
echo "Remove the built artifacts"
# shellcheck disable=SC2154
MPI_WRAPPER_DIR="${CONDA_PREFIX}/mpi-wrapper"
rm -rf "${MPI_WRAPPER_DIR}/include" "${MPI_WRAPPER_DIR}/lib*"
