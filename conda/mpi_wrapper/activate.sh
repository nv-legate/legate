#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
echo -e "\n\n--------------------- CONDA/MPI_WRAPPER/ACTIVATE.SH -----------------------\n"

# shellcheck disable=SC2154
wrapper_file=$(find "${CONDA_PREFIX}/mpi-wrapper" -regex ".*/liblegate_mpi_wrapper\.\(so\|dylib\)" -print -quit)
export LEGATE_MPI_WRAPPER="${wrapper_file}"
echo "LEGATE_MPI_WRAPPER=${LEGATE_MPI_WRAPPER}"
