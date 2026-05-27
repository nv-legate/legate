#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
echo -e "\n\n--------------------- CONDA/MPI_WRAPPER/ACTIVATE.SH -----------------------\n"

wrapper_file=""
# shellcheck disable=SC2154
wrapper_dir="${CONDA_PREFIX}/mpi-wrapper"
if [[ -d "${wrapper_dir}" ]]; then
  wrapper_file=$(
    find "${wrapper_dir}" \
      -regex ".*/liblegate_mpi_wrapper\.\(so\|dylib\)" \
      -print \
      -quit
  )
fi
export LEGATE_MPI_WRAPPER="${wrapper_file}"
echo "LEGATE_MPI_WRAPPER=${LEGATE_MPI_WRAPPER}"
