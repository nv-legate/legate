#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

echo -e "\n\n--------------------- CONDA/GASNET_WRAPPER/ACTIVATE.SH -----------------------\n"

# shellcheck disable=SC2154
wrapper_file=$(find "${CONDA_PREFIX}/gex-wrapper" \( -name "librealm_gex_wrapper.so" -o -name "librealm_gex_wrapper.dylib" \) -print -quit)
export REALM_GASNETEX_WRAPPER="${wrapper_file}"
# WAR for:
# https://gasnet-bugs.lbl.gov/bugzilla/show_bug.cgi?id=4638
export GASNET_OFI_SPAWNER=mpi
export FI_CXI_RDZV_THRESHOLD=256

echo "REALM_GASNETEX_WRAPPER=${REALM_GASNETEX_WRAPPER}"
echo "GASNET_OFI_SPAWNER=${GASNET_OFI_SPAWNER}"
echo "FI_CXI_RDZV_THRESHOLD=${FI_CXI_RDZV_THRESHOLD}"
