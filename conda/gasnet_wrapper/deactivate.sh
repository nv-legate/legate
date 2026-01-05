#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

echo -e "\n\n--------------------- CONDA/GASNET_WRAPPER/DEACTIVATE.SH -----------------------\n"

unset REALM_GASNETEX_WRAPPER
unset GASNET_OFI_SPAWNER
unset FI_CXI_RDZV_THRESHOLD
