#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# shellcheck disable=SC2154
cat << EOF >> "${PREFIX}/.messages.txt"

To finish configuring the Legate MPI wrapper, activate your environment and run ${CONDA_PREFIX}/mpi-wrapper/build-mpi-wrapper.sh

EOF
