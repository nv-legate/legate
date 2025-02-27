#!/usr/bin/env bash

# SPDX-FileCopyrightText: Copyright (c) 2025-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

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
