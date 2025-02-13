# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from pathlib import Path

__all__ = ("LEGATE_LOG_DIR", "LEGATE_NODES", "LEGATE_RANKS_PER_NODE")

LEGATE_NODES = 1
LEGATE_RANKS_PER_NODE = 1
LEGATE_LOG_DIR = Path.cwd()
