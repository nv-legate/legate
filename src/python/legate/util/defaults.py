# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

__all__ = ("LEGATE_LOG_DIR", "LEGATE_NODES", "LEGATE_RANKS_PER_NODE")

LEGATE_NODES = 1
LEGATE_RANKS_PER_NODE = 1
LEGATE_LOG_DIR = Path.cwd()
