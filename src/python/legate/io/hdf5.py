# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# This module is the "public" interface for this function, so import it purely
# to re-export it.
from ._lib.hdf5.hdf5_interface import from_file, to_file

__all__ = ("from_file", "to_file")
