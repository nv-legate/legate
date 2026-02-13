# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

# This needs to be loaded before anything from ._lib
from .._libucx_loader import _libucx
from ._lib.timing import time

__all__ = ("time",)
