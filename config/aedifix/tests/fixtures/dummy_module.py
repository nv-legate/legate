# SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

MODULE_ATTRIBUTE = "a string"


def function() -> None:
    pass


function.MAGIC_NUMBER = 1  # type: ignore[attr-defined]


class Class:
    MAGIC_ATTR = complex(1, 3)
