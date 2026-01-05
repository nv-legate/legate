# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

__all__ = ("main",)


def main() -> int:  # noqa: D103
    import sys  # noqa: PLC0415

    from .main import main as _main  # noqa: PLC0415

    return _main(sys.argv)
