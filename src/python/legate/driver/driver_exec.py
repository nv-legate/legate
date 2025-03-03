#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

if __name__ == "__main__":
    import sys

    try:
        # imported for effect
        import legate  # noqa: F401
    except ModuleNotFoundError:
        import os

        sys.path.insert(
            0,
            os.path.abspath(  # noqa: PTH100
                os.path.join(  # noqa: PTH118
                    os.path.dirname(__file__),  # noqa: PTH120
                    os.path.pardir,
                    os.path.pardir,
                )
            ),
        )

    from legate import driver

    sys.exit(driver.main())
