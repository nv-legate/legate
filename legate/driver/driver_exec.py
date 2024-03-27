#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
if __name__ == "__main__":
    import sys

    try:
        # imported for effect
        import legate  # noqa F401
    except ModuleNotFoundError:
        import os

        sys.path.insert(
            0,
            os.path.abspath(
                os.path.join(
                    os.path.dirname(__file__), os.path.pardir, os.path.pardir
                )
            ),
        )

    import legate.driver as driver

    sys.exit(driver.main())
