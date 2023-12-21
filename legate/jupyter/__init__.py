# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

from __future__ import annotations

from typing import TYPE_CHECKING

from legate.jupyter.magic import LegateInfoMagics

if TYPE_CHECKING:
    from IPython import InteractiveShell


def load_ipython_extension(ipython: InteractiveShell) -> None:
    ipython.register_magics(LegateInfoMagics(ipython))


def main() -> int:
    import sys

    from .main import main as _main

    return _main(sys.argv)
