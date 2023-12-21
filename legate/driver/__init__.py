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

from .config import Config
from .driver import LegateDriver, CanonicalDriver
from .launcher import Launcher


def main() -> int:
    import os, shlex, sys

    from .main import legate_main as _main

    # A little explanation. We want to encourage configuration options be
    # passed via LEGATE_CONFIG, in order to be considerate to user scripts.
    # But we still need to accept actual command line args for comaptibility,
    # and those should also take precedences. Here we splice the options from
    # LEGATE_CONFIG in before sys.argv, and take advantage of the fact that if
    # there are any options repeated in both places, argparse will use the
    # latter (i.e. the actual command line provided ones).
    env_args = shlex.split(os.environ.get("LEGATE_CONFIG", ""))
    argv = sys.argv[:1] + env_args + sys.argv[1:]

    return _main(argv)
