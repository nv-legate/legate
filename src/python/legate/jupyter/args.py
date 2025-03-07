# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from legate.util import shared_args as sa

__all__ = ("parser",)


parser = ArgumentParser(
    description="Install a Legate Jupyter Kernel",
    allow_abbrev=False,
    formatter_class=ArgumentDefaultsHelpFormatter,
)

kernel = parser.add_argument_group("Kernel configuration")

kernel.add_argument(
    "--user",
    action="store_true",
    default=True,
    dest="user",
    help="Install the kernel in user home directory",
)

kernel.add_argument(
    "--name",
    default="Legate_SM_GPU",
    dest="spec_name",
    help="A name for the kernel spec",
)

kernel.add_argument(
    "--display-name",
    default=None,
    dest="display_name",
    help="A display name for the kernel (if not provided, --name is used)",
)

kernel.add_argument(
    "--prefix",
    default=None,
    dest="prefix",
    help="A prefix to install the kernel into",
)


multi_node = parser.add_argument_group("Multi-node configuration")
multi_node.add_argument(sa.NODES.name, **sa.NODES.kwargs)
multi_node.add_argument(sa.RANKS_PER_NODE.name, **sa.RANKS_PER_NODE.kwargs)
multi_node.add_argument(sa.LAUNCHER.name, **sa.LAUNCHER.kwargs)
multi_node.add_argument(sa.LAUNCHER_EXTRA.name, **sa.LAUNCHER_EXTRA.kwargs)


core = parser.add_argument_group("Core allocation")
core.add_argument(sa.CPUS.name, **sa.CPUS.kwargs)
core.add_argument(sa.GPUS.name, **sa.GPUS.kwargs)
core.add_argument(sa.OMPS.name, **sa.OMPS.kwargs)
core.add_argument(sa.OMPTHREADS.name, **sa.OMPTHREADS.kwargs)
core.add_argument(sa.UTILITY.name, **sa.UTILITY.kwargs)


memory = parser.add_argument_group("Memory allocation")
memory.add_argument(sa.SYSMEM.name, **sa.SYSMEM.kwargs)
memory.add_argument(sa.NUMAMEM.name, **sa.NUMAMEM.kwargs)
memory.add_argument(sa.FBMEM.name, **sa.FBMEM.kwargs)
memory.add_argument(sa.ZCMEM.name, **sa.ZCMEM.kwargs)
memory.add_argument(sa.REGMEM.name, **sa.REGMEM.kwargs)


info = parser.add_argument_group("Informational")

info.add_argument(
    "-v",
    "--verbose",
    dest="verbose",
    action="count",
    default=0,
    help="Display verbose output. Use -vv for even more output (test stdout)",
)

info.add_argument(
    "--color",
    dest="color",
    action="store_true",
    required=False,
    help="Whether to use color terminal output (if rich is installed)",
)
