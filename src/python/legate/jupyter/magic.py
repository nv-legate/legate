# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from IPython.core.magic import Magics, line_magic, magics_class
from jupyter_client.kernelspec import KernelSpecManager, NoSuchKernel
from rich import print as rich_print
from rich.console import Group

from legate.jupyter.kernel import (
    LEGATE_JUPYTER_KERNEL_SPEC_KEY,
    LEGATE_JUPYTER_METADATA_KEY,
    LegateMetadata,
)
from legate.util.ui import table

if TYPE_CHECKING:
    from IPython import InteractiveShell


core = {
    "cpus": "CPUs to use per rank",
    "gpus": "GPUs to use per rank",
    "omps": "OpenMP groups to use per rank",
    "ompthreads": "Threads per OpenMP group",
    "utility": "Utility processors per rank",
}

memory = {
    "sysmem": "DRAM memory per rank (in MBs)",
    "numamem": "DRAM memory per NUMA domain per rank (in MBs)",
    "fbmem": "Framebuffer memory per GPU (in MBs)",
    "zcmem": "Zero-copy memory per rank (in MBs)",
    "regmem": "Registered CPU-side pinned memory per rank (in MBs)",
}


class LegateInfo:
    config: LegateMetadata

    def __init__(self) -> None:
        if LEGATE_JUPYTER_KERNEL_SPEC_KEY not in os.environ:
            msg = "Cannot determine currently running kernel"
            raise RuntimeError(msg)

        spec_name = os.environ[LEGATE_JUPYTER_KERNEL_SPEC_KEY]

        try:
            spec = KernelSpecManager().get_kernel_spec(spec_name)
        except NoSuchKernel:
            msg = f"Cannot find a Legate Jupyter kernel named {spec_name!r}"
            raise RuntimeError(msg)

        self.spec_name = spec_name
        self.config = spec.metadata[LEGATE_JUPYTER_METADATA_KEY]

    @property
    def ui(self) -> Group:  # noqa: D102
        nodes = self.config["multi_node"]["nodes"]
        header = f"Kernel {self.spec_name!r} configured for {nodes} node(s)"
        core_table = {
            desc: self.config["core"][field] for field, desc in core.items()
        }
        memory_table = {
            desc: self.config["memory"][field]
            for field, desc in memory.items()
        }
        return Group(
            header,
            "Cores:",
            table(core_table, justify="left"),
            "Memory:",
            table(memory_table, justify="left"),
        )


@magics_class
class LegateInfoMagics(Magics):
    def __init__(self, shell: InteractiveShell | None = None) -> None:
        super().__init__(shell=shell)
        self.info = LegateInfo()

    @line_magic
    def legate_info(self, line: str) -> None:  # noqa: ARG002, D102
        rich_print(self.info.ui)
