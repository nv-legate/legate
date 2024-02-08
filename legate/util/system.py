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

from __future__ import annotations

import multiprocessing
import os
import platform
import sys
from functools import cached_property
from itertools import chain

from .fs import get_legate_paths, get_legion_paths
from .types import CPUInfo, GPUInfo, LegatePaths, LegionPaths

__all__ = ("System",)


class System:
    """Encapsulate details of the current system, e.g. runtime paths and OS."""

    def __init__(self) -> None:
        self.env = dict(os.environ)

    @cached_property
    def legate_paths(self) -> LegatePaths:
        """All the current runtime Legate Paths

        Returns
        -------
            LegionPaths

        """
        return get_legate_paths()

    @cached_property
    def legion_paths(self) -> LegionPaths:
        """All the current runtime Legion Paths

        Returns
        -------
            LegionPaths

        """
        return get_legion_paths(self.legate_paths)

    @cached_property
    def os(self) -> str:
        """The OS for this system

        Raises
        ------
            RuntimeError, if OS is not supported

        Returns
        -------
            str

        """
        if (os := platform.system()) not in {"Linux", "Darwin"}:
            raise RuntimeError(f"Legate does not work on {os}")
        return os

    @cached_property
    def cpus(self) -> tuple[CPUInfo, ...]:
        """A list of CPUs on the system."""

        if sys.platform.startswith("darwin"):
            N = multiprocessing.cpu_count()
            return tuple(CPUInfo((i,)) for i in range(N))
        elif sys.platform.startswith("linux"):
            all_sets = linux_load_sibling_sets()

            available_sets = (
                s for s in all_sets if s[0] in os.sched_getaffinity(0)
            )
            return tuple(CPUInfo(x) for x in sorted(available_sets))
        else:
            raise NotImplementedError(sys.platform)

    @cached_property
    def gpus(self) -> tuple[GPUInfo, ...]:
        """A list of GPUs on the system, including total memory information."""

        try:
            # This pynvml import is protected inside this method so that in
            # case pynvml is not installed, tests stages that don't need gpu
            # info (e.g. cpus, eager) will proceed unaffected. Test stages
            # that do require gpu info will fail here with an ImportError.
            #
            # Need to add unused-ignore here since mypy complains:
            #
            # Unused "type: ignore" comment, use narrower [import-untyped]
            # instead of [import] code
            #
            # But pynvml may or may not be installed, and so the suggested
            # narrower error code ends up being wrong half the time.
            import pynvml  # type: ignore[import, unused-ignore]

            # Also a pynvml package is available on some platforms that won't
            # have GPUs for some reason. In which case this init call will
            # fail.
            pynvml.nvmlInit()
        except Exception:
            if platform.system() == "Darwin":
                raise RuntimeError("GPU execution is not available on OSX.")
            else:
                raise RuntimeError(
                    "GPU detection failed. Make sure nvml and pynvml are "
                    "both installed."
                )

        num_gpus = pynvml.nvmlDeviceGetCount()

        results = []
        for i in range(num_gpus):
            info = pynvml.nvmlDeviceGetMemoryInfo(
                pynvml.nvmlDeviceGetHandleByIndex(i)
            )
            results.append(GPUInfo(i, info.total))

        return tuple(results)


def expand_range(value: str) -> tuple[int, ...]:
    if value == "":
        return tuple()
    if "-" not in value:
        return tuple((int(value),))
    start, stop = value.split("-")

    return tuple(range(int(start), int(stop) + 1))


def extract_values(line: str) -> tuple[int, ...]:
    return tuple(
        sorted(
            chain.from_iterable(
                expand_range(r) for r in line.strip().split(",")
            )
        )
    )


def linux_load_sibling_sets() -> set[tuple[int, ...]]:
    N = multiprocessing.cpu_count()

    sibling_sets: set[tuple[int, ...]] = set()
    for i in range(N):
        line = open(
            f"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list"
        ).read()
        sibling_sets.add(extract_values(line.strip()))

    return sibling_sets
