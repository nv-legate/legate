# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
import platform
import multiprocessing
from functools import cached_property
from itertools import chain
from pathlib import Path

from .fs import get_legate_paths
from .types import CPUInfo, GPUInfo, LegatePaths

__all__ = ("System",)


class System:
    """Encapsulate details of the current system, e.g. runtime paths and OS."""

    def __init__(self) -> None:
        self.env = dict(os.environ)

    @cached_property
    def legate_paths(self) -> LegatePaths:
        """All the current runtime Legate Paths.

        Returns
        -------
            LegatePaths

        """
        return get_legate_paths()

    @cached_property
    def os(self) -> str:
        """The OS for this system.

        Raises
        ------
            RuntimeError, if OS is not supported

        Returns
        -------
            str

        """
        if (os := platform.system()) not in {"Linux", "Darwin"}:
            msg = f"Legate does not work on {os}"
            raise RuntimeError(msg)
        return os

    @cached_property
    def cpus(self) -> tuple[CPUInfo, ...]:
        """A list of CPUs on the system."""
        # Need to use if->elif pattern (instead of several if with early
        # return) for mypy not to warn that either the linux or macOS path is
        # "unreachable".
        if sys.platform.startswith("darwin"):
            N = multiprocessing.cpu_count()
            ret = ((i,) for i in range(N))
        elif sys.platform.startswith("linux"):
            all_sets = linux_load_sibling_sets()
            affinity = os.sched_getaffinity(0)
            ret = sorted(s for s in all_sets if s[0] in affinity)
        else:
            raise NotImplementedError(sys.platform)
        return tuple(map(CPUInfo, ret))

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
                msg = "GPU execution is not available on OSX."
                raise RuntimeError(msg)
            msg = (
                "GPU detection failed. Make sure nvml and pynvml are "
                "both installed."
            )
            raise RuntimeError(msg)

        num_gpus = pynvml.nvmlDeviceGetCount()

        results = []
        for i in range(num_gpus):
            info = pynvml.nvmlDeviceGetMemoryInfo(
                pynvml.nvmlDeviceGetHandleByIndex(i)
            )
            results.append(GPUInfo(i, int(info.total)))

        # Since pyNVML currently ignores CUDA_VISIBLE_DEVICES, we need to parse
        # it ourselves
        if env_string := self.env.get("CUDA_VISIBLE_DEVICES"):
            # TODO(wonchanl): technically, CUDA_VISIBLE_DEVICES can have UUIDs
            # as GPU identifiers, but here we assume it only has GPU indices

            # We stop at the first invalid token to be consistent with how
            # CUDA_VISIBLE_DEVICES is parsed by CUDA

            results = [
                results[device_id]
                for device_id in parse_cuda_visible_devices(
                    env_string, num_gpus
                )
            ]

        return tuple(results)


def expand_range(value: str) -> tuple[int, ...]:
    if value == "":
        return ()
    if "-" not in value:
        return (int(value),)
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
        with Path(
            f"/sys/devices/system/cpu/cpu{i}/topology/thread_siblings_list"
        ).open() as fd:
            line = fd.read()
        sibling_sets.add(extract_values(line.strip()))

    return sibling_sets


def parse_cuda_visible_devices(env_string: str, num_gpus: int) -> list[int]:
    device_ids = []

    for token in env_string.split(","):
        try:
            device_id = int(token)
            if device_id < 0 or device_id >= num_gpus:
                break
            device_ids.append(device_id)
        except ValueError:
            break

    return device_ids
