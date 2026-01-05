# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
import sys
import json
import platform
from functools import cache
from importlib import import_module
from subprocess import CalledProcessError, check_output
from textwrap import indent
from typing import Any, TypedDict

from .. import install_info


class BuildInfo(TypedDict):
    build_type: str
    use_openmp: str
    use_cuda: str
    networks: str
    conduit: str
    configure_options: str


__all__ = [
    "build_info",
    "info",
    "machine_info",
    "package_dists",
    "package_versions",
    "print_build_info",
    "print_package_details",
    "print_package_versions",
    "print_system_info",
    "system_info",
]


def build_info() -> BuildInfo:
    """Information about how legate was configured and built."""
    networks = install_info.networks
    return {
        "build_type": f"{install_info.build_type}",
        "use_openmp": f"{install_info.use_openmp}",
        "use_cuda": f"{install_info.use_cuda}",
        "networks": f"{','.join(networks) if networks else ''}",
        "conduit": f"{install_info.conduit}",
        "configure_options": f"{install_info.configure_options}",
    }


FAILED_TO_DETECT = "(failed to detect)"


def _try_version(module_name: str, attr: str) -> str:
    try:
        module = import_module(module_name)
        if not module:
            return FAILED_TO_DETECT
        return getattr(module, attr)
    except ModuleNotFoundError:
        return FAILED_TO_DETECT
    except ImportError as e:
        err = re.sub(r" \(.*\)", "", str(e))  # remove any local path
        return f"(ImportError: {err})"
    except Exception as e:
        return f"(Exception on import: {e})"


def _legion_version() -> str:
    result = install_info.legion_version
    if result == "":
        return FAILED_TO_DETECT

    if install_info.legion_git_branch:
        result += f" (commit: {install_info.legion_git_branch})"
    return result


def _try_conda(package: str) -> str:
    try:
        if out := check_output(["conda", "list", package, "--json"]):
            info = json.loads(out.decode("utf-8"))[0]
            return f"{info['dist_name']} ({info['channel']})"

    except (CalledProcessError, IndexError, KeyError):
        return FAILED_TO_DETECT
    except FileNotFoundError:
        return "(conda missing)"
    else:
        return FAILED_TO_DETECT


Devices = dict[str, str] | str


def _devices() -> Devices:
    cmd = ["nvidia-smi", "-L"]
    gpu_dict = {}
    try:
        out = check_output(cmd)
        gpus = re.sub(
            r" \(UUID: .*\)", "", out.decode("utf-8").strip()
        ).splitlines()
        for gpu in gpus:
            gpu_strings = gpu.split(":", 1)
            gpu_dict[gpu_strings[0].strip()] = gpu_strings[1].strip()
    except (CalledProcessError, IndexError, KeyError):
        return FAILED_TO_DETECT
    except FileNotFoundError:
        return "(nvidia-smi missing)"
    return gpu_dict


def _driver_version() -> str:
    cmd = (
        "nvidia-smi",
        "--query-gpu=driver_version",
        "--format=csv,noheader",
        "--id=0",
    )
    try:
        out = check_output(cmd)
        return out.decode("utf-8").strip()
    except (CalledProcessError, IndexError, KeyError):
        return FAILED_TO_DETECT
    except FileNotFoundError:
        return "(nvidia-smi missing)"


SystemInfo = TypedDict(
    "SystemInfo",
    {
        "Python": str,
        "Platform": str,
        "GPU driver": str,
        "GPU devices": Devices,
    },
)


def system_info() -> SystemInfo:
    """Information about the system on which the program is running."""
    return {
        "Python": f"{sys.version.splitlines()[0]}",
        "Platform": f"{platform.platform()}",
        "GPU driver": f"{_driver_version()}",
        "GPU devices": _devices(),
    }


PackageVersions = dict[str, str]


def package_versions() -> PackageVersions:
    """Versions for packages in the legate and numpy ecosystems."""
    return {
        "legion": f"{_legion_version()}",
        "legate": f"{_try_version('legate', '__version__')}",
        "cupynumeric": f"{_try_version('cupynumeric', '__version__')}",
        "numpy": f"{_try_version('numpy', '__version__')}",
        "scipy": f"{_try_version('scipy', '__version__')}",
        "numba": f"{_try_version('numba', '__version__')}",
    }


PackageDists = dict[str, str]


# _try_conda() is slow, and the result should be the same every time,
# so we cache it
@cache
def _package_dists() -> PackageDists:
    dist_info = {}
    packages = ("cuda-version", "legate", "cupynumeric")
    for pkg in packages:
        dist_info[pkg] = f"{_try_conda(pkg)}"
    return dist_info


def package_dists() -> PackageDists:
    """Distribution information for packages in the legate ecosystem."""
    return _package_dists().copy()


MachineInfo = TypedDict(
    "MachineInfo",
    {"Preferred target": str, "GPU": str, "OMP": str, "CPU": str},
)


def machine_info() -> MachineInfo:
    """Machine information as a dictionary of strings."""
    # we import this here because importing anything from
    # ..core will try to start the runtime, and we want
    # other functions in this module to be usable from legate-issue,
    # which shouldn't require the runtime
    from ..core import TaskTarget, get_legate_runtime  # noqa: PLC0415

    machine = get_legate_runtime().get_machine()
    return {
        "Preferred target": machine.preferred_target.name,
        "GPU": str(machine.get_processor_range(TaskTarget.GPU)),
        "OMP": str(machine.get_processor_range(TaskTarget.OMP)),
        "CPU": str(machine.get_processor_range(TaskTarget.CPU)),
    }


Info = TypedDict(
    "Info",
    {
        "Program": str,
        "Legate runtime configuration": str,
        "Machine": MachineInfo,
        "System info": SystemInfo,
        "Package versions": PackageVersions,
        "Package details": PackageDists,
        "Legate build configuration": BuildInfo,
    },
)


def info() -> Info:
    """
    Construct a dictionary of information about the current legate program
    that can be used for debugging or for reproducibility.

    returns: a hierarchical dictionary of information strings"
    """
    return {
        "Program": " ".join(sys.argv),
        "Legate runtime configuration": os.getenv(
            "LEGATE_CONFIG", default="None"
        ),
        "Machine": machine_info(),
        "System info": system_info(),
        "Package versions": package_versions(),
        "Package details": package_dists(),
        "Legate build configuration": build_info(),
    }


def _nested_dict_pretty_print(obj: Any, ind: int = 0) -> list[str]:
    """Print a nest dictionary of strings with indenting and aligned colons."""

    def _nested_dict_pretty_print_impl(
        obj: Any, ind: int, out: list[str]
    ) -> None:
        _INDENT_INCR = 2
        if isinstance(obj, dict):
            N = max(len(str(key)) for key in obj)
            for key, value in obj.items():
                if isinstance(value, dict):
                    out.append(indent(f"{key!s:<{N + 1}}:", " " * ind))
                    _nested_dict_pretty_print_impl(
                        value, ind + _INDENT_INCR, out
                    )
                else:
                    out.append(
                        indent(f"{key!s:<{N + 1}}:  {value!s}", " " * ind)
                    )

    out: list[str] = []
    _nested_dict_pretty_print_impl(obj, ind, out)
    return out


def print_system_info() -> None:  # noqa: D103
    print(  # noqa: T201
        "\n".join(
            ["System info:", *_nested_dict_pretty_print(system_info(), 2)]
        )
    )


def print_package_versions() -> None:  # noqa: D103
    print(  # noqa: T201
        "\n".join(
            [
                "Package versions:",
                *_nested_dict_pretty_print(package_versions(), 2),
            ]
        )
    )


def print_package_details() -> None:  # noqa: D103
    print(  # noqa: T201
        "\n".join(
            [
                "Package details:",
                *_nested_dict_pretty_print(package_dists(), 2),
            ]
        )
    )


def print_build_info() -> None:  # noqa: D103
    print(  # noqa: T201
        "\n".join(
            [
                "Legate build configuration:",
                *_nested_dict_pretty_print(build_info(), 2),
            ]
        )
    )
