# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# ruff: noqa: D103, T201
import os
import re
import sys
import json
import platform
from importlib import import_module
from subprocess import CalledProcessError, check_output
from textwrap import indent

from .util.info import print_build_info

FAILED_TO_DETECT = "(failed to detect)"


def try_version(module_name: str, attr: str) -> str:
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


def legion_version() -> str:
    import legate.install_info as info

    result = info.legion_version
    if result == "":
        return FAILED_TO_DETECT

    if info.legion_git_branch:
        result += f" (commit: {info.legion_git_branch})"

    return result


def try_conda(package: str) -> str:
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


def print_system_info() -> None:
    print("System info:")
    print(f"  Python      :  {sys.version.splitlines()[0]}")
    print(f"  Platform    :  {platform.platform()}")
    print(f"  GPU driver  :  {driver_version()}")
    print(f"  GPU devices :  {devices()}")


def print_package_versions() -> None:
    print("Package versions:")
    print(f"  legion      :  {legion_version()}")
    print(f"  legate      :  {try_version('legate', '__version__')}")
    print(f"  cupynumeric :  {try_version('cupynumeric', '__version__')}")
    print(f"  numpy       :  {try_version('numpy', '__version__')}")
    print(f"  scipy       :  {try_version('scipy', '__version__')}")
    print(f"  numba       :  {try_version('numba', '__version__')}")


def print_pacakge_details() -> None:
    print("Package details:")
    packages = ("cuda-version", "legate", "cupynumeric")
    N = max(len(pkg) for pkg in packages)
    for pkg in packages:
        print(f"  {pkg:<{N + 1}}: {try_conda(pkg)}")


def driver_version() -> str:
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


def devices() -> str:
    cmd = ["nvidia-smi", "-L"]
    try:
        out = check_output(cmd)
        gpus = re.sub(r" \(UUID: .*\)", "", out.decode("utf-8").strip())
        return f"\n{indent(gpus, '    ')}"
    except (CalledProcessError, IndexError, KeyError):
        return FAILED_TO_DETECT
    except FileNotFoundError:
        return "(nvidia-smi missing)"


def main() -> None:
    # legate-issue should never fail, but sometimes auto-config is
    # too aggressive and will cause legate-issue itself to crash
    os.environ["LEGATE_AUTO_CONFIG"] = "0"

    print_system_info()
    print()
    print_package_versions()
    print()
    print_pacakge_details()
    print()
    print_build_info()
