# Copyright 2024 NVIDIA Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
from __future__ import annotations

import json
import platform
import re
import sys
from importlib import import_module
from subprocess import CalledProcessError, check_output
from textwrap import indent

NEWLINE = "\n"
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


def legion_version() -> str:
    import legate.install_info as info

    if (result := info.legion_version) == "":
        return FAILED_TO_DETECT

    if info.legion_git_branch:
        result += f" (commit: {info.legion_git_branch})"

    return result


def cuda_version() -> str:
    try:
        if out := check_output("conda list cuda-version --json".split()):
            info = json.loads(out.decode("utf-8"))[0]
            return f"{info['dist_name']} ({info['channel']})"
        return FAILED_TO_DETECT
    except (CalledProcessError, IndexError, KeyError):
        return FAILED_TO_DETECT
    except FileNotFoundError:
        return "(conda missing)"


def driver_version() -> str:
    try:
        out = check_output(
            "nvidia-smi --query-gpu=driver_version --format=csv,noheader --id=0".split()  # noqa
        )
        return out.decode("utf-8").strip()
    except (CalledProcessError, IndexError, KeyError):
        return FAILED_TO_DETECT
    except FileNotFoundError:
        return "(nvidia-smi missing)"


def devices() -> str:
    try:
        out = check_output("nvidia-smi -L".split())
        gpus = re.sub(r" \(UUID: .*\)", "", out.decode("utf-8").strip())
        return f"\n{indent(gpus, '  ')}"
    except (CalledProcessError, IndexError, KeyError):
        return FAILED_TO_DETECT
    except FileNotFoundError:
        return "(nvidia-smi missing)"


def main() -> None:
    print(f"Python      :  {sys.version.split(NEWLINE)[0]}")
    print(f"Platform    :  {platform.platform()}")
    print(f"Legion      :  {legion_version()}")
    print(f"Legate      :  {try_version('legate', '__version__')}")
    print(f"cuPynumeric :  {try_version('cupynumeric', '__version__')}")
    print(f"Numpy       :  {try_version('numpy', '__version__')}")
    print(f"Scipy       :  {try_version('scipy', '__version__')}")
    print(f"Numba       :  {try_version('numba', '__version__')}")
    print(f"CTK package :  {cuda_version()}")
    print(f"GPU driver  :  {driver_version()}")
    print(f"GPU devices :  {devices()}")
