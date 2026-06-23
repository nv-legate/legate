# SPDX-FileCopyrightText: Copyright (c) 2021-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import re
import sys
import json
import platform
from contextlib import suppress
from functools import cache
from importlib import import_module, metadata as importlib_metadata
from importlib.util import find_spec
from pathlib import Path
from subprocess import CalledProcessError, check_output
from textwrap import indent
from typing import TYPE_CHECKING, Any, TypedDict
from urllib.parse import unquote, urlparse

from .. import install_info
from .has_started import runtime_has_started

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence


class BuildInfo(TypedDict):
    """Information about how legate was configured and built."""

    build_type: str
    use_openmp: str
    use_cuda: str
    networks: str
    conduit: str
    configure_options: str


__all__ = [
    "build_info",
    "conda_package_dists",
    "info",
    "machine_info",
    "package_dists",
    "package_versions",
    "print_build_info",
    "print_conda_package_details",
    "print_package_details",
    "print_package_versions",
    "print_system_info",
    "system_info",
]


def build_info() -> BuildInfo:
    """Get information about how legate was configured and built."""
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


def _realm_version() -> str:
    result = install_info.realm_version
    if result == "":
        return FAILED_TO_DETECT

    if install_info.realm_git_branch:
        result += f" (commit: {install_info.realm_git_branch})"
    return result


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
    """Get information about the system on which the program is running."""
    return {
        "Python": f"{sys.version.splitlines()[0]}",
        "Platform": f"{platform.platform()}",
        "GPU driver": f"{_driver_version()}",
        "GPU devices": _devices(),
    }


PackageVersions = dict[str, str]


def package_versions() -> PackageVersions:
    """Get versions for packages in the legate and numpy ecosystems."""
    return {
        "realm": f"{_realm_version()}",
        "legion": f"{_legion_version()}",
        "legate": f"{_try_version('legate', '__version__')}",
        "cupynumeric": f"{_try_version('cupynumeric', '__version__')}",
        "numpy": f"{_try_version('numpy', '__version__')}",
        "scipy": f"{_try_version('scipy', '__version__')}",
        "numba": f"{_try_version('numba', '__version__')}",
    }


PackageDists = dict[str, str]
CondaPackageDists = dict[str, str]

# Python details follow imports for the broader runtime ecosystem. Conda
# details stay focused on packages that encode the Legate/CUDA environment.
PYTHON_PACKAGES = ("legate", "cupynumeric", "numpy", "scipy", "numba")
CONDA_PACKAGES = ("cuda-version", "legate", "cupynumeric")
NO_CONDA_METADATA = "(no conda metadata in active Python prefix)"


def _module_path(module_name: str) -> str | None:
    try:
        spec = find_spec(module_name)
    except (ImportError, ValueError):
        return None

    if spec is None or not spec.submodule_search_locations:
        return None

    return str(next(iter(spec.submodule_search_locations)))


def _dist_owns_path(
    dist: importlib_metadata.Distribution, active_path: Path
) -> bool:
    files = dist.files
    if files is None:
        return False

    for file in files:
        candidate = Path(str(dist.locate_file(file))).resolve(strict=False)
        if candidate == active_path or active_path in candidate.parents:
            return True

    return False


def _editable_root(dist: importlib_metadata.Distribution) -> Path | None:
    direct_url = dist.read_text("direct_url.json")
    if direct_url is None:
        return None

    with suppress(KeyError, TypeError, ValueError):
        data = json.loads(direct_url)
        if not data["dir_info"]["editable"]:
            return None
        url = urlparse(str(data["url"]))
        if url.scheme == "file":
            return Path(unquote(url.path)).resolve(strict=False)
    return None


def _distribution_details(
    module_name: str, package_to_dists: Mapping[str, Sequence[str]]
) -> str:
    path = _module_path(module_name)
    if path is None:
        return FAILED_TO_DETECT

    dist_names = package_to_dists.get(module_name)
    if not dist_names:
        return f"{FAILED_TO_DETECT} (path: {path})"

    active_path = Path(path).resolve(strict=False)
    for dist_name in sorted(dist_names):
        with suppress(importlib_metadata.PackageNotFoundError):
            dist = importlib_metadata.distribution(dist_name)
            if _dist_owns_path(dist, active_path):
                installer = (dist.read_text("INSTALLER") or "").strip()
                installer_text = (
                    f"installer: {installer}, " if installer else ""
                )
                return (
                    f"{dist.name} {dist.version} "
                    f"({installer_text}path: {path})"
                )
            root = _editable_root(dist)
            if root is not None and (
                active_path == root or root in active_path.parents
            ):
                return f"{dist.name} {dist.version} (editable, path: {path})"

    return f"{FAILED_TO_DETECT} (path: {path})"


@cache
def _package_dists() -> PackageDists:
    package_to_dists = importlib_metadata.packages_distributions()
    return {
        pkg: _distribution_details(pkg, package_to_dists)
        for pkg in PYTHON_PACKAGES
    }


def package_dists() -> PackageDists:
    """Get distribution information for packages in the legate ecosystem."""
    return _package_dists().copy()


def _conda_dist_name(info: Mapping[str, Any]) -> str | None:
    name = info.get("name")
    version = info.get("version")
    build = info.get("build")
    if (
        isinstance(name, str)
        and isinstance(version, str)
        and isinstance(build, str)
        and name
        and version
        and build
    ):
        return f"{name}-{version}-{build}"

    return None


def _conda_package_detail(info: dict[str, Any]) -> str:
    result = _conda_dist_name(info)
    if result is None:
        return FAILED_TO_DETECT
    if channel := info.get("channel"):
        result += f" ({channel})"
    return result


def _conda_meta_packages(meta_dir: Path) -> dict[str, dict[str, Any]]:
    packages = {}
    for path in meta_dir.glob("*.json"):
        with suppress(OSError, json.JSONDecodeError):
            data = json.loads(path.read_text(encoding="utf-8"))
            name = data.get("name")
            if name in CONDA_PACKAGES and _conda_dist_name(data) is not None:
                packages[name] = data

    return packages


@cache
def _conda_package_dists() -> CondaPackageDists:
    # sys.prefix anchors this to the active Python environment. In a venv
    # layered over conda, the Python package section remains authoritative and
    # this section honestly reports that the venv has no conda metadata.
    meta_dir = Path(sys.prefix) / "conda-meta"
    result = {"prefix": sys.prefix}
    if not meta_dir.is_dir():
        result.update(dict.fromkeys(CONDA_PACKAGES, NO_CONDA_METADATA))
        return result

    conda_packages = _conda_meta_packages(meta_dir)
    result.update(
        {
            pkg: (
                _conda_package_detail(conda_packages[pkg])
                if pkg in conda_packages
                else FAILED_TO_DETECT
            )
            for pkg in CONDA_PACKAGES
        }
    )
    return result


def conda_package_dists() -> CondaPackageDists:
    """Get conda package information from the active Python prefix."""
    return _conda_package_dists().copy()


MachineInfo = TypedDict(
    "MachineInfo",
    {"Preferred target": str, "GPU": str, "OMP": str, "CPU": str},
)


def machine_info() -> MachineInfo:
    """Get machine information as a dictionary of strings."""
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


def _runtime_info() -> dict[str, Any]:
    # we import this here because importing anything from
    # ..core will try to start the runtime, and we want
    # other functions in this module to be usable from legate-issue,
    # which shouldn't require the runtime
    from ..core import get_legate_runtime  # noqa: PLC0415

    config = get_legate_runtime().config()
    config_dict = {}
    for attr in filter(lambda a: not a.startswith("__"), dir(config)):
        config_dict[attr] = getattr(config, attr)
    return config_dict


def _realm_runtime_info() -> dict[str, Any]:
    from ..core import get_legate_runtime  # noqa: PLC0415

    config = get_legate_runtime().realm_config()
    config_dict = {}
    for attr in filter(lambda a: not a.startswith("__"), dir(config)):
        config_dict[attr] = getattr(config, attr)
    return config_dict


#: Info on how and where legate is being run.
Info = TypedDict(
    "Info",
    {
        "Program": str,
        "Legate runtime configuration": dict[str, Any] | str,
        "Realm runtime configuration": dict[str, Any] | str,
        "Machine": MachineInfo | str,
        "System info": SystemInfo,
        "Package versions": PackageVersions,
        "Package details": PackageDists,
        "Conda package details": CondaPackageDists,
        "Legate build configuration": BuildInfo,
    },
)


def info(*, start_runtime: bool = True) -> Info:
    """
    Construct a dictionary of information about the current legate program
    that can be used for debugging or for reproducibility.

    Parameters
    ----------
    start_runtime: bool = True
        If ``True``, the legate runtime will be started (if it has not already
        been) to get information that depends on the runtime.  If ``False`` and
        the runtime has not been started, information that depends on the
        runtime will be missing.

    Returns
    -------
    Info
        A hierarchical dictionary of information strings, with the following
        top-level keys:

        - ``"Program"``
        - ``"Legate runtime configuration"``
        - ``"Realm runtime configuration"``
        - ``"Machine"``
        - ``"System info"``
        - ``"Package versions"``
        - ``"Package details"``
        - ``"Conda package details"``
        - ``"Legate build configuration"``

        The ``Legate runtime configuration``, ``Realm runtime configuration``,
        and ``Machine`` only have useful information if the runtime has
        started.
    """
    use_runtime = start_runtime or runtime_has_started()
    NO_RUNTIME = "(unavailable, legate runtime not started)"
    return {
        "Program": " ".join(sys.argv),
        "Legate runtime configuration": (
            NO_RUNTIME if not use_runtime else _runtime_info()
        ),
        "Realm runtime configuration": (
            NO_RUNTIME if not use_runtime else _realm_runtime_info()
        ),
        "Machine": (NO_RUNTIME if not use_runtime else machine_info()),
        "System info": system_info(),
        "Package versions": package_versions(),
        "Package details": package_dists(),
        "Conda package details": conda_package_dists(),
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


def print_conda_package_details() -> None:  # noqa: D103
    details = conda_package_dists()
    if all(details[pkg] == NO_CONDA_METADATA for pkg in CONDA_PACKAGES):
        details = {"prefix": details["prefix"], "status": NO_CONDA_METADATA}
    print(  # noqa: T201
        "\n".join(
            ["Conda package details:", *_nested_dict_pretty_print(details, 2)]
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
