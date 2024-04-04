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

import os
import re
import sys
from pathlib import Path

from .types import LegatePaths, LegionPaths

__all__ = (
    "get_legate_build_dir",
    "get_legate_paths",
    "get_legion_paths",
    "read_c_define",
    "read_cmake_cache_value",
)


def read_c_define(header_path: Path, name: str) -> str | None:
    """Open a C header file and read the value of a #define

    Parameters
    ----------
    header_path : Path
        Location of the C header file to scan

    name : str
        The name to search the header for

    Returns
    -------
        str : value from the header or None, if it does not exist

    """
    try:
        with open(header_path, "r") as f:
            lines = (line for line in f if line.startswith("#define"))
            for line in lines:
                tokens = line.split(" ")
                if tokens[1].strip() == name:
                    return tokens[2].strip()
    except IOError:
        pass

    return None


def read_cmake_cache_value(file_path: Path, pattern: str) -> str:
    """Search a cmake cache file for a given pattern and return the associated
    value.

    Parameters
    ----------
        file_path: Path
            Location of the cmake cache file to scan

        pattern : str
            A pattern to seach for in the file

    Returns
    -------
        str

    Raises
    ------
        RuntimeError, if the value is not found

    """
    with open(file_path, encoding="utf-8") as f:
        for line in f:
            if re.match(pattern, line):
                return line.strip().split("=")[1]

    raise RuntimeError(f"Could not find value for {pattern} in {file_path}")


def get_legate_build_dir(legate_dir: Path) -> Path | None:
    """Determine the location of the Legate build directory.

    If the build directory cannot be found, None is returned.

    Parameters
    ----------
        legate_dir : Path
            Directory containing a Legate executable


    Returns
        Path or None

    """

    def get_legate_core_arch() -> str | None:
        # We might be calling this from the driver (i.e. legate) in which case
        # we don't want to require the user to have this set.
        if legate_core_arch := os.environ.get("LEGATE_CORE_ARCH", "").strip():
            return legate_core_arch

        from importlib.metadata import PackageNotFoundError, distribution

        try:
            dist = distribution("legate-core")
        except PackageNotFoundError:
            # not installed at all, without a LEGATE_CORE_ARCH there is nothing
            # we can really do.
            return None

        if (files := dist.files) is None:
            return None

        for path in files:
            # if the path is of the form
            # <something>/_skbuild/<something_else>/cmake-install then we can
            # be reasonably sure that:
            #
            # 1. We are in an editable install.
            # 2. <something> is the arch directory.
            path_parts = path.parts
            if len(path_parts) < 3:
                continue

            if (
                path_parts[1] == "_skbuild"
                and path_parts[3] == "cmake-install"
            ):
                return path_parts[0]
        return None

    # If using a local non-scikit-build CMake build dir, read
    # Legion_BINARY_DIR and Legion_SOURCE_DIR from CMakeCache.txt
    legate_core_arch = get_legate_core_arch()
    if legate_core_arch is None:
        return None

    skbuild_dir = legate_dir / legate_core_arch / "_skbuild"
    if not skbuild_dir.exists():
        return None

    for f in skbuild_dir.iterdir():
        # If using a local scikit-build dir at _skbuild/<arch>/cmake-build,
        # read Legion_BINARY_DIR and Legion_SOURCE_DIR from CMakeCache.txt

        legate_build_dir = skbuild_dir / f / "cmake-build"
        cmake_cache_txt = legate_build_dir / "CMakeCache.txt"

        if not cmake_cache_txt.exists():
            continue

        try:
            # Test whether _legate_core_FOUND_METHOD is set to
            # SELF_BUILT. If it is, then we built legate_core C++ as a
            # side-effect of building legate_core_python.
            read_cmake_cache_value(
                cmake_cache_txt,
                "_legate_core_FOUND_METHOD:INTERNAL=SELF_BUILT",
            )
        except Exception:
            # _legate_core_FOUND_METHOD is either PRE_BUILT or INSTALLED,
            # so check to see if legate_core_DIR is a valid path. If it is,
            # check whether legate_core_DIR is a path to a legate_core
            # build dir i.e.  `-D legate_core_ROOT=/legate.core/build`
            legate_core_dir = Path(
                read_cmake_cache_value(
                    cmake_cache_txt, "legate_core_DIR:PATH="
                )
            )

            # If legate_core_dir doesn't have a CMakeCache.txt, CMake's
            # find_package found a system legate_core installation.
            # Return the installation paths.
            cmake_cache_txt = legate_core_dir / "CMakeCache.txt"
            if cmake_cache_txt.exists():
                return Path(
                    read_cmake_cache_value(
                        cmake_cache_txt, "legate_core_BINARY_DIR:STATIC="
                    )
                )
            return None

        return legate_build_dir

    return None


def get_legate_paths() -> LegatePaths:
    """Determine all the important runtime paths for Legate

    Returns
    -------
        LegatePaths

    """
    import legate

    legate_dir = Path(legate.__path__[0]).parent
    legate_build_dir = get_legate_build_dir(legate_dir)

    if legate_build_dir is None:
        if (legate_dir / ".git").exists():
            # we are in the source repository, but have neither configured nor
            # installed the libraries. Most of these paths are meaningless, but
            # let's at least fill out the right bind_sh_path.
            bind_sh_path = legate_dir / "bind.sh"
            legate_lib_path = Path("this_path_does_not_exist")
            assert not legate_lib_path.exists()
        else:
            # It's possible we are in an installed library, in which case
            # legate_dir is probably
            # <PREFIX>/lib/python<version>/site-packages/legate. In this case,
            # bind.sh and the libs are under <PREFIX>/bin/bind.sh and
            # <PREFIX>/lib respectively.
            prefix_dir = legate_dir.parents[2]
            bind_sh_path = prefix_dir / "bin" / "bind.sh"
            legate_lib_path = prefix_dir / "lib"
            assert legate_lib_path.exists() and legate_lib_path.is_dir()

        assert bind_sh_path.exists() and bind_sh_path.is_file()
        return LegatePaths(
            legate_dir=legate_dir,
            legate_build_dir=legate_build_dir,
            bind_sh_path=bind_sh_path,
            legate_lib_path=legate_lib_path,
        )

    cmake_cache_txt = legate_build_dir / "CMakeCache.txt"

    legate_source_dir = Path(
        read_cmake_cache_value(
            cmake_cache_txt, "legate_core_SOURCE_DIR:STATIC="
        )
    )

    legate_binary_dir = Path(
        read_cmake_cache_value(
            cmake_cache_txt, "legate_core_BINARY_DIR:STATIC="
        )
    )

    return LegatePaths(
        legate_dir=legate_dir,
        legate_build_dir=legate_build_dir,
        bind_sh_path=legate_source_dir / "bind.sh",
        legate_lib_path=legate_binary_dir / "lib",
    )


def get_legion_paths(legate_paths: LegatePaths) -> LegionPaths:
    """Determine all the important runtime paths for Legion

    Parameters
    ----------
        legate_paths : LegatePaths
            Locations of Legate runtime paths

    Returns
    -------
        LegionPaths

    """

    # Construct and return paths needed to launch `legion_python`,accounting
    # for multiple ways Legion and legate_core may be configured or installed.
    #
    # 1. Legion was found in a standard system location (/usr, $CONDA_PREFIX)
    # 2. Legion was built as a side-effect of building legate_core:
    #    ```
    #    CMAKE_ARGS="" python -m pip install .
    #    ```
    # 3. Legion was built in a separate directory independent of legate_core
    #    and the path to its build directory was given when configuring
    #    legate_core:
    #    ```
    #    CMAKE_ARGS="-D Legion_ROOT=/legion/build" \
    #        python -m pip install .
    #    ```
    #
    # Additionally, legate_core has multiple run modes:
    #
    # 1. As an installed Python module (`python -m pip install .`)
    # 2. As an "editable" install (`python -m pip install --editable .`)
    #
    # When determining locations of Legion and legate_core paths, prioritize
    # local builds over global installations. This allows devs to work in the
    # source tree and re-run without overwriting existing installations.

    def installed_legion_paths(legion_dir: Path) -> LegionPaths:
        legion_lib_dir = legion_dir / "lib"
        for f in legion_lib_dir.iterdir():
            legion_module = f / "site-packages"
            if legion_module.exists():
                break

        # NB: for-else clause! (executes if NO loop break)
        else:
            raise RuntimeError("could not determine legion module location")

        legion_bin_path = legion_dir / "bin"
        legion_include_path = legion_dir / "include"

        return LegionPaths(
            legion_bin_path=legion_bin_path,
            legion_lib_path=legion_lib_dir,
            realm_defines_h=legion_include_path / "realm_defines.h",
            legion_defines_h=legion_include_path / "legion_defines.h",
            legion_spy_py=legion_bin_path / "legion_spy.py",
            legion_python=legion_bin_path / "legion_python",
            legion_prof=legion_bin_path / "legion_prof",
            legion_module=legion_module,
            legion_jupyter_module=legion_module,
        )

    if (legate_build_dir := legate_paths.legate_build_dir) is None:
        legate_build_dir = get_legate_build_dir(legate_paths.legate_dir)

    # If no local build dir found, assume legate installed into the python env
    if legate_build_dir is None:
        return installed_legion_paths(legate_paths.legate_dir.parents[2])

    # If a legate build dir was found, read `Legion_SOURCE_DIR` and
    # `Legion_BINARY_DIR` from in CMakeCache.txt, return paths into the source
    # and build dirs. This allows devs to quickly rebuild inplace and use the
    # most up-to-date versions without needing to install Legion and
    # legate_core globally.

    cmake_cache_txt = legate_build_dir / "CMakeCache.txt"

    try:
        try:
            # Test whether Legion_DIR is set. If it isn't, then we built
            # Legion as a side-effect of building legate_core
            read_cmake_cache_value(
                cmake_cache_txt, "Legion_DIR:PATH=Legion_DIR-NOTFOUND"
            )
        except Exception:
            # If Legion_DIR is a valid path, check whether it's a
            # Legion build dir, i.e. `-D Legion_ROOT=/legion/build`
            legion_dir = Path(
                read_cmake_cache_value(cmake_cache_txt, "Legion_DIR:PATH=")
            )
            if legion_dir.joinpath("CMakeCache.txt").exists():
                cmake_cache_txt = legion_dir / "CMakeCache.txt"
    finally:
        # Hopefully at this point we have a valid cmake_cache_txt with a
        # valid Legion_SOURCE_DIR and Legion_BINARY_DIR
        try:
            # If Legion_SOURCE_DIR and Legion_BINARY_DIR are in CMakeCache.txt,
            # return the paths to Legion in the legate_core build dir.
            legion_source_dir = Path(
                read_cmake_cache_value(
                    cmake_cache_txt, "Legion_SOURCE_DIR:STATIC="
                )
            )
            legion_binary_dir = Path(
                read_cmake_cache_value(
                    cmake_cache_txt, "Legion_BINARY_DIR:STATIC="
                )
            )

            legion_runtime_dir = legion_binary_dir / "runtime"
            legion_bindings_dir = legion_source_dir / "bindings"

            return LegionPaths(
                legion_bin_path=legion_binary_dir / "bin",
                legion_lib_path=legion_binary_dir / "lib",
                realm_defines_h=legion_runtime_dir / "realm_defines.h",
                legion_defines_h=legion_runtime_dir / "legion_defines.h",
                legion_spy_py=legion_source_dir / "tools" / "legion_spy.py",
                legion_python=legion_binary_dir / "bin" / "legion_python",
                legion_prof=legion_binary_dir / "bin" / "legion_prof",
                legion_module=legion_bindings_dir / "python" / "build" / "lib",
                legion_jupyter_module=legion_source_dir / "jupyter_notebook",
            )
        except Exception:
            pass

    # Otherwise return the installation paths.
    return installed_legion_paths(Path(sys.argv[0]).parents[1])
