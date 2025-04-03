# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import re
from pathlib import Path

from .types import LegatePaths

__all__ = (
    "get_legate_build_dir",
    "get_legate_paths",
    "read_c_define",
    "read_cmake_cache_value",
)


def assert_file_exists(path: Path) -> None:
    assert path.exists()
    assert path.is_file()


def assert_dir_exists(path: Path) -> None:
    assert path.exists()
    assert path.is_dir()


def read_c_define(header_path: Path, name: str) -> str | None:
    """Open a C header file and read the value of a #define.

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
        with header_path.open() as f:
            lines = (line for line in f if line.startswith("#define"))
            for line in lines:
                tokens = line.split(" ")
                if tokens[1].strip() == name:
                    return tokens[2].strip()
    except OSError:
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
            A pattern to search for in the file

    Returns
    -------
        str

    Raises
    ------
        RuntimeError, if the value is not found

    """
    with file_path.open() as f:
        for line in f:
            if re.match(pattern, line):
                return line.strip().split("=")[1]

    msg = f"Could not find value for {pattern} in {file_path}"
    raise RuntimeError(msg)


def is_legate_path_in_wheel_tree(path: Path) -> bool:
    r"""Determine whether ``path`` is the path to the pip wheel legate module.

    Parameters
    ----------
    path : Path
        The path to check.

    Returns
    -------
    bool
        True if path is in the pip wheel legate module, False otherwise.
    """
    if not path.exists():
        return False

    wheel_layout_dir = path / "share" / "legate" / "libexec"
    if not wheel_layout_dir.exists():
        return False

    bind_path_in_wheel = wheel_layout_dir / "legate-bind.sh"
    return bind_path_in_wheel.exists()


def is_legate_path_in_src_tree(path: Path) -> bool:
    r"""Determine whether ``path`` is the path to the in-source legate module.

    Parameters
    ----------
    path : Path
        The path to check.

    Returns
    -------
    bool
        True if path is the in-source legate module, False otherwise.
    """
    ret = tuple(path.parts[-3:]) == ("src", "python", "legate")
    if ret:
        assert_file_exists(path / "CMakeLists.txt")
        assert_file_exists(path.parent / "CMakeLists.txt")
        assert_file_exists(path.parent.parent / "cpp" / "CMakeLists.txt")
    return ret


def get_legate_build_dir_from_arch_build_dir(
    legate_arch_dir: Path,
) -> Path | None:
    r"""Given a path to the legate arch directory, determine the original
    legate build directory.

    Parameters
    ----------
    legate_arch_dir : Path
        The path to the legate arch directory.

    Returns
    -------
    None
        If the build directory cannot be located.
    Path
        The path to the located legate build directory.
    """
    cmake_build_dir = legate_arch_dir / "cmake_build"
    if (cmake_build_dir / "CMakeCache.txt").exists():
        return cmake_build_dir
    return None


def get_legate_build_dir_from_skbuild_dir(skbuild_dir: Path) -> Path | None:
    r"""Given the path to the skbuild directory, determine the original
    legate build directory (i.e. the one containing the original CMake files).

    Parameters
    ----------
    skbuild_dir : Path
        The path to the scikit-build-core build directory.

    Returns
    -------
    None
        If the legate build dir could not be located.
    Path
        The path to the located legate build directory.
    """
    if not skbuild_dir.exists():
        return None

    assert_dir_exists(skbuild_dir)
    cmake_cache_txt = skbuild_dir / "CMakeCache.txt"
    if not cmake_cache_txt.exists():
        msg = f"scikit-build-core CMakeCache does not exist: {cmake_cache_txt}"
        raise RuntimeError(msg)

    legate_found_method = read_cmake_cache_value(
        cmake_cache_txt, "_legate_FOUND_METHOD:INTERNAL="
    )
    match legate_found_method:
        case "SELF_BUILT":
            return skbuild_dir
        case "PRE_BUILT":
            # Use of legate_DIR vs LEGATE_DIR is deliberate. The latter points
            # to the "base" directory, the former will point to the arch
            # directory (since that's where the pre-built version lives).
            legate_cmake_build_dir = Path(
                read_cmake_cache_value(cmake_cache_txt, "legate_DIR:PATH=")
            )
            assert_dir_exists(legate_cmake_build_dir)
            assert_file_exists(legate_cmake_build_dir / "CMakeCache.txt")
            return legate_cmake_build_dir
        case _:
            m = (
                "Unknown legate found method: "
                f"{legate_found_method} in {cmake_cache_txt}"
            )
            raise ValueError(m)


def get_legate_build_dir(legate_parent_dir: Path) -> Path | None:
    """Determine the location of the Legate build directory.

    If the build directory cannot be found, None is returned.

    Parameters
    ----------
    legate_parent_dir : Path
        Directory containing the legate Python module, i.e. if given
        '/path/to/legate/__init__.py', legate_dir is '/path/to'.


    Returns
    -------
    Path or None
    """

    def get_legate_arch() -> str | None:
        # We might be calling this from the driver (i.e. legate) in which case
        # we don't want to require the user to have this set.
        if legate_arch := os.environ.get("LEGATE_ARCH", "").strip():
            return legate_arch

        # User has LEGATE_ARCH undefined, but we may yet still be an editable
        # installation.
        if is_legate_path_in_src_tree(legate_parent_dir / "legate"):
            # We have an editable install, only now is it safe to consult the
            # variable. We cannot do it above because we needed to make sure we
            # weren't in fully-installed-mode
            from ..install_info import LEGATE_ARCH

            return LEGATE_ARCH

        # Must be either fully installed, or not yet configured, in any case,
        # we have no arch.
        return None

    legate_arch = get_legate_arch()
    if legate_arch is None:
        return None

    # legate_parent_dir is either <PREFIX>/lib/python<version>/site-packages or
    # $LEGATE_DIR/src/python.
    #
    # If it is the former, then we are fully installed, in which case both of
    # these functions will return None
    #
    # If the latter, then the arch dir will be up 2 and into legate_arch.
    legate_arch_dir = legate_parent_dir.parents[1] / legate_arch
    if (skbuild_dir := legate_arch_dir / "skbuild_core").exists():
        return get_legate_build_dir_from_skbuild_dir(skbuild_dir)
    return get_legate_build_dir_from_arch_build_dir(legate_arch_dir)


def make_legate_bind_path(base: Path) -> Path:
    return base / "share" / "legate" / "libexec" / "legate-bind.sh"


def get_legate_paths_from_installed_dir(legate_mod_dir: Path) -> LegatePaths:
    import sysconfig

    # If legate_build_dir is None, then we are either dealing with an
    # installed version of legate or we may have been called from
    # test.py. legate_mod_dir is either
    # <PREFIX>/lib/python<version>/site-packages/legate, or
    # $LEGATE_DIR/src/python/legate.
    site_package_dir_name = Path(sysconfig.get_paths()["purelib"]).name
    if is_legate_path_in_src_tree(legate_mod_dir):
        # we are in the source repository, and legate_mod_dir =
        # src/python/legate, but have neither configured nor installed the
        # libraries. Most of these paths are meaningless, but let's at
        # least fill out the right bind_sh_path.
        bind_sh_path = make_legate_bind_path(legate_mod_dir.parents[2])
        legate_lib_path = Path("this_path_does_not_exist")
        assert not legate_lib_path.exists()
    elif is_legate_path_in_wheel_tree(legate_mod_dir):
        # We are in a Python pip wheel installed library. This means that
        # things are installed within the module directory such that
        # <PREFIX>/lib/python<version>/site-packages/legate is the base
        # with <base>/share/legate/libexec/legate-bind.sh having bind.sh.
        # Note that the main difference right now is that the wheel layout
        # has everything within the module directory.
        prefix_dir = legate_mod_dir
        bind_sh_path = make_legate_bind_path(prefix_dir)
        legate_lib_path = prefix_dir / "lib64"
        assert_dir_exists(legate_lib_path)
    elif legate_mod_dir.parent.name == site_package_dir_name:
        # It's possible we are in an installed library, in which case
        # legate_mod_dir is probably
        # <PREFIX>/lib/python<version>/site-packages/legate. In this case,
        # legate-bind.sh and the libs are under
        # <PREFIX>/share/legate/libexec/legate-bind.sh and <PREFIX>/lib
        # respectively.
        prefix_dir = legate_mod_dir.parents[3]
        bind_sh_path = make_legate_bind_path(prefix_dir)
        legate_lib_path = prefix_dir / "lib"
        assert_dir_exists(legate_lib_path)
    else:
        msg = f"Unhandled legate module install location: {legate_mod_dir}"
        raise RuntimeError(msg)

    assert_file_exists(bind_sh_path)
    return LegatePaths(
        bind_sh_path=bind_sh_path, legate_lib_path=legate_lib_path
    )


def get_legate_paths_from_build_dir(legate_build_dir: Path) -> LegatePaths:
    # If build_dir is not None, then we almost certainly have an editable
    # install, or are being called by test.py
    cmake_cache_txt = legate_build_dir / "CMakeCache.txt"

    src_dir = Path(
        read_cmake_cache_value(
            cmake_cache_txt, "legate_cpp_SOURCE_DIR:STATIC="
        )
    ).parent
    bind_sh_path = make_legate_bind_path(src_dir)

    legate_binary_dir = Path(
        read_cmake_cache_value(
            cmake_cache_txt, "legate_cpp_BINARY_DIR:STATIC="
        )
    )
    legate_lib_path = legate_binary_dir / "cpp" / "lib"

    assert_file_exists(bind_sh_path)
    assert_dir_exists(legate_lib_path)
    return LegatePaths(
        bind_sh_path=bind_sh_path, legate_lib_path=legate_lib_path
    )


def get_legate_paths() -> LegatePaths:
    """Determine all the important runtime paths for Legate.

    Returns
    -------
    LegatePaths

    Notes
    -----
    This function may be called in 1 of 3 scenarios:
    1. The python libraries are not installed, and this is called
       (transitively) from e.g. test.py.
    2. The python libraries are regularly installed.
    3. The python libraries are installed 'editable' mode.
    """
    import legate

    legate_mod_dir = Path(legate.__path__[0])
    legate_build_dir = get_legate_build_dir(legate_mod_dir.parent)

    if legate_build_dir is None:
        return get_legate_paths_from_installed_dir(legate_mod_dir)
    return get_legate_paths_from_build_dir(legate_build_dir)
