# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os
import re
import sys
import shutil
import subprocess
from pathlib import Path

from ._io import vprint
from ._legate_config import get_legate_config


def read_cmake_cache_value(pattern: str, cache_dir: Path | None = None) -> str:
    r"""Read a CMakeCache.txt value.

    Parameters
    ----------
    pattern : str
        A regex pattern to match the variable name
    cache_dir : Path, optional
        The directory containing the CMakeCache.txt. Defaults to
        the scikit-build cmake-build dir.

    Returns
    -------
    value : str
        The value of the cache variable

    Raises
    ------
    RuntimeError
        If the cache variable is not found
    """
    if cache_dir is None:
        cache_dir = get_legate_config().SKBUILD_BUILD_DIR
    file_path = cache_dir / "CMakeCache.txt"
    re_pat = re.compile(pattern)
    with file_path.open() as fd:
        for line in filter(re_pat.match, fd):
            return line.split("=")[1].strip()

    msg = f"ERROR: Did not find {pattern} in {file_path}"
    raise RuntimeError(msg)


def fix_env() -> None:
    r"""Fix up the environment, removing certain arguments from CMAKE_ARGS
    that scikit-build-core does not like, and setting default values for
    others based on the original C++ configure run.

    After this call CMAKE_ARGS is always defined in the environment.
    """
    try:
        # not sure how this could fail, but I suppose it could?
        gen = read_cmake_cache_value(
            "CMAKE_GENERATOR", cache_dir=get_legate_config().LEGATE_CMAKE_DIR
        )
        gen_provenance = str(get_legate_config().LEGATE_CMAKE_DIR)
    except (RuntimeError, FileNotFoundError):
        gen = os.environ.get("CMAKE_GENERATOR", "")
        gen_provenance = "environ"

    if gen:
        vprint(f"Setting cmake generator to {gen} (from {gen_provenance})")
        os.environ["CMAKE_GENERATOR"] = gen
    else:
        vprint("Cmake generator unknown, letting cmake decide")

    env_cmake_args = os.environ.get("CMAKE_ARGS", "")
    vprint(f"Found environment cmake args: {env_cmake_args}")
    # This is set by the configuration, we don't want e.g. Conda overriding
    # this.
    if env_cmake_args:
        vprint("Removing CMAKE_BUILD_TYPE from environment cmake args")
        env_cmake_args = re.sub(
            r"\-DCMAKE_BUILD_TYPE([:\w]*)=\w+", "", env_cmake_args
        )
        vprint("Removing CMAKE_INSTALL_PREFIX from environment cmake args")
        env_cmake_args = re.sub(
            r"\-DCMAKE_INSTALL_PREFIX([:\w]*)=\w+", "", env_cmake_args
        )
    os.environ["CMAKE_ARGS"] = env_cmake_args

    try:
        par = read_cmake_cache_value(
            "CMAKE_BUILD_PARALLEL_LEVEL",
            cache_dir=get_legate_config().LEGATE_CMAKE_DIR,
        )
    except (RuntimeError, FileNotFoundError):
        from multiprocessing import cpu_count

        par = str(cpu_count())

    par = os.environ.setdefault("CMAKE_BUILD_PARALLEL_LEVEL", par)
    vprint(f"Build using {par} CPU cores")


def was_built_with_build_isolation() -> bool:
    r"""Determine whether a possibly pre-existing build was built using
    PEP 517 build isolation enabled.

    Returns
    -------
    bool
        True if the previous build was built with build isolation, False
        otherwise.

    Notes
    -----
    This is a guess, and an agressive one at that. It could be a false
    positive, it could be a false negative, but it hasn't failed us yet!
    """
    skbuild_cache_file = (
        get_legate_config().SKBUILD_BUILD_DIR / "CMakeCache.txt"
    )
    return (
        skbuild_cache_file.exists()
        and "pip-build-env" in skbuild_cache_file.read_text()
    )


def building_with_build_isolation() -> bool:
    r"""Determine whether the current build is being built with build
    isolation.

    Returns
    -------
    bool
        True if being built under build isolation, False otherwise.

    Raises
    ------
    AssertionError
        If pip in unable to be located (should never happen).
    """
    pip_loc = shutil.which("pip")
    assert pip_loc is not None, "Could not locate pip"
    return "pip-build-env" in pip_loc


def get_original_python_executable() -> tuple[str, dict[str, str]]:
    r"""Retrieve a python executable and environment that corresponds to
    the original, non build-isolated environment.

    Returns
    -------
    str
        The full path to the original python interpreter
    dict[str, str]
        The environment such that a call to
        subprocess.run([python_exe, ...], env=env) exactly mimics the original
        environment.
    """
    if not building_with_build_isolation():
        return sys.executable, os.environ.copy()

    import sysconfig

    orig_bin_dir = Path(sysconfig.get_path("scripts"))
    vprint(f"Found original binary directory: {orig_bin_dir}")
    assert "pip-build-env" not in str(orig_bin_dir)
    for name in (
        "python3",
        "python",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
    ):
        maybe_python = orig_bin_dir / name
        vprint(f"Trying {maybe_python} for original python executable")
        if not maybe_python.exists():
            vprint(f"{maybe_python} does not exist")
            continue

        if not maybe_python.is_file():
            vprint(f"{maybe_python} is not a file")
            continue

        vprint(f"Success, using {maybe_python}")
        python_exe = str(maybe_python)
        break
    else:
        msg = (
            "Could not locate original python installation while building "
            "under build isolation"
        )
        raise RuntimeError(msg)

    remove = {"PYTHONNOUSERSITE", "PYTHONPATH"}
    env = {k: v for k, v in os.environ.items() if k not in remove}
    return python_exe, env


def clean_skbuild_dir() -> None:
    r"""Reduce the scikit-build directory to smithereens."""
    # Need to get the original python executable, because by the time this
    # function is called, we will already be in the fake temporary directory
    # made by pip, which technically does not have legate installed.
    python_exe, env = get_original_python_executable()

    uninstall_cmd = [
        python_exe,
        "-m",
        "pip",
        "uninstall",
        "-y",
        "legate",
        "--verbose",
    ]
    vprint(
        f"Running {uninstall_cmd} for a clean build to accommodate build "
        "isolation"
    )
    sys.stdout.flush()
    sys.stderr.flush()
    subprocess.check_call(uninstall_cmd, env=env)
    config = get_legate_config()
    for path in (
        config.SKBUILD_BUILD_DIR,
        config.LEGATE_DIR / "legate.egg-info",
        config.LEGATE_DIR / "src" / "python" / "legate.egg-info",
    ):
        vprint(f"Removing {path}")
        try:
            shutil.rmtree(path)
        except Exception as exn:
            vprint(f"Problem removing {path}: {exn}")
