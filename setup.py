#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
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

import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import skbuild
from setuptools import find_packages
from skbuild import setup

try:
    # Currently skbuild does not have a develop wrapper, but try them first in
    # case they ever make one
    from skbuild.command.develop import develop
except ModuleNotFoundError:
    from setuptools.command.develop import develop

from skbuild.command.bdist_wheel import bdist_wheel

from versioneer import (
    get_cmdclass as versioneer_get_cmdclass,
    get_version as versioneer_get_version,
)

if TYPE_CHECKING:
    from collections.abc import Generator
    from typing import Any, Final


BANNER: Final = f"xxx-- {80 * '='} --xxx"
MINI_BANNER: Final = len(BANNER) * "-"
ERROR_BANNER: Final = f"XXX== {80 * '='} ==XXX"
BUILD_MODE = "UNKNOWN" if len(sys.argv) <= 1 else sys.argv[1]

if BUILD_MODE.startswith("-"):
    BUILD_MODE = "UNKNOWN"

if BUILD_MODE == "editable_wheel":
    # HACK: scikit-build does not properly support PEP 660
    # (https://peps.python.org/pep-0660/) and the new editable install
    # mechanism fails to call the proper hooks. Specifically, setup.py develop
    # is never called, nor are the other build commands. There is a workaround,
    # which is to define the environment variable
    # SETUPTOOLS_ENABLE_FEATURES="legacy-editable" before calling pip, which
    # restores the original functionality.
    #
    # setuptools internally uses "editable_wheel" as the target name
    # for this, so if we find that, then someone has forgotten to export the
    # env var.
    if "legacy-editable" not in os.environ.get(
        "SETUPTOOLS_ENABLE_FEATURES", ""
    ):
        mess = (
            "\n"
            f"{ERROR_BANNER}"
            "\n"
            "PEP 660 editable installs are not yet fully "
            "supported by scikit-build. Chances are this will crash during "
            "installation, or not properly install the libraries. You must "
            'define SETUPTOOLS_ENABLE_FEATURES="legacy-editable" in your '
            "environment before running pip install. I.e."
            "\n"
            "\n"
            '$ SETUPTOOLS_ENABLE_FEATURES="legacy-editable" python3 -m pip '
            "install --editable . --your --other ---args"
            "\n"
            "\n"
            "or"
            "\n"
            "\n"
            '$ export SETPTOOLS_ENABLE_FEATURES="legacy-editable"'
            "\n"
            "$ python3 -m pip install --editable . --your --other ---args"
            "\n"
            f"{ERROR_BANNER}"
        )
        raise RuntimeError(mess)


def verbose_print(*args: Any, **kwargs: Any) -> None:
    print("x-- legate.internal:setup.py:", *args, **kwargs)


try:
    legate_dir = os.environ["LEGATE_DIR"]
except KeyError:
    from scripts.get_legate_dir import get_legate_dir

    legate_dir = get_legate_dir()

LEGATE_DIR: Final = Path(legate_dir).resolve(strict=True)

try:
    LEGATE_ARCH: Final = os.environ["LEGATE_ARCH"].strip()
    if not LEGATE_ARCH:
        raise KeyError("empty arch")
except KeyError as ke:
    try:
        from scripts.get_legate_arch import (  # type: ignore[import-not-found,unused-ignore] # noqa: E501
            get_legate_arch,
        )
    except ModuleNotFoundError:
        # User has not run configure yet
        raise RuntimeError(
            "\n"
            f"{ERROR_BANNER}"
            "\n"
            "Must export LEGATE_ARCH in environment before continuing "
            " and/or run configure"
            "\n"
            f"{ERROR_BANNER}"
        ) from ke

    LEGATE_ARCH = get_legate_arch()


LEGATE_ARCH_DIR: Final = LEGATE_DIR / LEGATE_ARCH
LEGATE_CMAKE_DIR: Final = LEGATE_ARCH_DIR / "cmake_build"

if not LEGATE_ARCH_DIR.exists() or not LEGATE_CMAKE_DIR.exists():
    if not (
        configure_cmd := LEGATE_ARCH_DIR / f"reconfigure-{LEGATE_ARCH}.py"
    ).exists():
        configure_cmd = LEGATE_DIR / "configure"
    raise RuntimeError(
        f"\n{ERROR_BANNER}\n"
        f"Current Legate arch '{LEGATE_ARCH}' either does not\n"
        "exist, or does not appear to have been configured with python\n"
        "bindings enabled. Please run the following before continuing:\n\n"
        f"$ {configure_cmd} --LEGATE_ARCH='{LEGATE_ARCH}' "
        "--with-python"
        f"\n{ERROR_BANNER}\n"
    )


def subprocess_check_call(*args: Any, **kwargs: Any) -> Any:
    r"""If the subprocess call results in any I/O, then it will appear out of
    order with respect to the python printing. So we need to explicitly flush
    our pipes first.
    """
    sys.stdout.flush()
    sys.stderr.flush()
    return subprocess.check_call(*args, **kwargs)


def monkey_patch_skbuild() -> None:
    r"""scikit-build likes to just plop down its "_skbuild" build directory at
    project top-level. This directory is where it puts all of its build
    artifacts, as well as any caches or other files.

    Ideally, we want scikit-build to put it in the arch folder (to keep all
    builds contained), but scikit-build provides no way (at least that I could
    find) of changing where that directory is, so we have to resort to
    monkey-patching it ourselves.
    """
    arch_skbuild_dir = os.path.join(
        LEGATE_ARCH, skbuild.constants.SKBUILD_DIR()
    )

    def new_skbuild_dir() -> str:
        return arch_skbuild_dir

    skbuild.constants.SKBUILD_DIR = new_skbuild_dir


def read_cmake_cache_value(pattern: str, cache_dir: Path | None = None) -> str:
    r"""Read a CMakeCache.txt value

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
        cache_dir = LEGATE_DIR / skbuild.constants.CMAKE_BUILD_DIR()
    file_path = cache_dir / "CMakeCache.txt"
    re_pat = re.compile(pattern)
    with file_path.open() as fd:
        for line in filter(re_pat.match, fd):
            return line.split("=")[1].strip()

    raise RuntimeError(f"ERROR: Did not find {pattern} in {file_path}")


def inject_sys_argv() -> None:
    r"""Inject various useful things into sys.argv

    Notes
    -----
    scikit-build takes a bunch of command-line arguments. Some it takes via
    setup() extensions, others it takes via global sys.argv. Thus, this
    function serves to inject these arguments into sys.argv before we reach
    the setup() call.
    """
    import multiprocessing as mp

    try:
        CMAKE_EXE = read_cmake_cache_value(
            "CMAKE_COMMAND", cache_dir=LEGATE_CMAKE_DIR
        )
    except RuntimeError:
        CMAKE_EXE = "cmake"

    extra_args = ["--cmake-executable", CMAKE_EXE, "-j", str(mp.cpu_count())]

    try:
        # not sure how this could fail, but I suppose it could?
        CMAKE_GENERATOR = read_cmake_cache_value(
            "CMAKE_GENERATOR", cache_dir=LEGATE_CMAKE_DIR
        )
    except RuntimeError:
        CMAKE_GENERATOR = os.environ.get("CMAKE_GENERATOR", "")

    if CMAKE_GENERATOR:
        extra_args.extend(["--skip-generator-test", "-G", CMAKE_GENERATOR])

    # Insert the extra args in a very specific place:
    #
    # sys.argv[0] = program name
    # sys.argv[1] = command, e.g. "develop" or "bdist"
    #
    # The rest of sys.argv is god knows what, including a e.g. user-set
    # --cmake-executable! We still want the user to be able to overrule the
    # default at runtime, so we cannot just append to sys.argv. Hence, we must
    # put it exactly here.
    sys.argv[2:2] = extra_args


def create_log_file() -> Path:
    r"""Sets up the log file for this invokation of setup.py

    Returns
    -------
    log_file : Path
        The log file.

    Notes
    -----
    This does not create the log file, but does ensure that any leading paths
    are created. That is, ``log_file.open()`` should return without error.
    """
    log_file = (
        LEGATE_DIR / skbuild.constants.SKBUILD_DIR() / f"{BUILD_MODE}.log"
    )
    log_file.parent.mkdir(parents=True, exist_ok=True)
    return log_file


@contextlib.contextmanager
def setup_tee(log_file: Path) -> Generator[str, None, None]:
    r"""Set up tee-ing of stdout and stderr to the log file

    Parameters
    ----------
    log_file : Path
        The log file to tee stdout and stderr file streams into.

    Yields
    ------
    stdout : TextIO
        The temporary stdout file handle
    stderr : TextIO
        The temporary stderr file handle

    Notes
    -----
    On systems that support it, this routine will use the standard ``tee``
    executable to perform the stream indirection. This ensures that any output
    by spawned subprocesses is also captured.

    For systems which do not support ``tee``, built-in contextlib redirection
    is used, which has no effect on the output of spawned subprocesses.
    """
    stdout = sys.stdout
    stderr = sys.stderr
    stdout.flush()
    stderr.flush()
    if tee_exec := shutil.which("tee"):
        # Note: this Popen must not be closed by python! Since we connect our
        # own stdout to it, the current process must first exit before it can
        # be closed.
        tee = subprocess.Popen([tee_exec, log_file], stdin=subprocess.PIPE)
        assert tee.stdin is not None  # pacify mypy
        tee_stdin = tee.stdin.fileno()
        # Cause tee's stdin to get a copy of our stdin/stdout (as well as that
        # of any child processes we spawn)
        os.dup2(tee_stdin, stdout.fileno())
        os.dup2(tee_stdin, stderr.fileno())
        yield "tee"
        stdout.flush()
        stderr.flush()
    else:
        with (
            log_file.open("w") as fd,
            contextlib.redirect_stdout(fd),
            contextlib.redirect_stderr(fd),
        ):
            yield "contextlib"


def fixup_manifest(manifest: list[str]) -> list[str]:
    r"""Fixup the scikit-build generated manifest file

    Parameters
    ----------
    manifest : list[str]
        The list of files to be added to the manifest

    Returns
    -------
    manifest : list[str]
        The new manifest

    Notes
    -----
    scikit-build uses the install_manifest.txt generated by a call to cmake
    --install to determine which files to put into the python wheel.

    If the CMake build found the legate C++ libraries via the *pre-built*
    path (i.e. they exist in LEGATE_DIR / LEGATE_ARCH / cmake_build),
    then those libs are considered IMPORTED by CMake and won't be automatically
    installed. Instead, we insert an install() command directly into
    cmake_install.cmake for the Python binding, which installs the C++ libs
    into the scikit-build install dir.

    But we still need to tell scikit-build about this, otherwise it won't
    include those files in the wheel.
    """
    found_method = read_cmake_cache_value("_legate_FOUND_METHOD")
    verbose_print(f"Found legate method: {found_method}")
    if found_method != "PRE_BUILT":
        # If SELF_BUILT then the cmake --install command will have already
        # installed them for us, and if INSTALLED then there is nothing to do.
        return manifest

    cpp_manifest = (
        (Path(read_cmake_cache_value("legate_DIR")) / "install_manifest.txt")
        .read_text()
        .splitlines()
    )
    # scikit-build checks each of the manifest files for whether they originate
    # from within the cmake-build folder. But their check is stupid, because
    # they only check that the path starts with `SKBUILD_DIR()`, which is a
    # *relative* path. So all their files are
    #
    # [
    #   'test-arch/_skbuild/.../cmake-install/foo.h' ,
    #   'test-arch/_skbuild/.../cmake-install/bar.h'
    # ]
    #
    # while ours are
    #
    # [
    #   '/full/path/to/test-arch/_skbuild/.../cmake-install/baz.h' ,
    #   '/full/path/to/test-arch/_skbuild/.../cmake-install/bop.h'
    # ]
    #
    # Obviously, these all point to the same cmake-install directory, but
    # scikit-build doesn't grok this. So we have to do this song and dance and
    # strip out '/full/path/to'.
    cpp_manifest = [str(Path(p).relative_to(LEGATE_DIR)) for p in cpp_manifest]
    manifest.extend(cpp_manifest)
    return manifest


def was_built_with_build_isolation() -> bool:
    r"""Determine whether a possibly pre-existing build was built using
    PEP 517 build isolation enabled.

    Returns
    -------
    ret : bool
        True if the previous build was built with build isolation, False
        otherwise.

    Notes
    -----
    This is a guess, and an agressive one at that. It could be a false
    positive, it could be a false negative, but it hasn't failed us yet!
    """
    skbuild_cache_file = (
        LEGATE_DIR / skbuild.constants.CMAKE_BUILD_DIR() / "CMakeCache.txt"
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
    ret : bool
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
    python_exe : str
        The full path to the original python interpreter
    env : dict[str, str]
        The environment such that a call to
        subprocess.run([python_exe, ...], env=env) exactly mimics the original
        environment.
    """
    if not building_with_build_isolation():
        return sys.executable, os.environ.copy()

    import sysconfig

    orig_bin_dir = Path(sysconfig.get_path("scripts"))
    verbose_print(f"Found original binary directory: {orig_bin_dir}")
    assert "pip-build-env" not in str(orig_bin_dir)
    for name in (
        "python3",
        "python",
        f"python{sys.version_info.major}.{sys.version_info.minor}",
    ):
        maybe_python = orig_bin_dir / name
        verbose_print(f"Trying {maybe_python} for original python executable")
        if not maybe_python.exists():
            verbose_print(f"{maybe_python} does not exist")
            continue

        if not maybe_python.is_file():
            verbose_print(f"{maybe_python} is not a file")
            continue

        verbose_print(f"Success, using {maybe_python}")
        python_exe = str(maybe_python)
        break
    else:
        raise RuntimeError(
            "Could not locate original python installation while building "
            "under build isolation"
        )

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
    verbose_print(
        f"Running {uninstall_cmd} for a clean build to accommodate build "
        "isolation"
    )
    subprocess_check_call(uninstall_cmd, env=env)
    for path in (
        LEGATE_DIR / skbuild.constants.CMAKE_INSTALL_DIR(),
        LEGATE_DIR / skbuild.constants.SETUPTOOLS_INSTALL_DIR(),
        LEGATE_DIR / "legate.egg-info",
    ):
        verbose_print(f"Removing {path}")
        try:
            shutil.rmtree(path)
        except Exception as exn:
            verbose_print(f"Problem removing {path}: {exn}")


def do_setup_impl() -> None:
    cmake_args = []
    if BUILD_MODE == "bdist_wheel" and was_built_with_build_isolation():
        # Explicitly uninstall legate if doing a clean/isolated build.
        #
        # A prior installation may have built and installed legate C++
        # dependencies.
        #
        # CMake will find and use them for the current build, which would
        # normally be correct, but pip uninstalls files from any existing
        # installation as the last step of the install process, including the
        # libraries found by CMake during the current build.
        #
        # Therefore this uninstall step must occur *before* CMake attempts to
        # find these dependencies, triggering CMake to build and install them
        # again.
        clean_skbuild_dir()
        cmake_args.append("--fresh")

    if BUILD_MODE != "UNKNOWN":
        cmake_args.append(f"-Dlegate_SETUP_PY_MODE:INTERNAL={BUILD_MODE}")
    # The below is needed because when pip installs the packages, it does
    # so in an isolated environment. But cmake (i.e. scikit-build) doesn't
    # know that, and so it searches next to the python executable, i.e. it
    # looks for dirname(which("python3")) / "cython". It won't (yet) find
    # cython there, and so it barfs:
    #
    # CMake Error at
    # /private/var/folders/0m/7yv6pgt57vzdqfbzcr8lztkh0000gp/T/pip-build-env-foj45ssx/overlay/lib/python3.11/site-packages/skbuild/resources/cmake/FindCython.cmake:71
    # (message):
    #  Command
    #    "/Users/jfaibussowit/soft/nv/legate.core.internal/arch_venv/bin/cython;--version"
    #  failed with output:
    #    ...
    if cython_exec := shutil.which("cython"):
        verbose_print(f"Found cython: {cython_exec}")
        cmake_args.append(f"-DCYTHON_EXECUTABLE:PATH={cython_exec}")

    env_cmake_args = os.environ.get("CMAKE_ARGS", "")
    verbose_print(f"Found environment cmake args: {env_cmake_args}")
    # This is set by the configuration, we don't want e.g. Conda overriding
    # this.
    if env_cmake_args:
        verbose_print("Removing CMAKE_BUILD_TYPE from environment cmake args")
        env_cmake_args = re.sub(
            r"\-DCMAKE_BUILD_TYPE([:\w]*)=\w+", "", env_cmake_args
        )
        verbose_print(
            "Removing CMAKE_INSTALL_PREFIX from environment cmake args"
        )
        env_cmake_args = re.sub(
            r"\-DCMAKE_INSTALL_PREFIX([:\w]*)=\w+", "", env_cmake_args
        )
        os.environ["CMAKE_ARGS"] = env_cmake_args

    # We wish to configure the build using exactly the same arguments as the
    # C++ lib so that ./configure options are respected.
    cmd_spec_path = LEGATE_CMAKE_DIR / "aedifix_cmake_command_spec.json"
    verbose_print(f"Using cmake_command file: {cmd_spec_path}")
    with cmd_spec_path.open() as fd:
        spec_cmake_cmds = json.load(fd)["CMAKE_COMMANDS"]
    verbose_print(f"Adding line to cmake args: {spec_cmake_cmds}")
    cmake_args.extend(
        arg for arg in spec_cmake_cmds if "CMAKE_INSTALL_PREFIX" not in arg
    )

    print(BANNER)
    print(MINI_BANNER)
    print(f"sys.argv = {sys.argv}")
    version = versioneer_get_version()
    print(f"Using version: {version}")
    packages = find_packages(
        where="src/python", include=["legate", "legate.*"]
    )
    package_data = {pack: ["py.typed", "*.pyi", "*.so"] for pack in packages}
    print("Using package data:")
    for pack, data in package_data.items():
        print(f"{pack} = {data}")
    package_dir = {"": "src/python"}
    print("Using package dir:")
    for dirname, redirect in package_dir.items():
        print("dir:", repr(dirname), "->", redirect)
    print("Using command class:")
    cmd_class = versioneer_get_cmdclass(
        {"develop": develop, "bdist_wheel": bdist_wheel}
    )
    print("\n".join(f"{key} = {value}" for key, value in cmd_class.items()))
    print(f"Using cmake args: {cmake_args}")
    print(f"Using environment cmake args: {env_cmake_args}")
    languages = ("CXX",)
    print(f"Using languages: {languages}")
    print(MINI_BANNER)
    print(BANNER, flush=True)

    setup(
        version=version,
        cmdclass=cmd_class,
        cmake_args=cmake_args,
        cmake_languages=languages,
        cmake_process_manifest_hook=fixup_manifest,
        # have to duplicate this from pyproject.toml because scikit-build
        # expects to modify its contents...
        packages=packages,
        # Need to specify these twice (both here and in pyproject.toml) because
        # scikit-build does not read pyproject.toml
        package_data=package_data,
        include_package_data=True,
        package_dir=package_dir,
    )


def do_setup() -> None:
    monkey_patch_skbuild()
    inject_sys_argv()

    log_file = create_log_file()

    try:
        with setup_tee(log_file) as tee_impl:
            print(BANNER)
            rn = time.strftime("%a, %d %b %Y %H:%M:%S %z")
            print(f"Starting {BUILD_MODE} run at {rn}")
            print("System info:")
            print(platform.uname())
            print(f"Using {tee_impl!r} as log tee-ing implementation")
            print(BANNER)

            do_setup_impl()
    except Exception as exn:
        mess = (
            ERROR_BANNER,
            f"ERROR: {exn}",
            "",
            f"Please see {log_file} for further details!",
            ERROR_BANNER,
        )
        print("\n".join(mess), flush=True)
        raise


do_setup()
