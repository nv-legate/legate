#!/usr/bin/env python3

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

import contextlib
import hashlib
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

from setuptools import find_packages
from skbuild import constants as sk_constants, setup

try:
    # Currently skbuild does not have a develop wrapper, but try them first in
    # case they ever make one
    from skbuild.command.develop import develop as orig_develop
except ModuleNotFoundError:
    from setuptools.command.develop import develop as orig_develop

from skbuild.command.bdist_wheel import bdist_wheel as orig_bdist_wheel

from versioneer import (
    get_cmdclass as versioneer_get_cmdclass,
    get_version as versioneer_get_version,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Iterator
    from typing import Any


def verbose_print(*args: Any, **kwargs: Any) -> None:
    print("x-- legate.core.internal:setup.py:", *args, **kwargs)


BANNER = f"xxx-- {80 * '='} --xxx"
ERROR_BANNER = f"XXX== {80 * '='} ==XXX"
BUILD_MODE = "UNKNOWN" if len(sys.argv) <= 1 else sys.argv[1]

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
        raise RuntimeError(
            f"\n{ERROR_BANNER}\n"
            "PEP 660 editable installs are not yet fully "
            "supported by scikit-build. Chances are this will crash during "
            "installation, or not properly install the libraries. You must "
            'define SETUPTOOLS_ENABLE_FEATURES="legacy-editable" in your '
            "environment before running pip install. I.e.\n\n"
            '$ SETUPTOOLS_ENABLE_FEATURES="legacy-editable" python3 -m pip '
            "install --editable . --your --other ---args\n\nor\n\n"
            '$ export SETPTOOLS_ENABLE_FEATURES="legacy-editable"\n'
            "$ python3 -m pip install --editable . --your --other ---args\n"
            f"{ERROR_BANNER}"
        )

if BUILD_MODE.startswith("-"):
    BUILD_MODE = "UNKNOWN"

try:
    lg_core_dir = os.environ["LEGATE_CORE_DIR"]
except KeyError as ke:
    raise RuntimeError(
        "ERROR: Must export LEGATE_CORE_DIR in environment before continuing"
    ) from ke
else:
    LEGATE_CORE_DIR = Path(lg_core_dir).resolve(strict=True)

try:
    LEGATE_CORE_ARCH = os.environ["LEGATE_CORE_ARCH"]
except KeyError as ke:
    raise RuntimeError(
        "ERROR: must export LEGATE_CORE_ARCH in environment before continuing"
    ) from ke

LEGATE_CORE_ARCH_DIR = LEGATE_CORE_DIR / LEGATE_CORE_ARCH
LEGATE_CORE_CMAKE_DIR = LEGATE_CORE_ARCH_DIR / "cmake_build"

if not LEGATE_CORE_ARCH_DIR.exists() or not LEGATE_CORE_CMAKE_DIR.exists():
    if not (
        configure_cmd := LEGATE_CORE_ARCH_DIR
        / f"reconfigure-{LEGATE_CORE_ARCH}.py"
    ).exists():
        configure_cmd = LEGATE_CORE_DIR / "configure"
    raise RuntimeError(
        f"\n{ERROR_BANNER}\n"
        f"Current Legate.Core arch '{LEGATE_CORE_ARCH}' either does not\n"
        "exist, or does not appear to have been configured with python\n"
        "bindings enabled. Please run the following before continuing:\n\n"
        f"$ {configure_cmd} --LEGATE_CORE_ARCH='{LEGATE_CORE_ARCH}' "
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
        LEGATE_CORE_ARCH, sk_constants.SKBUILD_DIR()
    )

    def new_skbuild_dir() -> str:
        return arch_skbuild_dir

    sk_constants.SKBUILD_DIR = new_skbuild_dir


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
        cache_dir = LEGATE_CORE_DIR / sk_constants.CMAKE_BUILD_DIR()
    file_path = cache_dir / "CMakeCache.txt"
    re_pat = re.compile(pattern)
    with file_path.open() as fd:
        for line in filter(re_pat.match, fd):
            return line.split("=")[1].strip()

    raise RuntimeError(f"ERROR: Did not find {pattern} in {file_path}")


try:
    CMAKE_EXE = read_cmake_cache_value(
        "CMAKE_COMMAND", cache_dir=LEGATE_CORE_CMAKE_DIR
    )
except RuntimeError:
    CMAKE_EXE = "cmake"


def inject_sys_argv() -> None:
    r"""Inject various useful things into sys.argv

    Notes
    -----
    scikit-build takes a bunch of command-line arguments. Some it takes via
    setup() extensions, others it takes via global sys.argv. Thus, this
    function serves to inject these arguments into sys.argv before we reach
    the setup() call.
    """
    extra_args = ["--cmake-executable", CMAKE_EXE]
    try:
        # not sure how this could fail, but I suppose it could?
        CMAKE_GENERATOR = read_cmake_cache_value(
            "CMAKE_GENERATOR", cache_dir=LEGATE_CORE_CMAKE_DIR
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


def get_install_dir() -> Path:
    def try_env_var(
        varname: str, extra: Callable[[], bool] | None = None
    ) -> Path | None:
        if extra is None:

            def default_extra() -> bool:
                return True

            extra = default_extra

        verbose_print(f"Trying {varname} for install dir")
        try:
            if (install_dir := os.environ[varname]) and extra():
                verbose_print(f"Success: {varname} = {install_dir}")
                return Path(install_dir)
        except KeyError:
            verbose_print(f"Rejected {varname}, not defined in environment")
            return

        if install_dir:
            verbose_print(
                f"Rejected {varname}, extra requirements not satisfied"
            )
        else:
            verbose_print(
                f"Rejected {varname}, defined in environment, but empty!"
            )

    if install_dir := try_env_var(
        "PREFIX", extra=lambda: os.environ.get("CONDA_BUILD", "") == "1"
    ):
        return install_dir
    if install_dir := try_env_var("CONDA_PREFIX"):
        return install_dir
    if install_dir := try_env_var("VIRTUAL_ENV"):
        return install_dir
    # This will happen if we try to install to system python prefix without
    # a virtual env or the like. I have no idea how to reliably get the
    # path to the system install, because neither pip nor setuptools
    # exposes it to us.
    raise RuntimeError("Could not identify install directory!")


def install_legion_python_bindings() -> None:
    r"""Install the Legion python bindings if needed."""

    def get_legion_py_bindings_path() -> Path | None:
        verbose_print("Attempting to install Legion python bindings")
        found_method = read_cmake_cache_value("_legate_core_FOUND_METHOD")
        verbose_print(f"Found legate core method: {found_method}")
        match found_method:
            case "PRE_BUILT":
                # The legate.core bindings were already built by a previous
                # call to "make" by the user and were imported by cmake. They
                # live in LEGATE_CORE_ARCH/cmake_build, and legate_core_DIR
                # points there.
                return (
                    Path(read_cmake_cache_value("legate_core_DIR"))
                    / "_deps"
                    / "legion-build"
                    / "bindings"
                    / "python"
                )
            case "SELF_BUILT":
                # We built the legate.core bindings ourselves as part of the
                # bdist_wheel step. Legion
                return Path(
                    read_cmake_cache_value("LegionBindings_python_BINARY_DIR")
                )
            case "INSTALLED":
                return

        raise RuntimeError(
            "Unhandled find-method for legate.core encountered:"
            f" {found_method}"
        )

    try:
        bdir = get_legion_py_bindings_path()
    except RuntimeError as rte:
        # If this fails then I am pretty sure this means that while we did
        # build legate.core, we didn't build Legion. Conceivably this means
        # that Legion is already installed, but I guess it could also occur if
        # we passed a source directory?
        raise RuntimeError(
            f"{rte}\n\nThis would seem to indicate that "
            "legate.core did not build Legion. Please report this "
            "to legate.core developers immediately, and be sure "
            "to include the appriopriate log file!"
        ) from rte
    if bdir is None:
        # Pre-installed legate.core. Not sure how well the following assumption
        # holds.
        verbose_print(
            "Legate core appears to have been pre-installed. Assuming that "
            "legion python bindings are as well and bailing!"
        )
        return

    install_dir = get_install_dir()
    cmake_cmd = [CMAKE_EXE, "--install", bdir, "--prefix", install_dir]
    verbose_print(f"Running: {cmake_cmd}")
    subprocess_check_call(cmake_cmd)

    # pip is able to uninstall most installed packages. Known exceptions
    # are:
    #
    # - Pure distutils packages installed with python setup.py install,
    #   which leave behind no metadata to determine what files were
    #   installed.
    # - Script wrappers installed by python setup.py develop.
    #
    # We are potentially both of these! Pip internally uses a "dist-info"
    # directory in the site-packages directory to track which files a
    # particular package has installed. In particular it uses
    # <PACKAGE>-<VERSION>-dist-info/RECORD, which is essentially a list of
    # "path,file_hash,file_size":
    #
    # https://packaging.python.org/en/latest/specifications/recording-installed-packages/#the-record-file
    #
    # Our goal is therefore to append all of the crud installed by the
    # above cmake install command to this record file, so that pip
    # uninstall legate/legion just works as expected.
    def gen_uninstall_paths() -> Iterator[Path]:
        yield from (install_dir / "lib").glob("liblegion_canonical_python*")
        yield install_dir / "share" / "Legion" / "python"
        yield install_dir / "bin" / "legion_python"

    def sha256sum(filename: str | Path, buffer_size: int = 128 * 1024) -> str:
        hasher = hashlib.sha256()
        mv = memoryview(bytearray(buffer_size))
        with open(filename, "rb", buffering=0) as fd:
            while n := fd.readinto(mv):
                hasher.update(mv[:n])
        return hasher.hexdigest()

    records = []
    for file_path in gen_uninstall_paths():
        verbose_print(f"Adding uninstall path: {file_path}")
        file_hash = (
            "" if file_path.is_dir() else f"sha256={sha256sum(file_path)}"
        )
        records.append(f"{file_path},{file_hash},{file_path.stat().st_size}")

    site_package_dir = (
        install_dir
        / "lib"
        / f"python{sys.version_info.major}.{sys.version_info.minor}"
        / "site-packages"
    )
    assert site_package_dir.exists(), f"{site_package_dir} does not exist"
    lg_distinfo_re = re.compile(r"legion.*dist-info$")
    verbose_print(
        f"Searching site package dir: {site_package_dir} for legion "
        "dist-info"
    )
    for dist_info in site_package_dir.iterdir():
        if dist_info.is_dir() and lg_distinfo_re.match(dist_info.name):
            verbose_print(f"Found legion dist-info dir: {dist_info}")
            break
    else:
        raise RuntimeError(
            f"Failed to find legion dist-info file in {site_package_dir}"
        )

    record_file = dist_info / "RECORD"
    if not record_file.exists():
        raise RuntimeError(
            f"RECORD file: {record_file} does not exist post legion "
            "python bindings install?"
        )
    verbose_print(f"Appending new records to RECORD file ({record_file}):")
    verbose_print("\n".join(records))
    record_file.write_text(
        "\n".join(filter(None, record_file.read_text().splitlines() + records))
    )
    verbose_print("Finished adding records")


class Develop(orig_develop):
    def run(self, *args: Any, **kwargs: Any) -> Any:
        verbose_print(f"Running {orig_develop} develop command")
        ret = super().run(*args, **kwargs)
        install_legion_python_bindings()
        return ret


class BdistWheel(orig_bdist_wheel):
    def run(self, *args: Any, **kwargs: Any) -> Any:
        verbose_print(f"Running {orig_bdist_wheel} develop command")
        ret = super().run(*args, **kwargs)
        install_legion_python_bindings()
        return ret


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
        LEGATE_CORE_DIR / sk_constants.SKBUILD_DIR() / f"{BUILD_MODE}.log"
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

    If the CMake build found the legate.core C++ libraries via the *pre-built*
    path (i.e. they exist in LEGATE_CORE_DIR / LEGATE_CORE_ARCH / cmake_build),
    then those libs are considered IMPORTED by CMake and won't be automatically
    installed. Instead, we insert an install() command directly into
    cmake_install.cmake for the Python core, which installs the C++ libs into
    the scikit-build install dir.

    But we still need to tell scikit-build about this, otherwise it won't
    include those files in the wheel.
    """
    found_method = read_cmake_cache_value("_legate_core_FOUND_METHOD")
    verbose_print(f"Found legate core method: {found_method}")
    if found_method != "PRE_BUILT":
        # If SELF_BUILT then the cmake --install command will have already
        # installed them for us, and if INSTALLED then there is nothing to do.
        return manifest

    cpp_manifest = (
        (
            Path(read_cmake_cache_value("legate_core_DIR"))
            / "install_manifest.txt"
        )
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
    cpp_manifest = [
        str(Path(p).relative_to(LEGATE_CORE_DIR)) for p in cpp_manifest
    ]
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
        LEGATE_CORE_DIR / sk_constants.CMAKE_BUILD_DIR() / "CMakeCache.txt"
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
    python_exe, env = get_original_python_executable()

    uninstall_cmd = [
        python_exe,
        "-m",
        "pip",
        "uninstall",
        "-y",
        "legate-core",
        "legion",
        "--verbose",
    ]
    verbose_print(
        f"Running {uninstall_cmd} for a clean build to accommodate build "
        "isolation"
    )
    subprocess_check_call(uninstall_cmd, env=env)
    for path in (
        # LEGATE_CORE_DIR / sk_constants.CMAKE_BUILD_DIR(),
        LEGATE_CORE_DIR / sk_constants.CMAKE_INSTALL_DIR(),
        LEGATE_CORE_DIR / sk_constants.SETUPTOOLS_INSTALL_DIR(),
        LEGATE_CORE_DIR / "legate_core.egg-info",
    ):
        verbose_print(f"Removing {path}")
        try:
            shutil.rmtree(path)
        except Exception as exn:
            verbose_print(f"Problem removing {path}: {exn}")


def do_setup_impl() -> None:
    cmake_args = []
    if BUILD_MODE == "bdist_wheel" and was_built_with_build_isolation():
        # Explicitly uninstall legate.core and Legion if doing a clean/isolated
        # build.
        #
        # A prior installation may have built and installed legate.core C++
        # dependencies (like Legion).
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
        cmake_args.append(f"-Dlegate_core_SETUP_PY_MODE:INTERNAL={BUILD_MODE}")
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
        verbose_print("Removing CMAKE_BUILD_TYPE from environment cmake args!")
        env_cmake_args = re.sub(
            r"\-DCMAKE_BUILD_TYPE([:\w]*)=\w+", "", env_cmake_args
        )
        os.environ["CMAKE_ARGS"] = env_cmake_args

    # We wish to configure the build using exactly the same arguments as the
    # C++ lib so that ./configure options are respected.
    cmd_spec_path = LEGATE_CORE_CMAKE_DIR / "aedifix_cmake_command_spec.json"
    verbose_print(f"Using cmake_command file: {cmd_spec_path}")
    with cmd_spec_path.open() as fd:
        spec_cmake_cmds = json.load(fd)["CMAKE_COMMANDS"]
    verbose_print(f"Adding line to cmake args: {spec_cmake_cmds}")
    cmake_args.extend(spec_cmake_cmds)

    MINI_BANNER = len(BANNER) * "-"
    print(BANNER)
    print(MINI_BANNER)
    print(f"sys.argv = {sys.argv}")
    version = versioneer_get_version()
    print(f"Using version: {version}")
    packages = find_packages(where=".", include=["legate", "legate.*"])
    print("Using packages:")
    print("\n".join(packages))
    print("Using command class:")
    cmd_class = versioneer_get_cmdclass(
        {"develop": Develop, "bdist_wheel": BdistWheel}
    )
    print("\n".join(f"{key} = {value}" for key, value in cmd_class.items()))
    print(f"Using cmake args: {cmake_args}")
    print(f"Using environment cmake args: {env_cmake_args}")
    languages = ("CXX",)
    print(f"Using languages: {languages}")
    print(MINI_BANNER)
    print(BANNER)

    if (
        read_cmake_cache_value(
            "Legion_USE_Python", cache_dir=LEGATE_CORE_CMAKE_DIR
        )
        != "ON"
    ):
        raise RuntimeError(
            "Invalid configuration, must rerun configure with "
            '"--with-python" flag enabled!'
        )

    setup(
        version=version,
        # have to duplicate this from pyproject.toml because scikit-build
        # expects to modify its contents...
        packages=packages,
        cmdclass=cmd_class,
        cmake_args=cmake_args,
        cmake_languages=languages,
        include_package_data=True,
        cmake_process_manifest_hook=fixup_manifest,
    )


def do_setup() -> None:
    monkey_patch_skbuild()
    inject_sys_argv()

    log_file = create_log_file()

    try:
        with setup_tee(log_file) as tee_impl:
            print(BANNER)
            print(
                f"Starting {BUILD_MODE} run at "
                f"{time.strftime('%a, %d %b %Y %H:%M:%S %z')}",
            )
            print("System info:")
            print(platform.uname())
            print(f"Using '{tee_impl}' as log tee-ing implementation")
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
