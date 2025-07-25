#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import tomllib
import platform
from argparse import Action, ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from textwrap import indent
from typing import Literal

# --- Types -------------------------------------------------------------------

Req = str
Reqs = tuple[Req, ...]
OSType = Literal["linux", "osx"]

MAX_SANITIZER_VERSION = (11, 4)
MAX_SANITIZER_VERSION_STR = ".".join(map(str, MAX_SANITIZER_VERSION))


def V(version: str) -> tuple[int, ...]:
    padded_version = ([*version.split("."), "0", "0"])[:3]
    return tuple(int(x) for x in padded_version)


def drop_patch(version: str) -> str:
    return ".".join(version.split(".")[:2])


def normalize_platform_arch() -> str:
    match arch := platform.machine():
        case "x86_64":
            return "64"
        case "aarch64":
            return arch
        case _:
            msg = f"Unknown platform architecture: {arch}"
            raise RuntimeError(msg)


class SectionConfig:
    header: str

    @property
    def conda(self) -> Reqs:
        return ()

    @property
    def pip(self) -> Reqs:
        return ()

    def __str__(self) -> str:
        return self.header

    def format(self, kind: str) -> str:
        reqs = "- " + "\n- ".join(self.conda if kind == "conda" else self.pip)
        return SECTION_TEMPLATE.format(header=self.header, reqs=reqs)


@dataclass(frozen=True)
class CUDAConfig(SectionConfig):
    ctk_version: str | None
    compilers: bool
    os: OSType
    cupynumeric: bool

    header = "cuda"

    @property
    def conda(self) -> Reqs:
        if not self.ctk_version:
            return ()

        deps = (
            f"cuda-version={drop_patch(self.ctk_version)}",  # runtime
            "cutensor>=2",  # runtime
            "nccl<2.27.0",  # runtime
            "pynvml",  # tests
        )

        if V(self.ctk_version) < (12, 0, 0):
            deps += (f"cudatoolkit={self.ctk_version}",)
        else:
            deps += (
                "cuda-cudart-dev",
                "cuda-driver-dev",
                "cuda-nvml-dev",
                "cuda-nvtx-dev",
                "cuda-cupti-dev",
                "libcufile-dev",
                "libcal-dev",
            )
            if self.cupynumeric:
                deps += (
                    "cuda-cudart-static",
                    "libcublas-dev",
                    "libcufft-dev",
                    "libcurand-dev",
                    "libcusolver-dev",
                    "libcusparse-dev",
                    "libnvjitlink-dev",
                )

        if self.compilers and self.os == "linux":
            if V(self.ctk_version) < (12, 0, 0):
                arch = normalize_platform_arch()
                deps += (f"nvcc_linux-{arch}={drop_patch(self.ctk_version)}",)
            else:
                deps += ("cuda-nvcc",)

            # gcc 11.3 is incompatible with nvcc < 11.6.
            if V(self.ctk_version) < (11, 6, 0):
                deps += ("gcc<=11.2", "gxx<=11.2")
            else:
                deps += ("gcc=11.*", "gxx=11.*")

        return deps

    def __str__(self) -> str:
        if not self.ctk_version:
            return ""

        return f"-cuda{self.ctk_version}"


@dataclass(frozen=True)
class BuildConfig(SectionConfig):
    compilers: bool
    openmpi: bool
    ucx: bool
    sanitizers: bool
    os: OSType
    cupynumeric: bool

    header = "build"

    @property
    def conda(self) -> Reqs:
        pkgs = (
            # 3.25.0 triggers gitlab.kitware.com/cmake/cmake/-/issues/24119
            "cmake>=3.24,!=3.25.0",
            "cython>=3.0.1",
            "git",
            "make",
            "rust",
            "ninja",
            "openssl",
            "pkg-config",
            "scikit-build>=0.13.1",
            # see https://github.com/nv-legate/cupynumeric.internal/issues/704
            # for more details why do we restrict to <=75.3.0
            "setuptools>60,<=75.3.0",
            "zlib",
            "numba",
            "libhwloc=*=*default*",
            # Brings tcmalloc into the environment, which can be
            # optionally used by legate invocations.
            "gperftools",
            "hdf5",
        )
        if self.compilers:
            pkgs += ("c-compiler", "cxx-compiler")
        if self.openmpi:
            # Using a more recent version of OpenMPI in combination with the
            # system compilers fails with: "Could NOT find MPI (missing:
            # MPI_CXX_FOUND CXX)". The reason is that conda-forge's libmpi.so
            # v5 is linked against the conda-forge libstdc++.so. CMake will not
            # use the mpicc-suggested host compiler (conda-forge's gcc) when
            # running the FindMPI tests. Instead it will use the "main"
            # compiler it was configured with, i.e. the system compiler, which
            # links against the system libstdc++.so, causing libmpi.so's symbol
            # version checks to fail. Using v5+ OpenMPI from conda-forge
            # requires using the conda-forge compilers.
            pkgs += ("openmpi<5",)
        if self.ucx:
            pkgs += ("ucx>=1.16",)
        if self.sanitizers:
            pkgs += (f"libsanitizer<={MAX_SANITIZER_VERSION_STR}",)
        if self.os == "linux":
            pkgs += ("elfutils",)
        return sorted(pkgs)

    def __str__(self) -> str:
        val = "-compilers" if self.compilers else ""
        val += "-openmpi" if self.openmpi else ""
        val += "-ucx" if self.ucx else ""
        if self.sanitizers:
            val += "-sanitizer"
        if self.cupynumeric:
            val += "-cupynumeric"
        return val


@dataclass(frozen=True)
class RuntimeConfig(SectionConfig):
    sanitizers: bool
    openmpi: bool

    header = "runtime"

    @property
    def conda(self) -> Reqs:
        pkgs = (
            "llvm-openmp",
            "numpy>=1.22,!=2.1.0",
            "libblas=*=*openblas*",
            "openblas=*=*openmp*",
            # work around https://github.com/StanfordLegion/legion/issues/1500
            "openblas<=0.3.21",
            "opt_einsum",
            "scipy",
            "libhwloc=*=*default*",
            "hdf5",
            "h5py",
            # FIXME(wonchanl): Kerchunk needs to be updated for Zarr v3
            "zarr<3",
            "fsspec",
            "kerchunk",
        )
        if self.sanitizers:
            pkgs += (f"libsanitizer<={MAX_SANITIZER_VERSION_STR}",)
        if self.openmpi:
            # see https://github.com/spack/spack/issues/18084
            pkgs += ("openssh",)
        return pkgs


@dataclass(frozen=True)
class TestsConfig(SectionConfig):
    header = "tests"

    @property
    def conda(self) -> Reqs:
        return (
            "clang-tools>=8",
            "clang>=8",
            "coverage",
            "libhwloc=*=*default*",
            "mock",
            "mypy>=0.961",
            "pre-commit",
            "psutil",
            "pynvml",
            "pytest-cov",
            "pytest-mock",
            "pytest",
            "rich",
            "tifffile",
            "types-docutils",
        )

    @property
    def pip(self) -> Reqs:
        return ()


@dataclass(frozen=True)
class DocsConfig(SectionConfig):
    header = "docs"

    @property
    def conda(self) -> Reqs:
        return ("pandoc", "doxygen", "ipython", "jinja2", "markdown")

    # Use pip for sphinx and breathe deps. Need Sphinx>8 for the NV theme
    # but conda breath requires Sphinx<=7.2 even though things work.
    @property
    def pip(self) -> Reqs:
        return (
            "breathe>=4.35.0",
            "myst-parser",
            "nbsphinx",
            "nvidia-sphinx-theme",
            "sphinx>=8.2",
            "sphinx-copybutton",
        )


@dataclass(frozen=True)
class EnvConfig:
    use: str
    os: OSType
    ctk_version: str | None
    compilers: bool
    openmpi: bool
    ucx: bool
    sanitizers: bool
    cupynumeric: bool

    @property
    def channels(self) -> str:
        channels = []
        channels.append("conda-forge")
        return "- " + "\n- ".join(channels)

    @property
    def sections(self) -> tuple[SectionConfig, ...]:
        return (self.cuda, self.build, self.runtime, self.tests, self.docs)

    @property
    def cuda(self) -> CUDAConfig:
        return CUDAConfig(
            self.ctk_version, self.compilers, self.os, self.cupynumeric
        )

    @property
    def build(self) -> BuildConfig:
        return BuildConfig(
            self.compilers,
            self.openmpi,
            self.ucx,
            self.sanitizers,
            self.os,
            self.cupynumeric,
        )

    @property
    def runtime(self) -> RuntimeConfig:
        return RuntimeConfig(self.sanitizers, self.openmpi)

    @property
    def tests(self) -> TestsConfig:
        return TestsConfig()

    @property
    def docs(self) -> DocsConfig:
        return DocsConfig()

    @property
    def filename(self) -> str:
        return f"environment-{self.use}-{self.os}{self.cuda}{self.build}"


# --- Setup -------------------------------------------------------------------
def get_min_py() -> str:
    with (Path(__file__).parent.parent / "pyproject.toml").open(
        mode="rb"
    ) as f:
        py_ver = tomllib.load(f)["project"]["requires-python"]

    for char in (">", "=", "<"):
        py_ver = py_ver.replace(char, "")

    return py_ver


MIN_PYTHON_VERSION = get_min_py()
OS_NAMES: tuple[OSType, ...] = ("linux", "osx")


ENV_TEMPLATE = """\
name: legate-{use}
channels:
{channels}
dependencies:

  - python>={min_python_version}

{conda_sections}{pip}
"""

SECTION_TEMPLATE = """\
# {header}
{reqs}

"""

PIP_TEMPLATE = """\
  - pip
  - pip:
{pip_sections}
"""


# --- Code --------------------------------------------------------------------


class BooleanFlag(Action):
    def __init__(  # noqa: PLR0913
        self,
        option_strings,
        dest,
        default,
        required=False,  # noqa: FBT002
        help="",  # noqa: A002
        metavar=None,
    ):
        assert all(not opt.startswith("--no") for opt in option_strings)

        def flatten(in_list):
            return [item for sublist in in_list for item in sublist]

        option_strings = flatten(
            [
                (
                    [opt, "--no-" + opt[2:], "--no" + opt[2:]]
                    if opt.startswith("--")
                    else [opt]
                )
                for opt in option_strings
            ]
        )
        super().__init__(
            option_strings,
            dest,
            nargs=0,
            const=None,
            default=default,
            type=bool,
            choices=None,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string):  # noqa: ARG002
        setattr(namespace, self.dest, not option_string.startswith("--no"))


if __name__ == "__main__":
    import sys

    parser = ArgumentParser()
    parser.add_argument(
        "--ctk", dest="ctk_version", help="CTK version to generate for"
    )
    parser.add_argument(
        "--os",
        choices=OS_NAMES,
        default=("osx" if sys.platform == "darwin" else "linux"),
        help="OS to generate for",
    )
    parser.add_argument(
        "--compilers",
        action=BooleanFlag,
        dest="compilers",
        default=False,
        help="Whether to include conda compilers or not",
    )
    parser.add_argument(
        "--sanitizers",
        action=BooleanFlag,
        dest="sanitizers",
        default=False,
        help="Whether to include libsanitizers or not",
    )
    parser.add_argument(
        "--openmpi",
        action=BooleanFlag,
        dest="openmpi",
        default=False,
        help="Whether to include openmpi or not",
    )
    parser.add_argument(
        "--ucx",
        action=BooleanFlag,
        dest="ucx",
        default=False,
        help="Whether to include UCX or not",
    )
    parser.add_argument(
        "--cupynumeric",
        action=BooleanFlag,
        dest="cupynumeric",
        default=False,
        help="Whether to include cupynumeric dependencies",
    )

    parser.add_argument(
        "--sections",
        nargs="*",
        help="""List of sections exclusively selected for inclusion in the
        generated environment file.""",
    )

    args = parser.parse_args(sys.argv[1:])

    selected_sections = None

    if args.sections is not None:
        selected_sections = set(args.sections)

    def section_selected(section):
        if not selected_sections:
            return True

        return bool(selected_sections and str(section) in selected_sections)

    config = EnvConfig(
        "test",
        args.os,
        args.ctk_version,
        args.compilers,
        args.openmpi,
        args.ucx,
        args.sanitizers,
        args.cupynumeric,
    )

    conda_sections = indent(
        "".join(
            s.format("conda")
            for s in config.sections
            if s.conda and section_selected(s)
        ),
        "  ",
    )

    pip_sections = indent(
        "".join(
            s.format("pip")
            for s in config.sections
            if s.pip and section_selected(s)
        ),
        "    ",
    )

    filename = config.filename
    if args.sections:
        filename = config.filename + "-partial"

    print(f"--- generating: {filename}.yaml")  # noqa: T201
    out = ENV_TEMPLATE.format(
        use=config.use,
        channels=config.channels,
        min_python_version=MIN_PYTHON_VERSION,
        conda_sections=conda_sections,
        pip=(
            PIP_TEMPLATE.format(pip_sections=pip_sections)
            if pip_sections
            else ""
        ),
    )
    Path(f"{filename}.yaml").write_text(out)
