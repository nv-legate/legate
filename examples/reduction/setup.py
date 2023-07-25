#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

import os
from pathlib import Path

from setuptools import find_packages
from skbuild import setup

import legate.install_info as lg_install_info

legate_dir = Path(lg_install_info.libpath).parent.as_posix()

cmake_flags = [
    f"-Dlegate_core_ROOT:STRING={legate_dir}",
]

env_cmake_args = os.environ.get("CMAKE_ARGS")
if env_cmake_args is not None:
    cmake_flags.append(env_cmake_args)
os.environ["CMAKE_ARGS"] = " ".join(cmake_flags)


setup(
    name="Legate Reduction Tutorial",
    version="0.1",
    description="Reduction examples for Legate",
    author="NVIDIA Corporation",
    license="Proprietary",
    classifiers=[
        "Intended Audience :: Developers",
        "Topic :: Database",
        "Topic :: Scientific/Engineering",
        "License :: Proprietary :: Nvidia Proprietary",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    packages=find_packages(
        where=".",
        include=["reduction", "reduction.*"],
    ),
    include_package_data=True,
    zip_safe=False,
    install_requires=["cunumeric"],
)
