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

from setuptools import find_packages
from skbuild import setup

import versioneer

setup(
    name="legate-core",
    version=versioneer.get_version(),
    description="legate.core - The Foundation for All Legate Libraries",
    url="https://github.com/nv-legate/legate.core",
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
    extras_require={
        "test": [
            "colorama",
            "coverage",
            "mock",
            "mypy>=0.961",
            "pynvml",
            "pytest-cov",
            "pytest",
        ]
    },
    packages=find_packages(
        where=".",
        include=[
            "legate",
            "legate.*",
            "legate.core",
            "legate.core.*",
            "legate.timing",
            "legate.timing.*",
        ],
    ),
    include_package_data=True,
    entry_points={
        "console_scripts": [
            "legate = legate.driver:main",
            "legate-jupyter = legate.jupyter:main",
            "lgpatch = legate.lgpatch:main",
        ],
    },
    scripts=["bind.sh"],
    cmdclass=versioneer.get_cmdclass(),
    install_requires=[
        "cffi",
        "numpy>=1.22",
        "typing_extensions",
    ],
    zip_safe=False,
)
