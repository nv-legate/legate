<!--
SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

[![Build conda Nightly release package](https://github.com/nv-legate/legate.internal/actions/workflows/ci-gh-nightly-release.yml/badge.svg?event=schedule)](https://github.com/nv-legate/legate.internal/actions/workflows/ci-gh-nightly-release.yml)

# Legate

The Legate project makes it easier for programmers to leverage the
power of large clusters of CPUs and GPUs. Using Legate, programs can be
developed and tested on moderately sized data sets on local machines and
then immediately scaled up to larger data sets deployed on many nodes in
the cloud or on a supercomputer, *without any code modifications*.

For more information about Legate's goals, architecture, and functioning,
see the [Legate overview](https://docs.nvidia.com/legate/latest/overview.html).

## Installation

Pre-built Legate packages are available from
[conda](https://docs.conda.io/projects/conda/en/latest/index.html) on the
[legate channel](https://anaconda.org/legate/legate) and from
[PyPI](https://pypi.org/project/legate/). See
https://docs.nvidia.com/legate/latest/installation.html for details about
different install configurations.

📌 **Note**

Packages are only offered for Linux (x86_64 and aarch64) supporting Python
versions 3.10 to 3.12.

## Documentation

A complete list of available features and APIs can be found in the [Legate
documentation](https://docs.nvidia.com/legate/latest/).

## Contact

For technical questions about Legate and Legate-based tools, please visit the
[community discussion forum](https://github.com/nv-legate/discussion).

If you have other questions, please contact us at legate(at)nvidia.com.
