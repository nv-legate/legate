<!--
SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: LicenseRef-NvidiaProprietary

NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
property and proprietary rights in and to this material, related
documentation and any modifications thereto. Any use, reproduction,
disclosure or distribution of this material and related documentation
without an express license agreement from NVIDIA CORPORATION or
its affiliates is strictly prohibited.
-->

# Legate

The Legate project makes it easier for programmers to leverage the
power of large clusters of CPUs and GPUs. Using Legate, programs can be
developed and tested on moderately sized data sets on local machines and
then immediately scaled up to larger data sets deployed on many nodes in
the cloud or on a supercomputer, *without any code modifications*.

---

The Legate API is implemented on top of the [Legion](https://legion.stanford.edu/)
programming model and runtime system, which was originally designed for large
HPC applications that target supercomputers.

The Legate project is built from two foundational principles:

**Implicit parallelism**

For end users, the programming model must be identical to programming a
single sequential CPU on their laptop or desktop. Parallelism, data
distribution, and synchronization must be implicit. The cloud or a
supercomputer should appear as nothing more than a super-powerful CPU core.

**Composibility**

Software must be compositional and not merely interoperable. Libraries
developed in the Legate ecosystem must be able to exchange partitioned
and distributed data without requiring "shuffles" or unnecessary blocking
synchronization. Computations from different libraries should be able to
use arbitrary data and still be reordered across abstraction boundaries
to hide communication and synchronization latencies (where the original
sequential semantics of the program allow). This is essential to achieve
optimal performance on large-scale machines.

For more information about Legate's goals, architecture, and functioning,
see the [Legate overview](https://nv-legate.github.io/legate.core/overview).

## Installation

Legate Core is available [on conda](https://anaconda.org/legate/legate-core).
Create a new environment containing Legate Core:

```
conda create -n myenv -c nvidia -c conda-forge -c legate legate-core
```

or install it into an existing environment:

```
conda install -c nvidia -c conda-forge -c legate legate-core
```

Only linux-64 packages are available at the moment.

The default package contains GPU support, and is compatible with CUDA >= 12.0
(CUDA driver version >= r520), and Volta or later GPU architectures. There are
also CPU-only packages available, and will be automatically selected when
installing on a machine without GPUs. You can force installation of a CPU-only
package by requesting it as follows:

```
conda ... legate-core=*=*_cpu
```

See [BUILD.md](BUILD.md) for instructions on building Legate Core from source.

## Documentation

A complete list of available features and APIs can be found in the [Legate Core
documentation](https://nv-legate.github.io/legate.core).


## Next Steps

We recommend starting by experimenting with at least one Legate application
library to test out performance and see how Legate works. If you are interested
in building your own Legate application library, then the
[Legate Hello World application library](https://github.com/nv-legate/legate.core/tree/HEAD/examples/hello)
provides a getting-started example of developing your own library
on top of Legion, using the Legate Core library.

We also encourage development and contributions to existing Legate libraries, as
well as the development of new Legate libraries. Pull requests are welcomed.

If you have questions, please contact us at legate(at)nvidia.com.
