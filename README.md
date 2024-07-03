<!--
SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

**Composability**

Software must be compositional and not merely interoperable. Libraries
developed in the Legate ecosystem must be able to exchange partitioned
and distributed data without requiring "shuffles" or unnecessary blocking
synchronization. Computations from different libraries should be able to
use arbitrary data and still be reordered across abstraction boundaries
to hide communication and synchronization latencies (where the original
sequential semantics of the program allow). This is essential to achieve
optimal performance on large-scale machines.

## Documentation

Please check the [Legate Core documentation](https://docs.nvidia.com/legate) for
installation instructions, API reference, and further resources on using Legate.

## Questions

If you have questions, please contact us at legate(at)nvidia.com.
