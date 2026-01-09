..
  SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

Overview
========

The Legate project makes it easier for programmers to leverage the
power of large clusters of CPUs and GPUs. Using Legate, programs can be
developed and tested on moderately sized data sets on local machines and
then immediately scaled up to larger data sets deployed on many nodes in
the cloud or on a supercomputer, *without any code modifications*.

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

Why Legate
----------

Computational problems today continue to grow both in their complexity as well
as the scale of the data that they consume and generate. This is true both in
traditional HPC domains as well as enterprise data analytics cases. Consequently,
more and more users truly need the power of large clusters of both CPUs and
GPUs to address their computational problems. Not everyone has the time or
resources required to learn and deploy the advanced programming models and tools
needed to target this class of hardware today. Legate aims to bridge this gap
so that any programmer can run code on any scale machine without needing to be
an expert in parallel programming and distributed systems, thereby allowing
developers to bring the problem-solving power of large machines to bear on
more kinds of challenging problems than ever before.

The Legate Data Model
---------------------

Legate's data model is inspired by `Apache Arrow <https://arrow.apache.org/>`_.
Apache Arrow has significantly improved composability of software libraries by
making it possible for different libraries to share in-memory buffers of data
without unnecessary copying. However, it falls short when it comes to meeting
two of our primary requirements for Legate:

1. Arrow only provides an API for describing a physical representation
   of data as a single memory allocation. There is no interface for describing
   cases where data has been partitioned and then capturing the logical
   relationships of those partitioned subsets of data.
2. Arrow is mute on the subject of synchronization. Accelerators such as GPUs
   achieve significantly higher performance when computations are performed
   asynchronously with respect to other components of the system. When data is
   passed between libraries today, accelerators must be pessimistically
   synchronized to ensure that data dependencies are satisfied across abstraction
   boundaries. This might result in tolerable overheads for single GPU systems,
   but can result in catastrophically poor performance when hundreds of GPUs are involved.

Legate provides an API very similar to Arrow's interface with several
important distinctions that provide stronger guarantees about data coherence and
synchronization to aid library developers when building Legate libraries. These
guarantees are the crux of how libraries in the Legate ecosystem are able to
provide excellent composability.

The Legate API imports several important concepts from Arrow such that
users that are familiar with Arrow already will find it unsurprising. We use
the same type system representation as Arrow so libraries that have already
adopted it do not need to learn or adapt to a new type system. We also reuse
the concept of an `Array <https://arrow.apache.org/docs/cpp/api/array.html>`_
from Arrow. The ``LegateArray`` class supports many of the same methods as
the Arrow Array interface (we'll continue to add methods to improve
compatibility). The main difference is that instead of obtaining
`Buffer <https://arrow.apache.org/docs/cpp/api/memory.html#buffers>`_
objects from arrays to describe allocations of data that back the array, the
Legate API introduces a new primitive called a ``LegateStore`` which
provides a new interface for reasoning about partitioned and distributed
data in asynchronous execution environments.

Any implementation of a ``LegateStore`` must maintain the following guarantees
to clients of the Legate API (i.e. Legate library developers):

1. The coherence of data contained in a ``LegateStore`` must be implicitly
   managed by the implementation of the Legate API. This means that
   no matter where data is requested to perform a computation in a machine,
   the most recent modifications to that data in program order must be
   reflected. It should never be clients responsibility to maintain this
   coherence.
2. It should be possible to create arbitrary views onto ``LegateStore`` objects
   such that library developers can precisely describe the working sets of
   their computations. Modifications to views must be reflected onto all
   aliasing views data. This property must be maintained by the Legate
   API implementation such that it is never the concern of clients.
3. Dependence management between uses of the ``LegateStore`` objects and their
   views is the responsibility of Legate API regardless of what
   (asynchronous) computations are performed on ``LegateStore`` objects or their
   views. This dependence analysis must be both sound and precise. It is
   illegal to over-approximate dependencies. This dependence analysis must also
   be performed globally in scope. Any use of the ``LegateStore`` on any
   processor/node in the system must abide by the original sequential
   semantics of the program

Note that we do not specify exactly what the abstractions are that are needed
for implementing ``LegateStore`` objects. Our goal is not prescribe what these
abstractions are as they may be implementation dependent. Our only requirements
are that they have these properties to ensure that incentives are aligned in
such a way for Legate libraries to achieve a high degree of composability
at any scale of machine. Indeed, these requirements shift many of the burdens
that make implementing distributed and accelerated libraries hard off of the
library developers and onto the implementation of the Legate API. This
is by design as it allows the costs to be amortized across all libraries in
the ecosystem and ensures that Legate library developers are more productive.

How Does Legate Work
--------------------

Our implementation of the Legate API is built on top of the
`Legion <https://legion.stanford.edu/>`_ programming model and runtime system.
Legion was originally designed for large HPC applications that target
supercomputers and consequently applications written in the Legion programming
model tend to both perform and scale well on large clusters of both CPUs and
GPUs. Legion programs are also easy to port to new machines as they inherently
decouple the machine-independent specification of computations from decisions
about how that application is mapped to the target machine. Due to this
abstract nature, many programmers find writing Legion programs challenging.
By implementing the Legate API on top of Legion, we've made it easier
to use Legion such that developers can still get access to the benefits of
Legion without needing to learn all of the lowest-level interfaces.

The `Legion programming model <https://legion.stanford.edu/pdfs/sc2012.pdf>`_
greatly aids in implementing the Legate API. Data types from libraries,
such as arrays in cuPyNumeric are mapped down onto ``LegateStore`` objects
that wrap Legion data types such as logical regions or futures.
In the case of regions, Legate application libraries rely heavily on
Legion's `support for partitioning of logical regions into arbitrary subregion views <https://legion.stanford.edu/pdfs/oopsla2013.pdf>`_.
Each library has its own heuristics for computing such partitions that
take into consideration the computations that will access the data, the
ideal sizes of data to be consumed by different processor kinds, and
the available number of processors. Legion automatically manages the coherence
of subregion views regardless of the scale of the machine.

Computations in Legate application libraries are described by Legion tasks.
Tasks describe their data usage in terms of ``LegateStore`` objects, thereby
allowing Legion to infer where dependencies exist. Legion uses distributed
bounding volume hierarchies, similar to a high performance ray-tracer,
to soundly and precisely perform dependence analysis on logical regions
and insert the necessary synchronization between tasks to maintain the
original sequential semantics of a Legate program.

Each Legate application library also comes with its own custom Legion
mapper that uses heuristics to determine the best choice of mapping for
tasks (e.g. are they best run on a CPU or a GPU). All
Legate tasks are currently implemented in native C or CUDA in order to
achieve excellent performance on the target processor kind, but Legion
has bindings in other languages such as Python, Fortran, and Lua for
users that would prefer to use them. Importantly, by using Legion,
Legate is able to control the placement of data in order to leave it
in-place in fast memories like GPU framebuffers across tasks.

When running on large clusters, Legate leverages a novel technology provided
by Legion called "`control replication <https://research.nvidia.com/sites/default/files/pubs/2021-02_Scaling-Implicit-Parallelism//ppopp.pdf>`_"
to avoid the sequential bottleneck
of having one node farm out work to all the nodes in the cluster. With
control replication, Legate will actually replicate the Legate program and
run it across all the nodes of the machine at the same time. These copies
of the program all cooperate logically to appear to execute as one
program. When communication is necessary between
different computations, the Legion runtime's program analysis will automatically
detect it and insert the necessary data movement and synchronization
across nodes (or GPU framebuffers). This is the transformation that allows
sequential programs to run efficiently at scale across large clusters
as though they are running on a single processor.

.. _overview-contact:

Contact
-------

For technical questions about Legate and Legate-based tools, please visit
the `community discussion forum <https://github.com/nv-legate/discussion>`_.
