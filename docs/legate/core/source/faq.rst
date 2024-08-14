Frequently Asked Questions
==========================

Does Legate only work on NVIDIA hardware?
    No, Legate will run on any processor supported by Legion (e.g. x86, ARM, and
    PowerPC CPUs), as well as any network supported by GASNet or UCX (e.g. Infiniband,
    Cray, Omnipath, and (ROC-)Ethernet based interconnects).

What languages does the Legate Core API have bindings for?
    Currently the Legate Core bindings are only available in Python. Watch
    this space for new language bindings soon or make a pull request to
    contribute your own. Legion has a C API which should make it easy to
    develop bindings in any language with a foreign function interface.

Do I have to build drop-in replacement libraries?
    No! While we've chosen to provide drop-in replacement libraries for
    popular Python libraries to illustrate the benefits of Legate, you
    are both welcomed and encouraged to develop your own libraries on top
    of Legate. We promise that they will compose well with other existing
    Legate libraries.

What other libraries are you planning to release for the Legate ecosystem?
    We're still working on that. If you have thoughts about what is important
    please let us know at *legate@nvidia.com* so that we can get a feel for
    where best to put our time.

Can I use Legate with other Legion libraries?
    Yes! If you're willing to extract the Legion primitives from the ``LegateStore``
    objects you should be able to pass them into other Legion libraries such as
    `FlexFlow <https://flexflow.ai/>`_.

Does Legate interoperate with X?
    Yes, probably, but we don't recommend it. Our motivation for building
    Legate is to provide the bare minimum subset of functionality that
    we believe is essential for building truly composable software that can still
    run at scale. No other systems out there met our needs. Providing
    interoperability with those other systems will destroy the very essence
    of what Legate is and significantly dilute its benefits. All that being
    said, Legion does provide some means of doing stop-the-world exchanges
    with other runtime system running concurrently in the same processes.
    If you are interested in pursuing this approach please open an issue
    on the `Legion github issue tracker <https://github.com/StanfordLegion/legion/issues>`_
    as it will be almost entirely orthogonal to how you use Legate.

What platform and library versions does Legate support?
    The following table lists Legate's minimum supported versions of major dependencies.

    "Full support" means that the corresponding versions (and all later ones) are
    being tested with some regularity, and are expected to work. Please report any
    incompatibility you find against a fully-supported version by opening a bug.

    "Best-effort support" means that the corresponding versions are not actively
    tested, but Legate should be compatible with them. We will not actively work to
    fix any incompatibilities discovered under these versions, but we accept
    contributions that fix such incompatibilities.

    ================ =============================== ====================================
    Dependency       Full support (min version)      Best-effort support (min version)
    ================ =============================== ====================================
    CPU architecture x86-64 (Haswell), aarch64       ppc64le, older x86-64, Apple Silicon
    OS               RHEL 8, Ubuntu 20.04, MacOS 12  other Linux
    C++ compiler     gcc 8, clang 7, nvc++ 19.1      any compiler with C++17 support
    GPU architecture Volta                           Pascal
    CUDA toolkit     12.0                            11.0
    Python           3.10
    NumPy            1.22
    ================ =============================== ====================================

    Legate is tested and guaranteed to be compatible with Volta and later GPU
    architectures. You can use Legate with Pascal GPUs as well, but there could
    be issues due to lack of independent thread scheduling. Please report any such
    issues on GitHub.

    Legate has been tested on Linux and MacOS, although only a few flavors of Linux
    such as Ubuntu have been thoroughly tested. There is currently no support for
    Windows.

.. _mpi_wrapper_faq:

What is the Legate MPI wrapper and why do I need it?
    MPI is considered "API stable", not "ABI stable"[#ABI]_. If a library is compiled on one
    machine, with a particular flavor and version of MPI (for example, OpenMPI v4.10),
    there is no guarantee that the same shared library will be able to link and run on
    another machine with another MPI implementation, or even a different version
    of the same MPI implementation.

    Normally this is not a problem for open-source libraries because users can simply
    compile the source on their local machine, with their local MPI. For closed-source
    libraries (such as Legate) which only ship binary artifacts, this presents a large
    problem.

    There are 2 choices to address this issue:

    #. Ship a pre-compiled version of the library for all supported MPI flavors and
       versions.
    #. "Shim" the MPI interface with a lightweight wrapper which can be compiled locally
       on the target machine.

    Solution 1 is easiest for the user, as they will -- courtesy of their package manager
    -- automatically get a compatible version of both MPI and Legate installed on their
    machine which are guaranteed to be compatible. However, this places an enormous
    packaging burden on us. We would have to ship a version of our library for each MPI
    version and regular variant (CPU, GPU, debug, etc.). If we have 3 library variants,
    and 5 compatible MPI versions, this means that we need to ship (and test!) 15
    different versions of our library. In practice, due to the longstanding ABI
    instability in MPI, the true number of MPI versions/flavors that would need this
    treatment would easily number in the 30-40 range.

    Solution 2 is a compromise. In this case, we can isolate the MPI compile surface to
    just the functions that Legate requires into a "shim" wrapper. This wrapper, once
    compiled on the end users' machine, can then be dynamically loaded by Legate at
    runtime to enable the interface.

.. rubric:: Footnotes

.. [#ABI] There are two classes of software interface stability: "API" and "ABI"
    stability. API stability means that a given piece of software will not make backwards
    incompatible source-level changes. This means that, if the software exports ``int
    foo(int)``, then users can expect a function ``foo()`` with a compatible signature to
    exist forever.

    Note that a "compatible" signature does not mean that modifications to ``foo()`` are
    forbidden!  For example, if the library is built using a language that supports
    default arguments, they are free to modify ``foo()`` to, for example, ``int foo(int,
    double = 1.2)``. This modification allows original callers of ``foo()``

    ABI stability, on the other hand, is a super-set of API stability, and mandates that
    not only will the library never change existing API between versions but that the
    binary interface will also remain unchanged. Note, the strictness of API compatibility
    is now absolute. While previously the library could get away with extending ``foo()``
    with, say, default parameters to keep a *compatible* API, the library is now bound to
    export ``int foo(int)`` exactly as written in perpetuity.
