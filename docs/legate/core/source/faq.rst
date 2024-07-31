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
