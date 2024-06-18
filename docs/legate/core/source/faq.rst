Frequently Asked Questions
==========================

Does Legate only work on NVIDIA hardware?
    No, Legate will run on any processor supported by Legion (e.g. x86, ARM, and
    PowerPC CPUs), and any network supported by GASNet or UCX (e.g. Infiniband,
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
    please let us know at *legate(at)nvidia.com* so that we can get a feel for
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