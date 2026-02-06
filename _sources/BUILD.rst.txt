..
  SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0


.. _legate_source_build:

===========================
Building Legate from Source
===========================

Basic build
===========

Influential environment variables
---------------------------------

Legate uses two environment variables ``${LEGATE_DIR}`` and ``${LEGATE_ARCH}`` to "orient"
itself on your system during the build phase.

.. note::

   The definition and use of these variables is only applicable during the local build
   phase. **Once Legate is installed, these variables are ignored.**

   They are used to determine the active build if you wish to have multiple concurrent
   configurations/builds of Legate.

   **If you only ever have a single configuration of Legate, you can ignore these variables.**


- ``${LEGATE_DIR}``: This variable should point to the root directory of the Legate source
  tree, i.e. the directory containing e.g. ``configure``, ``pyproject.toml``, and
  ``.clang-format``.
- ``${LEGATE_ARCH}``: This variable should be the (un-prefixed) name of the current build
  directory inside ``${LEGATE_DIR}``, such that ``${LEGATE_DIR}/${LEGATE_ARCH}`` points to
  the current "active" build directory.

  The actual value of ``${LEGATE_ARCH}`` is meaningless. It can be
  e.g. ``foo-bar-baz``. The only important thing is that it does not conflict with the
  name of another directory under ``${LEGATE_DIR}``.

  If you are unsure what to set for this variable, ignore it and run ``configure`` first
  -- it will choose an appropriate value for you and instruct you on how to set it.


If you have multiple configurations of Legate built, you can set these variables (usually
just ``${LEGATE_ARCH}``) to quickly switch between builds:

.. code-block:: sh

   # Run the tests using the "foo-bar" build of the library.
   $ LEGATE_ARCH=foo-bar ./test.py
   # Now run the tests using the "baz-bop" build of the library, all without needing to
   # recompile.
   $ LEGATE_ARCH=baz-bop ./test.py


.. note::

   Unless otherwise stated, all relative paths mentioned from here on out are relative to
   ``${LEGATE_DIR}``.


Building from source
--------------------

To build and install the basic C++ runtime:

.. code-block:: sh

   $ ./configure
   $ make install


Build and install C++ runtime and Python bindings:

.. code-block:: sh

   $ ./configure --with-python
   $ pip install .


Build and install basic C++ runtime with CUDA and HDF5 support, while disabling ZLIB, and
explicitly specifying a pre-built UCX directory. Specifying the UCX directory implies
enabling UCX support. Additionally, we install the library to a custom prefix:

.. code-block:: sh

   $ ./configure \
       --with-cuda \
       --with-hdf5 \
       --with-zlib=0 \
       --with-ucx-dir='/path/to/ucx'
   $ make install PREFIX=/path/to/prefix

A full list of options available during ``configure`` can be found by running:

.. code-block:: sh

   $ ./configure --help


For a list of example configurations, see the configure scripts under
``config/examples``. These contain configuration scripts for a wide variety of
machines. For example, to configure a debug build on a `DGX SuperPOD
<https://www.nvidia.com/en-us/data-center/dgx-superpod/>`_ you may use
``config/examples/arch-dgx-superpod-debug.py``.

For multi-node execution, Legate can use `UCX <https://openucx.org>`_ (use ``--with-ucx``)
or `GASNet <https://gasnet.lbl.gov/>`_ (use ``--with-gasnet``) see the discussion on
:ref:`dependencies <dependency_listing>` for more details.

Compiling with networking support requires MPI.

.. note::

   If you would like to build Legate with testing enabled, please refer to
   :ref:`ch_testing` for further information. In particular, the C++ tests require
   passing the additional ``--with-tests`` (and ``--with-benchmarks``) flags to
   ``configure`` to enable building the tests and benchmarks respectively.


.. _build_python_bindings:

Building Python Bindings
------------------------

When building the Python bindings for local development, it is strongly recommended to
build and install them into a Python virtual environment rather than the default system
prefix:


.. code-block:: sh

   # It does not matter if you run configure "inside" or "outside" the venv
   $ ./configure --with-python
   ...
   # Same with make
   $ make
   ...
   $ python3 -m pip install -U virtualenv
   $ python3 -m virtualenv ./my_venv_dir
   $ . ./my_venv_dir/bin/activate
   ...
   # Now that we are in our venv, we can build the python bindings. Legate will
   # detect the virtual environment and automatically install itself inside it
   # instead of the system prefix.
   $ pip install .


The reasons for doing so are as follows:

#. **Isolation and Cleanliness**

   Installing packages into a virtual environment avoids polluting the system
   prefix. Legate will install its dependencies alongside itself, so this helps maintain a
   clean system-wide environment, reducing the risk of conflicts.

#. **Accurate Dependency Resolution**

   Legate uses CMake to locate dependencies. If older installations or packages exist in
   the system prefix, they may be inadvertently reused, resulting in silent and unintended
   dependencies on stale or incompatible components. For example, Legate installs Legion,
   and subsequent reconfigurations may inadvertently pick up the installed Legion instead
   of building it from source, or using the given source directory
   (``--with-legion-src-dir``).

#. **Ease of Environment Reset**

   Deleting a virtual environment directory provides a quick and effective way to return
   to a clean development state. This is far simpler and safer than attempting to manually
   remove packages or files from the system prefix.

#. **Safe and Complete Uninstallation**

   Although ``pip uninstall legate`` will remove all the files it originally installed,
   the Python wheel specification does not define how to clean up empty directories. As a
   result, some artifacts (such as empty directory trees) may be left behind.


.. warning::

   In particular, the user SHOULD NOT install to a Conda prefix. The install will work,
   and subsequent reconfigurations have special handling if it detects that dependencies
   are being found in Conda prefixes, but it is extremely easy to accidentally circumvent
   this and subtly break your installation.

   This is not unique to Legate, any package requiring installation will run into this
   issue.

Dependencies
============

For many of its dependencies, ``configure`` will download and install them transparently
as part of the build. However for some (e.g. CUDA) this is not possible. In this case, the
user must use some other package manager or module system to load the necessary
dependencies.

The primary method of retrieving dependencies for Legate and downstream libraries is
through `conda <https://docs.conda.io/en/latest/>`_. You will need an installation of conda
to follow the instructions below. We suggest using the
`miniforge <https://github.com/conda-forge/miniforge>`_ distribution of conda.

Please use the ``scripts/generate-conda-envs.py`` script to create a conda environment
file listing all the packages that are required to build, run and test Legate (and
optionally some downstream libraries, e.g. cuPyNumeric). For example:

.. code-block:: sh

   $ ./scripts/generate-conda-envs.py --ctk 13.0.0 --ucx
   --- generating: environment-test-linux-cuda-13.0.0-ucx.yaml


Run this script with ``--help`` to see all available configuration options for the
generated environment file. See the :ref:`dependencies <dependency_listing>` section for more
details.

Once you have this environment file, you can install the required packages by creating a
new conda environment:

.. code-block:: sh

   $ conda env create -n legate -f /path/to/env/file.yaml


or by updating an existing environment:

.. code-block:: sh

   $ conda env update -f /path/to/env/file.yaml


You will want to "activate" this environment every time before (re-)building Legate, to
make sure it is always installed in the same directory (consider doing this in your shell
startup script):

.. code-block:: sh

   $ conda activate legate


Advanced build topics
=====================

.. _dependency_listing:

Dependency listing
------------------

In this section we comment further on our major dependencies. Please consult an
environment file created by ``scripts/generate-conda-envs.py`` for a full listing of
dependencies, e.g. building and testing tools, and for exact version requirements.

Operating system
----------------

Legate has been tested on Linux and macOS, although only a few flavors of Linux such as
Ubuntu have been thoroughly tested. Windows is currently only supported through WSL.

Python
------

In terms of Python compatibility, Legate *roughly* follows the timeline outlined in `NEP
29 <https://numpy.org/neps/nep-0029-deprecation_policy.html>`_.

C++ compiler
------------

We suggest that you avoid using the compiler packages available on conda-forge.  These
compilers are configured with the specific goal of building redistributable conda packages
(e.g. they explicitly avoid linking to system directories), which tends to cause issues
for development builds. Instead prefer the compilers available from your distribution's
package manager (e.g. apt/yum) or your HPC vendor.

If you want to pull the compilers from conda, use an environment file created by
``scripts/generate-conda-envs.py`` using the ``--compilers`` flag. An appropriate compiler
for the target OS will be chosen automatically.

CUDA (optional)
---------------

Only necessary if you wish to run with NVIDIA GPUs.

If CUDA is not installed under a standard system location, you will need to inform
``configure`` of its location using ``--with-cuda-dir`` (note, you don't need to pass
``--with-cuda`` when passing ``--with-cuda-dir``, desire for CUDA support is implied when
specifying the root directory).

If you intend to pull any CUDA libraries from conda (see below), conda will need to
install an environment-local copy of the CUDA toolkit, even if you have it installed
system-wide. To avoid versioning conflicts it is safest to match the version of CUDA
installed system-wide, by specifying it to ``scripts/generate-conda-envs.py`` through the
``--ctk`` flag.

CUDA libraries (optional)
-------------------------

Only necessary if you wish to run with NVIDIA GPUs.

The following additional CUDA libraries are required, for use by legate or downstream
libraries. Unless noted otherwise, these are included in the conda environment file.

- ``nccl``
- ``nvml``
- ``nvtx``
- ``CCCL`` (pulled from github)

If you wish to provide alternative installations for these, then you can remove them from
the environment file (or invoke ``scripts/generate-conda-envs.py`` with ``--ctk none``,
which will skip them all), and pass the corresponding ``--with-<dep>`` flag to
``configure`` (or let the build process attempt to locate them automatically).


Numactl (optional)
------------------

Required to support CPU and memory binding in the Legate launcher.

Not available on conda; typically available through the system-level package manager.

MPI (optional)
--------------

Only necessary if you wish to run on multiple nodes.

We suggest that you avoid using the generic build of OpenMPI available on
conda-forge. Instead prefer an MPI installation provided by your HPC vendor, or from
system-wide distribution channels like apt/yum and `MOFED
<https://network.nvidia.com/products/infiniband-drivers/linux/mlnx_ofed/>`_, since these
will likely be more compatible with (and tuned for) your particular system.

If you want to use the OpenMPI distributed on conda-forge, use an environment file created
by ``scripts/generate-conda-envs.py`` using the ``--openmpi`` flag.

Legate requires a build of MPI that supports ``MPI_THREAD_MULTIPLE``.

RDMA/networking libraries (e.g. Infiniband, RoCE, Slingshot) (optional)
-----------------------------------------------------------------------

Only necessary if you wish to run on multiple nodes, using the corresponding networking
hardware.

Not available on conda; typically available through MOFED or the system-level package
manager.

Depending on your hardware, you may need to use a particular Realm networking backend,
e.g. as of October 2023 HPE Slingshot is only compatible with GASNet.

GASNet (optional)
-----------------

Only necessary if you wish to run on multiple nodes, using the GASNet1 or GASNetEx Realm
networking backend.

This library will be automatically downloaded and built during Legate installation. If you
wish to provide an alternative installation, pass ``--with-gasnet`` to ``configure``.

When using GASNet, you also need to specify the interconnect network of the target machine
using the ``--gasnet-conduit`` flag.

UCX (optional)
--------------

Only necessary if you wish to run on multiple nodes, using the UCX Realm networking
backend.

You can use the version of UCX available on conda-forge by using an environment file
created by ``scripts/generate-conda-envs.py`` using the ``--ucx`` flag. Note that this
build of UCX might not include support for the particular networking hardware on your
machine (or may not be optimally tuned for such). In that case you may want to use an
environment file generated with ``--no-ucx`` (default), get UCX from another source
(e.g. MOFED, the system-level package manager, or compiled manually from `source
<https://github.com/openucx/ucx>`_), and pass the location of your UCX installation to
``configure`` (if necessary) using ``--with-ucx-dir``.

Legate requires a build of UCX configured with ``--enable-mt``.

Alternative sources for dependencies
------------------------------------

If you do not wish to use conda for some (or all) of the dependencies, you can remove the
corresponding entries from the environment file before passing it to conda.

Note that this is likely to result in conflicts between conda-provided and system-provided
libraries.

Conda distributes its own version of certain common libraries (in particular the C++
standard library), which are also typically available system-wide. Any system package you
include will typically link to the system version, while conda packages link to the conda
version. Often these two different versions, although incompatible, carry the same version
number (``SONAME``), and are therefore indistinguishable to the dynamic linker. Then, the
first component to specify a link location for this library will cause it to be loaded
from there, and any subsequent link requests for the same library, even if suggesting a
different link location, will get served using the previously linked version.

This can cause link failures at runtime, e.g. when a system-level library happens to be
the first to load GLIBC, causing any conda library that comes after to trip GLIBC's
internal version checks, since the conda library expects to find symbols with more recent
version numbers than what is available on the system-wide GLIBC:

.. code-block:: sh

   ...
   /lib/x86_64-linux-gnu/libstdc++.so.6: version GLIBCXX_3.4.30 not found (required by /opt/conda/envs/legate/lib/libfoo.so)


You can usually work around this issue by putting the conda library directory first in the
dynamic library resolution path:

.. code-block:: sh

   # On Linux
   $ export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH}"
   # On macOS
   $ export DYLD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${DYLD_LIBRARY_PATH}"


This way you can make sure that the (typically more recent) conda version of any common
library will be preferred over the system-wide one, no matter which component requests it
first.

Building Python Bindings Without Configure
------------------------------------------

Technically, when installing Python bindings, ``configure`` is optional. It is possible
to configure, build, and install Legate with python bindings using just:

.. code-block:: sh

   $ pip install .


While this workflow is supported (in the sense that it is functional), very little -- if
any -- effort is made to make it ergonomic. **The user is strongly encouraged to run
configure first**.

In particular, it requires the following from the user:

#. Defining all CMake options manually through ``CMAKE_ARGS`` environment variable.
#. Defining all scikit-build options, including any that might be implicitly set via
   ``configure``, manually via appropriate environment variables.
#. Ensuring that no prior installation of Legate, Legion, or any of its dependencies exist
   in the environment which might otherwise influence the CMake configuration.

   For example, due to how CMake picks up dependencies, a prior (stale) installation of
   Legion to a shared ``conda`` environment may be prioritized over downloading it from
   scratch. ``configure`` automatically detects this (and sets the appropriate CMake
   variables to guard against it) but a bare ``pip install`` will not do so.
#. Other potential quality-of-life improvements made by ``configure``.
