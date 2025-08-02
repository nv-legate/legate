..
  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. _networking_wheels:

Networking with Legate Wheels
=============================

MPI Support
-----------

The Legate wheels are built with UCX-based networking support, but MPI itself is
not provided as a PyPI wheel.  System-installed MPI is required to bootstrap the
UCX networking. Currently, Legate supports OpenMPI 4.x or above and MPICH 3.x or
above. You can install OpenMPI or MPICH using your system's package manager. For
example, on Ubuntu:

.. code-block:: bash

   $ sudo apt-get install libopenmpi-dev

   # =========== OR =========== #

   $ sudo apt-get install libmpich-dev

Legate attempts to discover which MPI implementation is available on your system
and use one of its bundled *MPI wrappers*.  There are two wrappers, one for
OpenMPI and one for MPICH.  Most of the time this process works automatically
and does not require user intervention.  However, in rare cases, the user may
need to manually specify the MPI wrapper to use.  For example, the following
error indicates that wrapper may have to be chosen explicitly

.. code-block:: bash

   LEGATE ERROR: #0 std::runtime_error: dlopen("libmpi.so") failed: libmpi.so: cannot open shared object file: No such file or directory, please make sure MPI is installed and libmpi.so is in your LD_LIBRARY_PATH.

The wheel comes bundled with two wrappers:

* ``liblegate_mpi_wrapper_mpich.so`` for MPICH, and
* ``liblegate_mpi_wrapper_ompi.so`` for OpenMPI.

Selecting the MPI wrapper can be done by setting the ``LEGATE_MPI_WRAPPER``
environment variable to the path of the desired MPI wrapper.  For example, if
you have installed MPICH, you can set the ``LEGATE_MPI_WRAPPER`` environment
variable to the MPICH wrapper:

.. code-block:: sh

   $ export LEGATE_MPI_WRAPPER=liblegate_mpi_wrapper_mpich.so

Note that only the name of the wrapper is needed because Legate is compiled with
appropriate rpaths that allow it to find the wrapper in the location where it
was installed by the wheel.

The NERSC's Perlmutter system is one example of where setting the wrapper may be
necessary.  Perlmutter has 3 different MPICH modules, ``cray-mpich``,
``cray-mpich-abi``, and ``mpich`` (this module is created by NERSC).  Legate
will work with ``cray-mpich-abi`` and ``mpich`` since it requires the stable ABI
version of MPICH provided by these modules.  However, the ``cray-mpich-abi``
module does not provide ``libmpi.so``, which is required by the Legate MPI
detection code.  So, when using this module, one must set the wrapper
explicitly.  The ``mpich`` module, on the other hand, provides ``libmpi.so``, so
the Legate MPI detection code will work without any user intervention.

If neither of the pre-built wrappers work for your system, check
:ref:`installation_of_mpi_wrapper` for instructions on building the MPI wrapper
from source.

UCX Support
-----------

The Legate wheels are built with UCX-based networking support against UCX
provided by the `libucx <https://pypi.org/project/libucx-cu12/>`_ wheel.  This
UCX build should be functional in most cases.  However, it does not support all
networking protocols that UCX can potentially support.  If you rely on support
for specific networking hardware, you may prefer to use system-installed UCX
that is built with that support.  To prefer dynamically loading UCX system
libraries set the following variable:

.. code-block:: bash

   $ export RAPIDS_LIBUCX_PREFER_SYSTEM_LIBRARY=1

Potential Issues
----------------

Note that the framework of automatic detection and automatic support for
networking depends on multiple dynamic loads of the MPI and the UCX libraries.  Sometimes, errors may occur when an incorrect or conflicting libraries are
loaded.  If that happens and a crash occurs, Legate should print out a
backtrace showing which versions of the different libraries were loaded.  If one
of the libraries is loaded from a different location than expected, you can set
the ``LD_LIBRARY_PATH`` environment variable to point to the correct location of
the libraries or use ``LD_PRELOAD`` to force the loading of the correct library:

.. code-block:: bash

   $ export LD_PRELOAD=/path/to/correct/libmpi.so

   # =========== OR =========== #

   $ export LD_LIBRARY_PATH=/path/to/correct/lib/:${LD_LIBRARY_PATH}

This approach may be used for a fine-grained control over which libraries are
loaded across the networking stack.  However, it is not recommended to set
``LD_PRELOAD`` or ``LD_LIBRARY_PATH`` globally, as this may cause conflicts with
other libraries and applications.  Instead, it is recommended to set these
variables only for the specific Legate application you are running.

.. tip::

   Legate wheels depend on three main components:

   * UCX,
   * MPI,
   * and the CUDA toolkit.

   As discussed previously, Legate installs a basic configuration of UCX as a
   dependency.  Configurations that are known to work with Legate wheels include
   Ubuntu 20.x and above with Ubuntu-installed OpenMPI, and with CUDA toolkit
   12.2 or higher.  The Legate wheel was also tested on Perlmutter with the
   ``cray-mpich-abi`` module and the ``mpich`` modules.

   However, because wheels are not as self contained as the Conda ecosystem, it is
   possible that other configurations may not work as well.  If you encounter
   problems with the Legate wheels, visit the :ref:`overview-contact` page for
   more information on how to get help.`
