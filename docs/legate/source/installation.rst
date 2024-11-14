.. _installation:

Installation
============

.. _how-do-i-install-legate:

How Do I Install Legate
-----------------------

Legate is available from `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_
on the `legate channel <https://anaconda.org/legate/legate>`_.
Please make sure you have at least conda version 24.1 installed, then create
a new environment containing Legate:

.. code-block:: sh

    conda create -n myenv -c conda-forge -c legate legate

or install it into an existing environment:

.. code-block:: sh

    conda install -c conda-forge -c legate legate

Packages with GPU support are available, and will be chosen automatically by
``conda install`` on systems with GPUs.

In an environment without GPUs available, ``conda install`` will by default
choose a CPU-only package. To install a version with GPU support in such an
environment, use environment variable ``CONDA_OVERRIDE_CUDA``:

.. code-block:: sh

    CONDA_OVERRIDE_CUDA="12.2" \
      conda install -c conda-forge -c legate legate

.. _support_matrix:

Support Matrix
--------------

The following table lists Legate's minimum supported versions of major dependencies.

"Full support" means that the corresponding versions (and all later ones) are
being tested with some regularity, and are expected to work. Please report any
incompatibility you find against a fully-supported version by opening a bug.

"Best-effort support" means that the corresponding versions are not actively
tested, but Legate should be compatible with them. We will not actively work to
fix any incompatibilities discovered under these versions, but we accept
contributions that fix such incompatibilities.

.. list-table:: Support Matrix
   :header-rows: 1

   * - Dependency
     - Full support (min version)
     - Best-effort support (min version)
   * - CPU architecture
     - x86-64 (Haswell), aarch64
     - older x86-64
   * - OS
     - RHEL 8, Ubuntu 20.04
     - other Linux
   * - GPU architecture
     - Volta
     - Pascal
   * - CUDA toolkit
     - 12.2
     - 11.0
   * - Python
     - 3.10
     -
   * - NumPy
     - 1.22
     -

Legate is tested and guaranteed to be compatible with Volta and later GPU
architectures. You can use Legate with Pascal GPUs as well, but there could
be issues due to lack of independent thread scheduling. Please report any such
issues by opening a bug.

.. _installation_of_mpi_wrapper:

Installation of the Legate MPI wrapper
--------------------------------------

If you encounter runtime failures such as

.. code-block:: sh

   failed to load MPI wrapper: 'some/path/to/liblegate_mpi_wrapper.so' ...

Or if you want to use Legate in combination with a different MPI library than the one it
was compiled against (see the dependencies on the Legate package), e.g. you are on an HPC
cluster and want to use the vendor's MPI library, then you will need to compile and
install the Legate MPI wrapper locally on your machine. See :ref:`FAQ<mpi_wrapper_faq>`
for more information on why this is needed.

Assuming Legate is installed to a directory called ``INSTALL_PREFIX``, to build and
install the wrappers simply run the following:

.. code-block:: sh

   $ INSTALL_PREFIX/share/legate/mpi_wrapper/install.bash

This command will build and install the MPI wrappers to the default installation
prefix. In order to build and install the wrappers you will need to have:

- CMake (at least version 3.0).
- A C++ compiler.
- A local installation of MPI.
- Write access to the installation prefix.

There are several influential environment variables that users may set in order to control
the build and installation process:

- ``CMAKE``: name or path to the ``cmake`` executable.
- ``CMAKE_INSTALL_PREFIX``, ``PREFIX``, or ``DESTDIR``: path to which the MPI wrappers
  should be installed. If one or more of these variables is set and not empty, they are
  preferred in the order listed. That is, ``CMAKE_INSTALL_PREFIX`` will be preferred over
  ``PREFIX``, which is preferred over ``DESTDIR``.
- ``CMAKE_ARGS`` or ``CMAKE_CONFIGURE_ARGS``: if set, arguments to be passed to the
  initial CMake configure command. If both are set, ``CMAKE_CONFIGURE_ARGS`` is preferred
  over ``CMAKE_ARGS``.
- ``CMAKE_BUILD_ARGS``: if set, arguments to be passed to the CMake build command.
- ``CMAKE_INSTALL_ARGS``: if set, arguments to be passed to the CMake install command.


Installation of the Legate IPython Kernel
-----------------------------------------

Please install Legate, then run the following command to install a default
Jupyter kernel:

.. code-block:: sh

    legate-jupyter

If installation is successful, you will see some output like the following:

.. code-block::

    Jupyter kernel spec Legate_SM_GPU (Legate_SM_GPU) has been installed

``Legate_SM_GPU`` is the default kernel name.

Licenses
--------

This project will download and install additional third-party open source
software projects at install time. Review the license terms of these open
source projects before use.

For license information regarding projects bundled directly, see
:ref:`thirdparty`.
