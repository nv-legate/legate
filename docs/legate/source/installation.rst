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

You will probably also want to install some downstream libraries built on top of
Legate, e.g. `cuPyNumeric <https://docs.nvidia.com/cupynumeric>`_:

.. code-block:: sh

    conda install -c conda-forge -c legate cupynumeric

.. _support_matrix:

Support Matrix
--------------

The following table lists Legate's minimum supported versions of major
dependencies.

"Full support" means that the corresponding versions (and all later ones) are
being tested regularly, released as downloadable packages, and are expected to work.
Please report any incompatibility you find against a fully-supported version
by opening a bug.

"Best-effort support" means that the corresponding versions may not be actively packaged
and shipped, but that Legate should be compatible with them. In particular, Legate should
be able to be compiled from source for these configurations.

We may not actively work to fix any incompatibilities discovered under these versions, but
we accept contributions that fix such incompatibilities.

.. list-table:: Support Matrix
   :header-rows: 1

   * - Dependency
     - Full support (min version)
     - Best-effort support (min version)
   * - CPU architecture
     - x86-64 (Haswell), aarch64
     - older x86-64, Apple Silicon
   * - OS
     - RHEL 8, Ubuntu 20.04
     - other Linux, macOS
   * - Compilers
     - GCC 7, clang 5
     - Any compiler supporting C++17
   * - GPU architecture
     - Volta
     - Pascal
   * - CUDA toolkit
     - 12.2
     - 11.8
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

Or if you want to use Legate in combination with a different MPI library than
the one it was compiled against (see the dependencies on the Legate package),
e.g. you are on an HPC cluster and want to use the vendor's MPI library, then
you will need to compile and install the Legate MPI wrapper locally on your
machine. See :ref:`FAQ<mpi_wrapper_faq>` for more information on why this is
needed.

Assume Legate is already installed in a conda environment named ``myenv``. We
need to activate this environment and install the wrapper packages that contain
the scripts necessary to build the wrappers (note the custom channel
``legate/label/gex`` in the install command):

.. code-block:: sh

    $ conda activate myenv
    $ conda install -c conda-forge -c legate/label/gex legate-mpi-wrapper

When the wrapper package is installed, the instructions for building the wrapper
are displayed:

.. code-block:: sh

    To finish configuring the Legate MPI wrapper, activate your environment and run /path-to-myenv/mpi-wrapper/build-mpi-wrapper.sh

To build the wrapper, first activate the ``myenv`` environment:

.. code-block:: sh

    $ conda activate myenv


    --------------------- CONDA/MPI_WRAPPER/ACTIVATE.SH -----------------------

    LEGATE_MPI_WRAPPER=

Note that when the environment is activated without the wrapper built, the
activation script for the wrapper package sets the ``LEGATE_MPI_WRAPPER``
environment variable to an empty value, since there is no wrapper shared library
to find yet.

After the environment is activated, we can build the MPI wrapper:

.. code-block:: sh

    $ /path-to-myenv/mpi-wrapper/build-mpi-wrapper.sh

In order to build and install the wrapper you will need to have:

- CMake (at least version 3.0).
- A C++ compiler.
- A local installation of MPI.
- Write access to the conda environment.

You can specify a compiler to the build script using the ``-c`` option.
Additionally, there are several environment variables that you can set in order
to control the build and installation process:

- ``CMAKE``: name or path to the ``cmake`` executable.
- ``CMAKE_ARGS`` or ``CMAKE_CONFIGURE_ARGS``: if set, arguments to be passed to
  the initial CMake configure command. If both are set, ``CMAKE_CONFIGURE_ARGS``
  is preferred over ``CMAKE_ARGS``.
- ``CMAKE_BUILD_ARGS``: if set, arguments to be passed to the CMake build
  command.
- ``CMAKE_INSTALL_ARGS``: if set, arguments to be passed to the CMake install
  command.

Once the wrapper is built, reactivate the environment to set the necessary
environment variables:

.. code-block:: sh

    $ conda deactivate


    --------------------- CONDA/MPI_WRAPPER/DEACTIVATE.SH -----------------------

    +++ unset LEGATE_MPI_WRAPPER
    +++ set +x
    $ conda activate myenv


    --------------------- CONDA/MPI_WRAPPER/ACTIVATE.SH -----------------------

    LEGATE_MPI_WRAPPER=/path-to-myenv/mpi-wrapper/lib64/liblgcore_mpi_wrapper.so

Note that the activation script now successfully located the MPI wrapper shared
library.

It might also be useful to remove the MPI conda package that Legate was compiled
against (typically ``openmpi``), to make sure that there is only one choice of
MPI to use:

```
conda uninstall --force openmpi
```

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
