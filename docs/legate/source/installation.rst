.. _installation:

============
Installation
============

.. _how-do-i-install-legate:

Installing Conda Packages
=========================

Legate is available from `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_
on the `legate channel <https://anaconda.org/legate/legate>`_.

.. note::
   conda version >= 24.1 required

.. code-block:: bash

   # with a new environment
   $ conda create -n myenv -c conda-forge -c legate legate

   # =========== OR =========== #

   # into an existing environment
   $ conda install -c conda-forge -c legate legate

You will probably also want to install some downstream libraries built on top of
Legate, e.g. `cuPyNumeric <https://docs.nvidia.com/cupynumeric>`_:

.. code-block:: bash

   $ conda install -c conda-forge -c legate cupynumeric

.. important::

  Packages are only offered for Linux (x86_64 and aarch64) supporting Python
  versions 3.11 to 3.13.

Nightly top-of-tree builds of Legate are available under the "experimental" label:

.. code-block:: bash

   $ conda install -c conda-forge -c legate/label/experimental legate

Refer to the `nightly docs <https://nv-legate.github.io/legate>`_ when using these.

.. important::

  These builds are only lightly validated. Use them at your own risk.

Conda and GPU / CPU Variants
----------------------------

`conda` automatically installs the right variant for the system:
* CPU variant if no NVIDIA GPU is detected
* GPU variant if an NVIDIA GPU is detected

To override this behavior and force install a version with GPU support, use the
following (with the desired CUDA version):

.. code-block:: sh

   $ CONDA_OVERRIDE_CUDA="12.2" \
       conda install -c conda-forge -c legate legate


Installing PyPI Packages
========================

Legate is also available as a `Python wheel <https://pypi.org/project/legate>`_.
To install the Legate wheel, use the the ``pip`` package manager:

.. code-block:: bash

   # into existing environment
   $ pip install legate

   # =========== OR =========== #

   # into new environment
   $ python -m venv myenv
   $ source myenv/bin/activate
   $ pip install legate

The Legate wheel comes with GPU support and UCX-based networking
support.  Similarly to the Conda packages, the Legate wheel will probably be
installed alongside downstream libraries, such as `cuPyNumeric
<https://docs.nvidia.com/cupynumeric>`_, which is also available as a wheel:

.. code-block:: sh

   $ pip install nvidia-cupynumeric

.. important::

   Packages are only offered for Linux (x86_64 and aarch64) supporting Python
   versions 3.11 to 3.13.

Networking with Legate Packages
===============================

.. toctree::
    :maxdepth: 1

    networking-wheels
    mpi-wrapper

Building and Installing from Source
===================================

Building Legate from source has multiple steps and can involve different dependencies,
depending on your system configuration. For the most up to date instructions for the
latest source code, see :ref:`legate_source_build`.

.. _support_matrix:

Support Matrix
==============

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
     - Ubuntu 22.04
     - other Linux, macOS
   * - Compilers
     - GCC 11
     - Any compiler supporting C++17
   * - GPU architecture
     - Volta
     - Pascal
   * - CUDA toolkit
     - 12.2
     - 11.8
   * - Python
     - 3.11, 3.12, 3.13
     -
   * - NumPy
     - 1.22
     -

Legate is tested and guaranteed to be compatible with Volta and later GPU
architectures. You can use Legate with Pascal GPUs as well, but there could
be issues due to lack of independent thread scheduling. Please report any such
issues by opening a bug.

Installation of the Legate IPython Kernel
=========================================

Please install Legate, then run the following command to install a default
Jupyter kernel:

.. code-block:: sh

   $ legate-jupyter

If installation is successful, you will see some output like the following:

.. code-block::

   Jupyter kernel spec Legate_SM_GPU (Legate_SM_GPU) has been installed

``Legate_SM_GPU`` is the default kernel name.

Licenses
========

This project will download and install additional third-party open source
software projects at install time. Review the license terms of these open
source projects before use.

For license information regarding projects bundled directly, see
:ref:`third_party`.
