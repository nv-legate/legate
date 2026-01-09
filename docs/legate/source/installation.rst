..
  SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

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

   Packages are offered for Linux (x86_64 and aarch64) and macOS (aarch64,
   pip wheels only), supporting Python versions 3.11 to 3.13. Windows is only
   supported through WSL.

Nightly top-of-tree builds of Legate are available on a separate channel,
`legate-nightly`:

.. code-block:: bash

   $ conda install -c conda-forge -c legate-nightly legate

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

   Packages are offered for Linux (x86_64 and aarch64) and macOS (aarch64,
   pip wheels only), supporting Python versions 3.11 to 3.13. Windows is only
   supported through WSL.

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

The following table lists Legate's supported version range for major
dependencies. The listed versions are being tested regularly, and supported by
our downloadable packages.

.. list-table:: Support Matrix
   :header-rows: 1

   * - Dependency
     - Versions supported
   * - CPU architecture
     -
        * x86-64 (Haswell and later)
        * aarch64 (armv8-a and later)
   * - OS
     -
        * Ubuntu (22.04 and later)
        * macOS (11 and later, ARM-based only)
        * WSL
   * - Compilers
     -
        * GCC (10-14)
   * - GPU architecture
     - Volta and later
   * - CUDA toolkit
     - 12.2 and later
   * - Python
     - 3.11, 3.12, 3.13
   * - NumPy
     - 1.22 and later

You may be able to build Legate from source and run under other configurations,
but no guarantees are given that Legate will work properly in that case. We may
not actively work to fix any incompatibilities discovered under unsupported
configurations, but we accept contributions that fix such incompatibilities.

In particular, if you try to use Legate on Pascal (or earlier) GPUs, there could
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
