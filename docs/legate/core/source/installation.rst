Installation
============

How Do I Install Legate
-----------------------

Legate Core is available from `conda <https://docs.conda.io/projects/conda/en/latest/index.html>`_
on the `legate channel <https://anaconda.org/legate/legate-core>`_.
Please make sure you have at least conda version 24.1 installed, then create
a new environment containing Legate Core:


.. code-block:: sh

    conda create -n myenv -c conda-forge -c legate legate-core

or install it into an existing environment:

.. code-block:: sh

    conda install -c conda-forge -c legate legate-core

Only linux-64 packages are available at the moment.

The default package contains GPU support, and is compatible with CUDA >= 12.0
(CUDA driver version >= r520), and Volta or later GPU architectures. There are
also CPU-only packages available, and will be automatically selected when
installing on a machine without GPUs. You can force installation of a CPU-only
package by requesting it as follows:

.. code-block:: sh

    conda ... legate-core=*=*_cpu

See `BUILD.md <BUILD.md>`_ for instructions on building Legate Core from source.

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