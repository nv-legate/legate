

Running Legate programs with Jupyter Notebook
=============================================

Same as normal Python programs, Legate programs can be run
using Jupyter Notebook. Currently we support single node execution with
multiple CPUs and GPUs, and plan to support multi-node execution in the future.
We leverage Legion's Jupyter support, so you may want to refer to the
`relevant section in Legion's README <https://github.com/StanfordLegion/legion/blob/master/jupyter_notebook/README.md>`_.
To simplify the installation, we provide a script specifically for Legate libraries.

Running with Jupyter Notebook
-----------------------------

You will need to start a Jupyter server, then you can use a Jupyter notebook
from any browser. Please refer to the following two sections from the README of
the `Legion Jupyter Notebook extension <https://github.com/StanfordLegion/legion/tree/master/jupyter_notebook>`_.

* Start the Jupyter Notebook server
* Use the Jupyter Notebook in the browser

Configuring the Jupyter Notebook
--------------------------------

The Legate Jupyter kernel is configured according to the command line
arguments provided at install time.  Standard ``legate`` options for Core,
Memory, and Multi-node configuration may be provided, as well as a name for
the kernel:

.. code-block:: sh

    legate-jupyter --name legate_cpus_2 --cpus 2

Other configuration options can be seen by using the ``--help`` command line
option.

Magic Command
-------------

We provide a Jupyter magic command to display the IPython kernel configuration.

.. code-block::

    %load_ext legate.jupyter
    %legate_info

results in output:

.. code-block::

    Kernel 'Legate_SM_GPU' configured for 1 node(s)

    Cores:
    CPUs to use per rank : 4
    GPUs to use per rank : 0
    OpenMP groups to use per rank : 0
    Threads per OpenMP group : 4
    Utility processors per rank : 2

    Memory:
    DRAM memory per rank (in MBs) : 4000
    DRAM memory per NUMA domain per rank (in MBs) : 0
    Framebuffer memory per GPU (in MBs) : 4000
    Zero-copy memory per rank (in MBs) : 32
    Registered CPU-side pinned memory per rank (in MBs) : 0
