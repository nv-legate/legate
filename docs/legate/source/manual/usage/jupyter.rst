

Running Legate programs with Jupyter Notebook
=============================================

Legate leverages `Legion's Jupyter Notebook support <https://github.com/StanfordLegion/legion/blob/master/jupyter_notebook/>`_
to enable Python programs to run in Jupyter Notebook environments.
Currently we support single node execution with
multiple CPUs and GPUs, and plan to support multi-node execution in the future.

Installing the Customized Legate Jupyter Notebook Kernel
--------------------------------------------------------

Legate provides the "legate-jupyter" script
for installing a customized Jupyter kernel tailored for Legate libraries.
Run the following command to install the Legate Jupyter kernel:

.. code-block:: sh

    legate-jupyter --name <kernel-name> --cpus <num-cpus> --gpus <num-gpus> <other-configurations>

The kernel's configuration will match the arguments provided during installation.
For a full list of configuration options, use:

.. code-block:: sh

    legate-jupyter --help

Each unique configuration requires creating and installing a new kernel using "legate-jupyter".
To view all installed kernels, run the following command:

.. code-block:: sh

    jupyter kernelspec list

Once installed, you can select one of the configured kernels to run Legate programs.

Running with Jupyter Notebook
-----------------------------

To run a Legate program using Jupyter Notebook, you first need to start a Jupyter Notebook server
and then access it through your browser. Here are the detailed steps:

Launch the Jupyter Notebook server on the machine where Legate will be executed:

.. code-block:: sh

    jupyter notebook --port=<port-number> --no-browser

A "token" value will be displayed on the terminal once the server starts.

Open your browser and navigate to:

.. code-block:: sh

    http://localhost:<port-number>/?token=<token>

to access the Jupyter Notebook.

If the server is running on a remote machine, you need to establish an SSH tunnel from your local machine to the remote server:

.. code-block:: sh

    ssh -4 -t -L <port-number>:localhost:<local-port-number> <username>@<remote-server-hostname> ssh -t -L <local-port-number>:localhost:<port-number> <remote-server-hostname>

Once the Jupyter Notebook page loads, click "New" in the top-right corner and select one of the installed Legate kernels from the dropdown menu. A new window/tab will be opened for you, to run your Python programs.

Magic Command
-------------

We provide a Jupyter magic command to display the IPython kernel configuration. This input:

.. code-block::

    %load_ext legate.jupyter
    %legate_info

will print out something like:

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
