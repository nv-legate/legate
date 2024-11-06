.. _usage:

Usage
=====

Running Legate Programs
-----------------------

Python programs using the Legate APIs (directly or through a Legate-based
library such as cuNumeric) can be run using the standard Python interpreter:

.. code-block:: sh

    $ python myprog.py <myprog.py options>

By default this invocation will use most of the hardware resources (e.g. CPU
cores, RAM and GPUs) available on the current machine, but this can be
controlled, see `Resource Allocation`_.

You can also use the custom ``legate`` driver script, that makes configuration
easier, and offers more execution options, particularly for `Distributed
Launch`_:

.. code-block:: sh

    $ legate myprog.py <myprog.py options>

Compiled C++ programs using the Legate API can also be run using the ``legate``
driver:

.. code-block:: sh

    $ legate ./myprog <myprog options>

or invoked directly:

.. code-block:: sh

    $ ./myprog <myprog options>

Resource Allocation
-------------------

By default Legate will query the available hardware on the current machine, and
reserve for its use all CPU cores, all GPUs and most of the available memory.
You can use ``LEGATE_SHOW_CONFIG=1`` to inspect the exact set of resources that
Legate decided to reserve. You can fine-tune Legate's default resource
reservation by passing specific flags to the ``legate`` driver script, listed
later in this section.

You can also use ``LEGATE_AUTO_CONFIG=0`` to disable Legate's automatic
configuration. In this mode Legate will only reserve a minimal set of resources
(only 1 CPU core for task execution, no GPUs, minimal system memory allocation),
and any increases must be specified manually.

The following ``legate`` flags control how many processors are used by Legate:

* ``--cpus``: how many individual CPU threads are spawned
* ``--omps``: how many OpenMP groups are spawned
* ``--ompthreads``: how many threads are spawned per OpenMP group
* ``--gpus``: how many GPUs are used

The following flags control how much memory is reserved by Legate:

* ``--sysmem``: how much DRAM (in MiB) to reserve
* ``--numamem``: how much NUMA-specific DRAM (in MiB) to reserve per NUMA node
* ``--fbmem``: how much GPU memory (in MiB) to reserve per GPU

See ``legate --help`` for a full list of accepted configuration options.

For example, if you wanted to use only part of the resources on a DGX station,
you might run your application as follows:

.. code-block:: sh

    $ legate --gpus 4 --sysmem 1000 --fbmem 15000 myprog.py

This will make only 4 of the 8 GPUs available for use by Legate. It will also
allow Legate to consume up to 1000 MiB of DRAM and 15000 MiB of each GPU's
memory for a total of 60000 MiB of GPU memory.

The same configuration can also be passed through the environment variable
``LEGATE_CONFIG``:

.. code-block:: sh

    $ LEGATE_CONFIG="--gpus 4 --sysmem 1000 --fbmem 15000" legate myprog.py

including when using the standard Python interpreter:

.. code-block:: sh

    $ LEGATE_CONFIG="--gpus 4 --sysmem 1000 --fbmem 15000" python myprog.py

or when running a compiled C++ Legate program directly:

.. code-block:: sh

    $ LEGATE_CONFIG="--gpus 4 --sysmem 1000 --fbmem 15000" ./myprog

To see the full list of arguments accepted in ``LEGATE_CONFIG``, you can pass
``LEGATE_CONFIG="--help"``:

.. code-block:: sh

    $ LEGATE_CONFIG="--help" ./myprog

You can also allocate resources when running in interactive mode (by not passing
any ``*.py`` files on the command line):

.. code-block:: sh

    $ legate --gpus 4 --sysmem 1000 --fbmem 15000
    Python 3.12.4 | packaged by conda-forge | (main, Jun 17 2024, 10:23:07) [GCC 12.3.0] on linux
    Type "help", "copyright", "credits" or "license" for more information.
    >>>

Distributed Launch
------------------

You can run your program across multiple nodes by using the ``--nodes`` option
followed by the number of nodes to be used. When doing a multi-process run, a
launcher program must be specified, that will do the actual spawning of the
processes:

.. code-block:: sh

    $ legate --nodes 2 --launcher mpirun --cpus 4 --gpus 1 myprog.py

In the above invocation the ``mpirun`` launcher will be used to spawn one Legate
process on each of two nodes. Each process will use 4 CPU cores and 1 GPU on its
assigned node.

The default Legate conda packages include networking support based on UCX, but
:ref:`GASNet-based<gasnet>` packages are also available.

Note that resource setting flags such as ``--cpus 4`` and ``--gpus 1`` refer to
each process. In the above invocation, each one of the two launched processes
will reserve 4 CPU cores and 1 GPU, for a total of 8 CPU cores and 2 GPUs across
the whole run.

Check the output of ``legate --help`` for the full list of supported launchers.

You can also perform the same launch as above externally to ``legate``:

.. code-block:: sh

    $ mpirun -n 2 -npernode 1 legate --cpus 4 --gpus 1 myprog.py

or use ``python`` directly:

.. code-block:: sh

    $ LEGATE_CONFIG="--cpus 4 --gpus 1" mpirun -n 2 -npernode 1 -x LEGATE_CONFIG python myprog.py

Multiple processes ("ranks") can also be launched on each node, using the
``--ranks-per-node`` ``legate`` option:

.. code-block:: sh

    $ legate --ranks-per-node 2 --launcher mpirun myprog.py

The above will launch two processes on the same node (the default value for
``--nodes`` is 1).

Because Legate's automatic configuration will not check for other processes
sharing the same node, each of these two processes will attempt to use the full
set of CPU cores on the node, causing contention. Even worse, each process will
try to reserve most of the system memory in the machine, leading to a memory
reservation failure at startup.

To work around this, you will want to explicitly reduce the resources requested
by each process:

.. code-block:: sh

    $ legate --ranks-per-node 2 --launcher mpirun --cpus 4 --sysmem 1000 myprog.py

With this change, each process will only reserve 4 CPU cores and 1000 MiB of
system memory, so there will be enough resources for both.

Even with the above change contention remains an issue, as the processes may end
up overlapping on their use of CPU cores. To work around this, you can
explicitly partition CPU cores between the processes running on the same node,
using the ``--cpus-bind`` ``legate`` option:

.. code-block:: sh

    $ legate --ranks-per-node 2 --launcher mpirun --cpus 4 --sysmem 1000 --cpu-bind 0-15/16-32 myprog.py

The above command will restrict the first process to CPU cores 0-15, and the
second to CPU cores 16-32, thus removing any contention. Each process will
reserve 4 out of its allocated cores for task execution.

You can similarly restrict processes to specific NUMA domains, GPUs and NICs
using ``--mem-bind``, ``--gpu-bind`` and ``--nic-bind`` respectively.

You can also launch multiple processes per node when doing an external launch,
but you then have to manually control the binding of resources:

.. code-block:: sh

    $ mpirun -n 2 -npernode 2 --bind-to socket legate --cpus 4 --sysmem 1000 myprog.py

The above will launch two processes on one node, and relies on ``mpirun`` to
bind each process to a separate CPU socket, thus partitioning the CPU cores
between them.

Debugging and Profiling
-----------------------

Legate comes with tools that you can use to better understand your program both
from a correctness and a performance standpoint.

For correctness, Legate has facilities for visualizing the dataflow and task
graph from a run of the application. First you need to run your application with
the ``--spy`` ``legate`` option (or pass the same through ``LEGATE_CONFIG``):

.. code-block:: sh

    legate --spy myprog.py

Legate will collect the necessary logging information in the ``legate_*.log``
files (one per process). To post-process these files, install `GraphViz
<https://www.graphviz.org/>`_ on your machine, then run:

.. code-block:: sh

    legion_spy.py --dataflow --event legate_*.log

This command will produce a dataflow visualization in ``dataflow_[...].pdf``,
and a task graph visualization in ``event_[...].pdf``. Note that these files can
grow to be fairly large for non-trivial programs so we encourage you to keep
your programs small when using these visualizations.

For profiling, first you need to install the Legate profile viewer, available on
the Legate conda channel as ``legate-profiler``:

.. code-block:: sh

    conda install -c conda-forge -c legate legate-profiler

Then you need to pass the ``--profile`` flag to the ``legate`` driver when
launching the application (or through ``LEGATE_CONFIG``):

.. code-block:: sh

    legate --profile myprog.py

At the end of execution you will have a set of ``legate_*.prof`` files (one per
process), that can be opened with the profile viewer, to see a timeline of your
program's execution:

.. code-block:: sh

    legion_prof view legate_*.prof

We recommend that you do not mix debugging and profiling in the same run, as
some of the logging for the debugging features requires significant file I/O
that can adversely effect the performance of the application.
