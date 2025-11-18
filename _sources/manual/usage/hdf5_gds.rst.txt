..
  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

Using GPUDirect Storage to perform HDF5 file I/O
================================================

Overview
---------

This document outlines the enablement of GPUDirect Storage(GDS in short) to perform I/O
between the GPU memory and the underlying storage device specifically for HDF5 files in an
efficient manner.

GPUDirect Storage enables a direct path between local or remote storage and GPU memory
avoiding an extra copies through a bounce buffer in the CPU's memory and enables a direct
memory access (DMA) engine near the NIC or storage to move data on a direct path into or
out of GPU memory, all without burdening the CPU or GPU.

HDF5 library provides a flexibility to attach any custom virtual file driver for I/O. In
Legate, HDF5 library can use vfd-gds virtual driver which in turn would call GDS specific
APIs for the i/o.

Installation
------------

Please refer to https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html
for the installation instructions of GPUDirect Storage.

The vfd-gds library will be installed as part of legate installation.

In order to enable GDS for the HDF5 library, do the following before starting legate
environment.

.. code-block:: sh

   $ export LEGATE_IO_USE_VFD_GDS=1

Read Example
------------

Below is the output of an example python program which iterates all the datasets in a HDF5
file and reads the data. The source code for the example may be found at
https://github.com/nv-legate/legate/blob/main/share/legate/examples/io/hdf5/ex1.py.

The program prints throughput numbers based on the total amount of data read over total
elapsed time. Please note that this is not a standard benchmark to calculate I/O
throughput. However, it can give an approximate throughput value.

Usage Example - 1
------------------

Assuming the HDF5 data set at /path/to/hdf/data exists, running the example with 1 rank:

.. code-block:: sh

   $ legate \
     --launcher mpirun \
     --ranks-per-node 1 \
     --gpus 1 \
     --gpu-bind 0 \
     --cpu-bind 48-63 \
     --mem-bind 3 \
     --io-use-vfd-gds \
     --sysmem 15000 \
     --fbmem 80000 \
     --zcmem 5000 \
     share/legate/examples/io/hdf5/ex1.py /path/to/hdf/data --n_rank 1
   IO MODE : GDS
   Total Data Read: 17179869184
   Total Turnaround Time (seconds): 31.506664514541626
   Throughput (MB/sec): 520.0169631551479

Usage Example - 1
------------------

Running the example with 4 ranks:

.. code-block:: sh

   $ legate \
     --launcher mpirun \
     --ranks-per-node 4 \
     --gpus 1 \
     --gpu-bind 0/1/2/3 \
     --cpu-bind 48-63/176-191/16-31/144-159 \
     --mem-bind 3/3/1/1 \
     --sysmem 15000 \
     --io-use-vfd-gds \
     --fbmem 80000 \
     --zcmem 5000 \
     share/legate/examples/io/hdf5/ex1.py /path/to/hdf/data --n_rank 4
   IO MODE : GDS
   IO MODE : GDS
   IO MODE : GDS
   IO MODE : GDS
   Total Data Read: 68719476736
   Total Turnaround time (seconds): 72.61114501953125
   Throughput (MB/sec): 902.5611699467328
   Total Data Read: 68719476736
   Total Turnaround time (seconds): 72.72363376617432
   Throughput (MB/sec): 901.1650904397261
   Total Data Read: 68719476736
   Total Turnaround time (seconds): 72.35940861701965
   Throughput (MB/sec): 905.7011555589922
   Total Data Read: 68719476736
   Total Turnaround time (seconds): 72.73599338531494
   Throughput (MB/sec): 901.0119605135058

Write Example
-------------

The write example can be found at https://github.com/nv-legate/legate/blob/main/share/legate/examples/io/hdf5/hdf5_write_benchmark.py.

The example writes data to a HDF5 file and measures the throughput.

Example Usage
-------------

.. code-block:: sh

   $ legate \
     --launcher mpirun \
     --ranks-per-node 1 \
     --io-use-vfd-gds \
     --gpus 1 \
     --gpu-bind 1 \
     --sysmem 15000 \
     --fbmem 40000 \
     --zcmem 5000 \
     share/legate/examples/io/hdf5/hdf5_write_benchmark.py --output-dir /path/to/output --sizes 1000000 --dtypes float32 --iterations 3

      ===============================================================================
      HDF5 WRITE BENCHMARK
      ================================================================================
      Output directory: output
      Sizes: [1000000]
      Data types: ['float32']
      Iterations per config: 3
      ================================================================================

      Benchmarking size=1,000,000, dtype=float32
      Iteration 1: Wall=0.444s, Legate=0.240s, Throughput=15.88 MB/s
      Iteration 2: Wall=0.014s, Legate=0.014s, Throughput=267.70 MB/s
      Iteration 3: Wall=0.010s, Legate=0.010s, Throughput=398.15 MB/s
      Average: Wall=0.156s, Legate=0.088s, Throughput=227.24 MB/s

      ================================================================================
      BENCHMARK SUMMARY
      ================================================================================
            Size     Type         MB    Wall(s)  Legate(s)   Throughput
      --------------------------------------------------------------------------------
         1,000,000  float32       3.81      0.156      0.088     227.24 MB/s
      ================================================================================

      Best throughput: 227.24 MB/s (size=1,000,000, dtype=float32)

GDS Not Available
-----------------

If GDS is not available in the system, the user can still use the
``LEGATE_CONFIG="--io-use-vfd-gds"`` with the cuFile compatibility mode.
In that case the user needs to set the following environment variable:

.. code-block:: sh

   $ export CUFILE_ALLOW_COMPAT_MODE='true'

GDS Performance Tuning
----------------------
If the HDF5 datasets are larger, then the following tuning helps improve the performance.
Edit ``/etc/cufile.json`` and add the following line under properties section
(for 12.6 CUDA release and above):

.. code-block::

   "execution" : {
            ...
            // max number of host threads per gpu to spawn for parallel IO
            "max_io_threads" : 8,
            // enable support for parallel IO
            "parallel_io" : true,
            // maximum parallelism for a single request
            "max_request_parallelism" : 8
            ...
   },

   "properties": {
     ...
	 "per_buffer_cache_size_kb" : 16384,
	 ...
   }

For more GDS specific performance tuning, please refer to
https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html

Usually cuFile.json is located at /etc/cufile.json. The user can specify the
path to the cuFile.json file using the CUFILE_CONFIG_FILE environment variable.

.. code-block:: sh

   $ export CUFILE_ENV_PATH_JSON=/path/to/cufile.json
