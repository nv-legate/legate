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

Example
-------

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

   $ LEGATE_IO_USE_VFD_GDS=1 legate \
     --launcher mpirun \
     --ranks-per-node 1 \
     --gpus 1 \
     --gpu-bind 0 \
     --cpu-bind 48-63 \
     --mem-bind 3 \
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

   $ LEGATE_IO_USE_VFD_GDS=1 legate \
     --launcher mpirun \
     --ranks-per-node 4 \
     --gpus 1 \
     --gpu-bind 0/1/2/3 \
     --cpu-bind 48-63/176-191/16-31/144-159 \
     --mem-bind 3/3/1/1 \
     --sysmem 15000 \
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

GDS Performance Tuning
----------------------
If the HDF5 datasets are larger, then the following tuning helps improve the performance.
Edit ``/etc/cufile.json`` and add the following line under properties section (for 12.6 CUDA
release and above):

.. code-block::


   "properties": {
     ...
	 "per_buffer_cache_size_kb" : 16384,
	 ...
   }

For more GDS specific performance tuning, please refer to
https://docs.nvidia.com/gpudirect-storage/troubleshooting-guide/index.html
