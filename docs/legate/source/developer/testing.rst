..
  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. _ch_testing:

=================
Legate Test Suite
=================

Quickstart
==========

Build and run the C++ tests:

.. code-block:: sh

   $ ./configure --with-tests
   $ make
   $ ./test.py

Build and run the Python tests:

.. code-block:: sh

   $ ./configure --with-python
   $ make
   $ pip install .
   $ pytest ./tests/python

Build and run both the C++ and Python tests:

.. code-block:: sh

   $ ./configure --with-tests --with-python
   $ make
   $ pip install .
   $ ./test.py
   $ pytest ./tests/python


Build and run the C++ benchmarks:

.. code-block:: sh

   $ ./configure --with-benchmarks
   $ make
   $ $LEGATE_DIR/$LEGATE_ARCH/cmake_build/cpp/benchmarks/bin/*

.. _sec_testing_cpp:

C++
===

Building
--------

In order to run the C++ tests, Legate must first be configured to build the tests. This
must be done during the ``configure`` call, **as the tests cannot be built standalone**:

.. code-block:: sh

   $ ./configure --with-tests
   # Or, if you have an existing configuration to which you'd like to add test
   # support:
   $ ./reconfigure-$LEGATE_ARCH.py --with-tests
   $ make


The tests will then be built alongside the main library during ``make``.

.. note::

   ``--with-tests`` only enables the C++ tests. It does *not* enable the Python bindings,
   or cause any of the Python tests to be enabled.

   To be able to run those, Legate must also be built with Python binding support, via the
   ``--with-python`` flag. See :ref:`sec_testing_python` for more information on how to
   run the Python tests, and :ref:`legate_source_build` for further discussion on how to
   build Legate itself.


Running
-------

The tests may be run in 1 of 2 ways: direct, or via the test runner (``test.py``) located
in the top-level directory (``$LEGATE_DIR``). It is advised that developers run them via
the test runner, as it provides a more ergonomic interface to the testing
infrastructure. ``test.py`` will also ensure several other Google-test specific
constraints are handled (such as running "death tests" in individual processes), that you
would otherwise have to do manually.

To run via the test runner, simply invoke it:

.. code-block:: sh

   $ ./test.py
   == Test Suite Configuration (GTest) ============================================
   |                                                                              |
   |        Feature stages  cpus                                                  |
   |    System description  12 cpus / N/A gpus                                    |
   |  Test files per stage  4                                                     |
   |                                                                              |
   ================================================================================
   --------------------------------------------------------------------------------
   | Entering stage: CPU (with 4 workers)                                         |
   --------------------------------------------------------------------------------
   [PASS] (CPU) 3.39s {15:51:27.82, 15:51:31.21} AlignedUnpack.Bug1
   [PASS] (CPU) 3.52s {15:51:27.82, 15:51:31.34} AliasViaPromote.Bug1
   [PASS] (CPU) 1.17s {15:51:31.22, 15:51:32.39} DanglingStorePartition.Bug1
   [PASS] (CPU) 1.17s {15:51:31.34, 15:51:32.51} LogicalStoreTransform.SliceBug1
   [PASS] (CPU) 1.19s {15:51:32.39, 15:51:33.58} LogicalStoreTransform.SliceBug2
   [PASS] (CPU) 1.44s {15:51:32.52, 15:51:33.95} LogicalStoreTransform.WeightedBug1
   ...


You may further filter the list of tests being run via the ``--gtest-filter`` flag for
``test.py``. This flag accepts a standard Python regular expression that will be applied
to the list of tests before running them:

.. code-block:: sh

   $ ./test.py --gtest-filter='LogicalStore.*ReinterpretAs'
   == Test Suite Configuration (GTest) ============================================
   |                                                                              |
   |        Feature stages  cpus                                                  |
   |    System description  12 cpus / N/A gpus                                    |
   |  Test files per stage  4                                                     |
   |                                                                              |
   ================================================================================
   --------------------------------------------------------------------------------
   | Entering stage: CPU (with 4 workers)                                         |
   --------------------------------------------------------------------------------
   [PASS] (CPU) 1.41s {15:54:49.28, 15:54:50.69} LogicalStoreUnit/ReinterpretAs.Basic/0
   [PASS] (CPU) 1.41s {15:54:49.28, 15:54:50.69} LogicalStoreUnit/ReinterpretAs.Basic/1
   [PASS] (CPU) 1.41s {15:54:49.28, 15:54:50.69} LogicalStoreUnit/ReinterpretAs.Basic/2
   --------------------------------------------------------------------------------
   | Exiting stage: CPU                                                           |
   |                                                                              |
   | Passed 3 of 3 tests (100.0%) in 1.42s                                        |
   --------------------------------------------------------------------------------
   --------------------------------------------------------------------------------
   == Overall summary =============================================================
   |                                                                              |
   | * CPU   : 3 / 3 passed in 1.42s                                              |
   |                                                                              |
   | Passed 3 of 3 tests (100.0%) in 1.42s                                        |
   |                                                                              |
   ================================================================================


.. note::

   Any additional command-line arguments to ``test.py`` (that it does not natively
   understand, such as ``--use cuda`` or ``--gpus 2``) are forwarded as-is to the
   executable. This allows the tester to handle any GoogleTest specific arguments
   transparently.

   See ``./test.py --help`` for a complete description of the arguments that it
   understands.


You may also run the tests directly, by invoking the raw executables:

.. code-block:: sh

   $ $LEGATE_DIR/$LEGATE_ARCH/cmake_build/cpp/tests/bin/tests_wo_runtime
   ...


.. warning::

   Running the tests directly is discouraged. Their exact location on disk is not
   considered stable; if they ever move, ``test.py`` will always be updated to find them,
   but any scripts manually running them won't be.


Running Tests Across Multiple Processes
---------------------------------------

When running the tests via ``test.py``, you may instruct it to run each test across
multiple processes on the same node (using ``--ranks-per-node``) or even across
multiple nodes (using ``--nodes``). In this case you also need to specify a ``--launcher``.

Each test will then be spawned using the given configuration. These arguments behave
identically to the corresponding ``legate`` launcher arguments, see
:ref:`ch_usage_running` for further discussion on the semantics.

You may further parallelize *within* processes by setting the various
``LEGATE_CONFIG`` flags. See :ref:`ch_usage_running` for further discussion on these
various flags.

.. _sec_testing_python:

Python
======

The python tests are invoked directly via ``pytest``. As such, you must install a set of
pre-requisite Python packages before you can run them (note the following list may not be
complete):

- ``pytest``
- ``pytest-cov``
- ``pytest-mock``
- ``psutil``

A full list of requirements may be found in ``pyproject.toml``. Note that some of these
dependencies are only required on GPU builds, others only when testing Jupyter notebook
related functionality.

Building
--------

In order to run the Python tests, the Legate Python bindings must first be built and
installed. See :ref:`build_python_bindings` for more discussion on how best to build the
Python bindings for Legate.

.. code-block:: sh

   $ ./configure --with-python
   $ make
   $ pip install .


.. note::

   Building the Python bindings does *not* imply that the C++ tests are built. To also build
   the C++ tests, you must additionally pass ``--with-tests`` to ``configure``.


Running
-------

The tests are run in the usual way, by invoking ``pytest`` on the directory or test you
wish to execute:

.. code-block:: sh

   $ pytest tests/python/unit/legate/core/test_task.py
   ============================= test session starts =============================
   platform darwin -- Python 3.13.3, pytest-8.3.5, pluggy-1.6.0
   cachedir: .cache/pytest
   rootdir: /Users/jfaibussowit/soft/nv/legate.core.internal
   configfile: pyproject.toml
   plugins: mock-3.14.1
   collected 186 items

   tests/python/unit/legate/core/test_task.py ............................ [ 15%]
   ....................................................................... [ 53%]
   ................................xs..................................... [ 91%]
   ................                                                        [100%]

   ================== 184 passed, 1 skipped, 1 xfailed in 1.06s ==================


Any Legate-related arguments must be passed via the ``LEGATE_CONFIG`` environment variable:


.. code-block:: sh

   $ LEGATE_CONFIG='--cpus 2 --logging legate=debug' pytest path/to/tests
   ...
