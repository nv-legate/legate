..
  SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

:html_theme.sidebar_secondary.remove:

NVIDIA Legate
=============

The Legate project endeavors to democratize computing by making it possible for
all programmers to leverage the power of large clusters of CPUs and GPUs by
running the same code that runs on a desktop or a laptop at scale. Using this
technology, computational and data scientists and researchers can develop and
test programs on moderately sized data sets on local machines and then
immediately scale up to larger data sets deployed on many nodes in the cloud or
on a supercomputer without any code modifications.

.. code-block:: python

   from legate.core import get_legate_runtime, types
   from legate.core.task import task, InputArray, OutputArray

   @task
   def saxpy(x: InputArray, alpha: float, y: OutputArray) -> None:
       xarr = np.asarray(x)
       yarr = np.asarray(y)

       yarr[:] = alpha * xarr[:] + yarr[:]


   runtime = get_legate_runtime()
   x = runtime.create_array(dtype=types.int32, shape=(100,))
   x.fill(123)
   y = runtime.create_array_like(x)

   saxpy(x, 10.0, y)


.. toctree::
  :maxdepth: 2
  :caption: Contents:

  Overview <overview.rst>
  Installing pre-built packages <installation.rst>
  Building from source <BUILD.rst>
  Contributing Guide <CONTRIBUTING.rst>
  Frequently Asked Questions <faq.rst>
  User Manual <manual/index.rst>
  API Reference <api/index.rst>

.. toctree::
  :maxdepth: 1

  Versions <versions.rst>
  Changelog <changes/index.rst>
  Third-party notices <oss-licenses.rst>

.. toctree::
  :maxdepth: 2
  :caption: Experimental Features:

  Legate.STL <legate.stl/source/legate-stl.rst>


.. toctree::

   Developer Documentation <developer/index.rst>
   GASNet-based Installation <gasnet.rst>
   Links to resources <resources>


Indices and tables
------------------

* :ref:`genindex`
* :ref:`search`
