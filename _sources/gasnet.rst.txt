..
  SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0

.. _gasnet:

GASNet-based Installation
=========================

.. _gasnet_overview:

Overview
--------

The :ref:`Installation<installation>` part of the guide describes the general
installation process.  However, the general Conda packages are built with `UCX
<https://openucx.org/>` networking, which is not compatible with some
supercomputers.  For example, the
`Perlmutter<https://docs.nersc.gov/systems/perlmutter/architecture/>` HPE Cray
supercomputer uses the Cray Slingshot 11 interconnect fabric with HPE Cray's
proprietary Cassini NICs.  This interconnect is not currently supported by the
UCX networking backend.  For machines that require a non-UCX backend, we provide
GASNet versions of the Legate packages, which leverage Legate's ability to
use `GASNet<https://gasnet.lbl.gov/>` for the networking backend.  The GASNet
backend supports many interconnects, including the Slingshot 11 interconnect.

Because the GASNet backend packages are meant for customized environments, these
packages do not include any networking components. They require that the MPI
communicator module and the GASNet networking backend be built by the user
after installing the Legate
package.  Thus, a GASNet package installation requires three components:

* The Legate package built with the GASNet wrapper support.
* A GASNet wrapper built for a particular environment.
* An MPI wrapper built for a particular environment.

The installation of the MPI wrapper is discussed in more detail in
:ref:`Installation<installation_of_mpi_wrapper>` and :ref:`MPI wrapper
FAQ<mpi_wrapper_faq>`.  In summary, the idea is to provide an API wrapper around
MPI that can be plugged into Leagate at start time.  The GASNet wrapper has a
similar purpose, but plugs into the underlying Realm networking module.  To make
this process easier, instead of building the wrappers from source, we
provide an MPI wrapper and a GASNet wrapper conda packages that install scripts
for building of the wrappers, and automate their use by setting the necessary
environment variables.

.. _how-do-i-install-legate_with_wrappers:

How Do I Install Legate with the MPI and GASNet wrappers
--------------------------------------------------------

The basics of the installation of Leagate are covered in :ref:`How Do I Install
Legate<how-do-i-install-legate>`.  Here, we discuss the differences from the
non-wrapper packages.

First, the GASNet wrapper Legate packages live under a different label in the
Legate conda repository.  The reason for this is to keep the UCX and the GASNet
packages separated, so they do not conflict with each other.  So, the
installation of the Legate GASNet package differs only slightly from the UCX
package installation:

.. code-block:: sh

    conda create -n myenv -c conda-forge -c legate/label/gex legate realm-gex-wrapper legate-mpi-wrapper

In addition to the Legate package, the user should install the wrapper packages
that contain the scripts necessary to build the wrappers. When the packages are
installed, the instructions for building the wrappers are displayed:

.. code-block:: sh

    To finish configuring the Legate MPI wrapper, activate your environment and run /conda/env/path/mpi-wrapper/build-mpi-wrapper.sh

    To finish configuring the Realm GASNet-EX wrapper, activate your environment and run /conda/env/path/gex-wrapper/build-gex-wrapper.sh

``/conda/env/path`` will be the path to the conda environment created during the
``conda create`` step.  When both of the scripts are run and the wrappers are
built, the environment needs to be deactivated and reactivated to set up the
necessary environment variables (when the wrapper packages activation scripts
are run).  After the packages are installed, Legate is ready to use.

Perlmutter example
------------------

Here, we will show a complete example of installing Legate on Perlmutter.  This
example will be a good starting point, even on machines with a different setup.

First, on Perlmutter, Conda is not available by default.  The simplest way to
get a working conda environment is to use the Cray module for Conda:

.. code-block:: sh

    module load conda

Next, create a Legate environment:

.. code-block:: sh

    login40:~> conda create -n legate-gex-anaconda -c legate/label/gex legate realm-gex-wrapper legate-mpi-wrapper

This installs Legate and all the necessary dependencies, just like for the UCX
Legate package, but, in addition to the usual output listing the installed
packages, the following message is displayed:

.. code-block:: sh

    To finish configuring the Legate MPI wrapper, activate your environment and run /conda/envs/legate-gex-anaconda/mpi-wrapper/build-mpi-wrapper.sh


    \
    To finish configuring the Realm GASNet-EX wrapper, activate your environment and run /conda/envs/legate-gex-anaconda/gex-wrapper/build-gex-wrapper.sh


    done

    To activate this environment, use

         $ conda activate legate-gex-anaconda

    To deactivate an active environment, use

         $ conda deactivate

To build the wrappers, we must first activate the ``legate-gex-anaconda``
environment we created.

.. code-block:: sh

    login40:~> conda activate legate-gex-anaconda


    --------------------- CONDA/MPI_WRAPPER/ACTIVATE.SH -----------------------

    LEGATE_MPI_WRAPPER=


    --------------------- CONDA/GASNET_WRAPPER/ACTIVATE.SH -----------------------

    REALM_GASNETEX_WRAPPER=
    GASNET_OFI_SPAWNER=mpi
    FI_CXI_RDZV_THRESHOLD=256

Note that when the environment is activated without the wrappers built, the
activation scripts do not find the built libraries (nothing follows the ``=``
mark).  After the environment is activated, we can first build the MPI wrapper
(although the order of building the wrappers does not matter):

.. code-block:: sh

    login40:~> /conda/envs/legate-gex-anaconda/mpi-wrapper/build-mpi-wrapper.sh
    Building Legate MPI wrapper:
      Installation directory: /conda/envs/legate-gex-anaconda/lib
      Compiler: CC
    -- The CXX compiler identification is GNU 12.3.0
    -- Cray Programming Environment 2.7.30 CXX
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Check for working CXX compiler: /opt/cray/pe/craype/2.7.30/bin/CC - skipped
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Using build type: Release
    -- Building shared library: ON
    -- Generating src install rules: ON
    -- Found MPI_CXX: /opt/cray/pe/craype/2.7.30/bin/CC (found version "3.1")
    -- Found MPI: TRUE (found version "3.1") found components: CXX
    -- Configuring done (2.2s)
    -- Generating done (0.0s)
    -- Build files have been written to: /conda/envs/legate-gex-anaconda/mpi-wrapper/build
    [ 50%] Building CXX object CMakeFiles/mpi_wrapper.dir/src/legate_mpi_wrapper/mpi_wrapper.cc.o
    [100%] Linking CXX shared library lib64/liblgcore_mpi_wrapper.so
    [100%] Built target mpi_wrapper
    -- Install configuration: "Release"
    -- Installing: /conda/envs/legate-gex-anaconda/mpi-wrapper/conda/envs/legate-gex-anaconda/mpi-wrapper/lib64/liblgcore_mpi_wrapper.so.1
    -- Installing: /conda/envs/legate-gex-anaconda/mpi-wrapper/conda/envs/legate-gex-anaconda/mpi-wrapper/lib64/liblgcore_mpi_wrapper.so
    -- Installing: /conda/envs/legate-gex-anaconda/mpi-wrapper/conda/envs/legate-gex-anaconda/mpi-wrapper/include/legate_mpi_wrapper/legate_mpi_wrapper/mpi_wrapper.h
    -- Installing: /conda/envs/legate-gex-anaconda/mpi-wrapper/conda/envs/legate-gex-anaconda/mpi-wrapper/include/legate_mpi_wrapper/legate_mpi_wrapper/mpi_wrapper_types.h
    -- Installing: /conda/envs/legate-gex-anaconda/mpi-wrapper/conda/envs/legate-gex-anaconda/mpi-wrapper/lib64/cmake/legate_mpi_wrapper/mpi_wrapperTargets.cmake
    -- Installing: /conda/envs/legate-gex-anaconda/mpi-wrapper/conda/envs/legate-gex-anaconda/mpi-wrapper/lib64/cmake/legate_mpi_wrapper/mpi_wrapperTargets-release.cmake
    -- Installing: /conda/envs/legate-gex-anaconda/mpi-wrapper/conda/envs/legate-gex-anaconda/mpi-wrapper/lib64/cmake/legate_mpi_wrapper/mpi_wrapperConfig.cmake
    -- Installing: /conda/envs/legate-gex-anaconda/mpi-wrapper/conda/envs/legate-gex-anaconda/mpi-wrapper/lib64/cmake/legate_mpi_wrapper/mpi_wrapperConfigVersion.cmake

    Reactivate the conda environment to set the necessary environment variables:

    $ conda deactivate
    $ conda activate legate-gex-anaconda

On Perlmutter, when attempting to build the MPI wrapper, at the time of writing
this document, the installed ``cmake`` is too old:

.. code-block:: sh

    $ /conda/envs/legate-gex-anaconda/mpi-wrapper/build-mpi-wrapper.sh
    Building Legate MPI wrapper:
      Installation directory: /conda/envs/legate-gex-anaconda/lib
      Compiler: CC
    CMake Error at CMakeLists.txt:13 (cmake_minimum_required):
      CMake 3.22.1 or higher is required.  You are running version 3.20.4

In case of CMake version error, a new version can be installed with ``conda
install cmake`` or by any other means.  Note that when the wrapper is built, the
final message suggests reactivating the environment, but that is not necessary
before building the GASNet wrapper:

.. code-block:: sh

    login40:~> /conda/envs/legate-gex-anaconda/gex-wrapper/build-gex-wrapper.sh
    Building GASNet-EX wrapper:
      Installation directory: /conda/envs/legate-gex-anaconda/lib
      Conduit: ofi
      System configuration: slingshot11
    -- The C compiler identification is GNU 12.3.0
    -- The CXX compiler identification is GNU 12.3.0
    -- Cray Programming Environment 2.7.30 C
    -- Detecting C compiler ABI info
    -- Detecting C compiler ABI info - done
    -- Check for working C compiler: /opt/cray/pe/craype/2.7.30/bin/cc - skipped
    -- Detecting C compile features
    -- Detecting C compile features - done
    -- Cray Programming Environment 2.7.30 CXX
    -- Detecting CXX compiler ABI info
    -- Detecting CXX compiler ABI info - done
    -- Check for working CXX compiler: /opt/cray/pe/craype/2.7.30/bin/CC - skipped
    -- Detecting CXX compile features
    -- Detecting CXX compile features - done
    -- Could NOT find GASNet (missing: GASNet_INCLUDE_DIR GASNet_CONDUITS GASNet_THREADING_OPTS)
    -- Configuring and building embedded GASNet...
    -- Downloading StanfordLegion/gasnet repo from: https://github.com/StanfordLegion/gasnet.git
    CMake Warning (dev) at /conda/envs/legate-gex-anaconda/share/cmake-3.30/Modules/FetchContent.cmake:1953 (message):
      Calling FetchContent_Populate(embed-gasnet) is deprecated, call
      FetchContent_MakeAvailable(embed-gasnet) instead.  Policy CMP0169 can be
      set to OLD to allow FetchContent_Populate(embed-gasnet) to be called
      directly for now, but the ability to call it with declared details will be
      removed completely in a future version.
    Call Stack (most recent call first):
      /conda/envs/legate-gex-anaconda/gex-wrapper/cmake/FetchAndBuildGASNet.cmake:105 (FetchContent_Populate)
      CMakeLists.txt:38 (include)
    This warning is for project developers.  Use -Wno-dev to suppress it.

    -- Found MPI_C: /opt/cray/pe/craype/2.7.30/bin/cc (found version "3.1")
    -- Found MPI: TRUE (found version "3.1") found components: C
    -- Found GASNet: /conda/envs/legate-gex-anaconda/gex-wrapper/src/build/embed-gasnet/install/include
    -- Found GASNet Conduits: ofi
    -- Found GASNet Threading models: par
    -- GASNet: Using ofi-par
    -- Performing Test COMPILER_HAS_HIDDEN_VISIBILITY
    -- Performing Test COMPILER_HAS_HIDDEN_VISIBILITY - Success
    -- Performing Test COMPILER_HAS_HIDDEN_INLINE_VISIBILITY
    -- Performing Test COMPILER_HAS_HIDDEN_INLINE_VISIBILITY - Success
    -- Performing Test COMPILER_HAS_DEPRECATED_ATTR
    -- Performing Test COMPILER_HAS_DEPRECATED_ATTR - Success
    -- Configuring done (131.1s)
    -- Generating done (0.1s)
    -- Build files have been written to: /conda/envs/legate-gex-anaconda/gex-wrapper/src/build
    [ 33%] Building CXX object CMakeFiles/realm_gex_wrapper_objs.dir/gasnetex_handlers.cc.o
    [ 66%] Building CXX object CMakeFiles/realm_gex_wrapper_objs.dir/gasnetex_wrapper.cc.o
    [ 66%] Built target realm_gex_wrapper_objs
    [100%] Linking CXX shared library librealm_gex_wrapper.so
    [100%] Built target realm_gex_wrapper
    -- Install configuration: ""
    -- Installing: /conda/envs/legate-gex-anaconda/gex-wrapper/lib/librealm_gex_wrapper.so.0.0.1
    -- Installing: /conda/envs/legate-gex-anaconda/gex-wrapper/lib/librealm_gex_wrapper.so.0
    -- Set non-toolchain portion of runtime path of "/conda/envs/legate-gex-anaconda/gex-wrapper/lib/librealm_gex_wrapper.so.0.0.1" to "$ORIGIN"
    -- Installing: /conda/envs/legate-gex-anaconda/gex-wrapper/lib/librealm_gex_wrapper.so

    Reactivate the conda environment to set the necessary environment variables:

    $ conda deactivate
    $ conda activate legate-gex-anaconda

Now, with both wrappers built, we can reactivate the environment:

.. code-block:: sh

    login40:~> conda deactivate


    --------------------- CONDA/GASNET_WRAPPER/DEACTIVATE.SH -----------------------

    +++ unset REALM_GASNETEX_WRAPPER
    +++ unset GASNET_OFI_SPAWNER
    +++ unset FI_CXI_RDZV_THRESHOLD
    +++ set +x


    --------------------- CONDA/MPI_WRAPPER/DEACTIVATE.SH -----------------------

    +++ unset LEGATE_MPI_WRAPPER
    +++ set +x
    login40:~> conda activate legate-gex-anaconda


    --------------------- CONDA/MPI_WRAPPER/ACTIVATE.SH -----------------------

    LEGATE_MPI_WRAPPER=/conda/envs/legate-gex-anaconda/mpi-wrapper/conda/envs/legate-gex-anaconda/mpi-wrapper/lib64/liblgcore_mpi_wrapper.so


    --------------------- CONDA/GASNET_WRAPPER/ACTIVATE.SH -----------------------

    REALM_GASNETEX_WRAPPER=/conda/envs/legate-gex-anaconda/gex-wrapper/lib/librealm_gex_wrapper.so
    GASNET_OFI_SPAWNER=mpi
    FI_CXI_RDZV_THRESHOLD=256

When the environment is deactivated, the deactivation scripts unset all the
variables, even if they were not set before.  Activating the environment sets
the paths to the wrappers libraries this time because the libraries are built
and available in the expected paths.  After the wrappers are built, Legate jobs
can be run.  When running Legate, the conda environment should be activated
first (for example, load the conda module and activate the Legate environment in
a batch script).  For example, we can create a simple hello world script:

.. code-block:: sh

    cat "print("Hello World")" > hello_world.py

Then, we can create a batch script to run it:

.. code-block:: sh

    #!/usr/bin/env bash
    #SBATCH --qos=debug
    #SBATCH --nodes=2
    #SBATCH --ntasks-per-node=1
    #SBATCH --constraint=gpu
    #SBATCH --gpus-per-node=4
    #SBATCH --time=03:00
    #SBATCH -o legate-%j.out

    module load conda
    conda activate legate-gex-anaconda
    legate --logging legate=1,gex=1 --launcher srun --nodes 2 --ranks-per-node 1 --cpus 1 --sysmem 4000 --gpus 4 --fbmem 4000 --verbose ./hello_world.py

The example batch script needs to be adjusted for a particular situation.  Here,
we launch Legate with some options that are more relevant in more involved
codes, but we show them for exposure.  The important options for invoking Legate
in the Perlmutter environment are ``--launcher srun`` and ``--nodes 2``.  These
options tell the Legate driver to use ``srun`` to launch Legate.  We could
actually provide the ``srun`` command around Legate, and then we would use
``--launcher none`` to prevent the Legate launcher from using any external
launcher.  With our options, Legate run results in the following output on
Perlmutter:

.. code-block:: sh

    --------------------- CONDA/MPI_WRAPPER/ACTIVATE.SH -----------------------

    LEGATE_MPI_WRAPPER=/conda/envs/legate-gex-anaconda/mpi-wrapper/conda/envs/legate-gex-anaconda/mpi-wrapper/lib64/liblgcore_mpi_wrapper.so


    --------------------- CONDA/GASNET_WRAPPER/ACTIVATE.SH -----------------------

    REALM_GASNETEX_WRAPPER=/conda/envs/legate-gex-anaconda/gex-wrapper/lib/librealm_gex_wrapper.so
    GASNET_OFI_SPAWNER=mpi
    FI_CXI_RDZV_THRESHOLD=256
    START

    --- Legion Python Configuration ------------------------------------------------

    Legate paths:
      legate_dir       : /conda/envs/legate-gex-anaconda/lib/python3.12/site-packages/legate
      legate_build_dir : None
      bind_sh_path     : /conda/envs/legate-gex-anaconda/share/legate/libexec/legate-bind.sh
      legate_lib_path  : /conda/envs/legate-gex-anaconda/lib

    Legion paths:
      legion_bin_path       : /conda/envs/legate-gex-anaconda/bin
      legion_lib_path       : /conda/envs/legate-gex-anaconda/lib
      realm_defines_h       : /conda/envs/legate-gex-anaconda/include/realm_defines.h
      legion_defines_h      : /conda/envs/legate-gex-anaconda/include/legion_defines.h
      legion_prof           : /conda/envs/legate-gex-anaconda/bin/legion_prof
      legion_module         : /conda/envs/legate-gex-anaconda/lib/python3.12/site-packages
      legion_jupyter_module : /conda/envs/legate-gex-anaconda/lib/python3.12/site-packages

    Versions:
      legate_version : 24.9.0.dev329+g32137a65

    Command:
      srun -n 2 --ntasks-per-node 1 /conda/envs/legate-gex-anaconda/share/legate/libexec/legate-bind.sh --launcher srun -- python ./hello_world.py

    Customized Environment:
      CUTENSOR_LOG_LEVEL=1
      GASNET_MPI_THREAD=MPI_THREAD_MULTIPLE
      LEGATE_CONFIG='--cpus 1 --gpus 4 --sysmem 4000 --fbmem 4000 --logging legate=1,gex=0 --logdir /log/dir --eager-alloc-percentage 1'
      LEGATE_MAX_DIM=4
      LEGATE_MAX_FIELDS=256
      NCCL_LAUNCH_MODE=PARALLEL
      PYTHONDONTWRITEBYTECODE=1
      PYTHONPATH=/opt/nersc/pymon:/conda/envs/legate-gex-anaconda/lib/python3.12/site-packages:/conda/envs/legate-gex-anaconda/lib/python3.12/site-packages
      REALM_BACKTRACE=1

    --------------------------------------------------------------------------------

    Hello World
    Hello World
