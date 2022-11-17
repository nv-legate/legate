#! /usr/bin/env bash

cd $(dirname "$(realpath "$0")")/..

# Use sccache if installed
source ./scripts/util/build-caching.sh
# Use consistent C[XX]FLAGS
source ./scripts/util/compiler-flags.sh
# Read Legion_ROOT from the environment or prompt the user to enter it
source ./scripts/util/read-legion-root.sh "$0"

# Remove existing build artifacts
rm -rf ./{build,_skbuild,dist,legate.core.egg-info}

# Use all but 2 threads to compile
ninja_args="-j$(nproc --ignore=2)"

# Pretend to install Legion because Legion's CMakeLists only generates the Legion CFFI bindings at install time
if [[ -f "$Legion_ROOT/CMakeCache.txt" ]]; then
(
    tmpdir=$(mktemp -d);
    cmake --build "$Legion_ROOT" ${ninja_args};
    cmake --install "$Legion_ROOT" --prefix "$tmpdir" &>/dev/null;
    rm -rf "$tmpdir";
)
fi

# Define CMake configuration arguments
cmake_args="${CMAKE_ARGS:-}"

# Use ninja-build if installed
if [[ -n "$(which ninja)" ]]; then cmake_args+="-GNinja"; fi

# Add other build options here as desired
cmake_args+="
-D CMAKE_CUDA_ARCHITECTURES=NATIVE
-D Legion_ROOT:STRING=\"$Legion_ROOT\"
";

# Configure legate_core C++
cmake -S . -B build ${cmake_args}
# Build legate_core C++
cmake --build build ${ninja_args}

cmake_args+="
-D FIND_LEGATE_CORE_CPP=ON
-D legate_core_ROOT:STRING=\"$(pwd)/build\"
"

# Build legion_core_python and perform an "editable" install
SKBUILD_BUILD_OPTIONS="$ninja_args"       \
CMAKE_ARGS="$cmake_args"                  \
SETUPTOOLS_ENABLE_FEATURES="legacy-editable" \
    python -m pip install                 \
        --root / --prefix "$CONDA_PREFIX" \
        --no-deps --no-build-isolation    \
        --editable                        \
        . -vv
