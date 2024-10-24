#!/usr/bin/env bash

echo -e "\n\n--------------------- CONDA/LEGATE-PROFILER/BUILD.SH -----------------------\n"

set -xeo pipefail

# Build the legate profiler
echo "Building legate-profiler..."
# shellcheck disable=SC2154
GIT_COMMIT=$(git -C "${SRC_DIR}" rev-parse HEAD)
echo "Legion checked-out with commit: ${GIT_COMMIT}"

# Navigate to the legate-profiler directory and build it
# shellcheck disable=SC2154
LIBCLANG_PATH=${BUILD_PREFIX}/lib cargo install --debug --path "${SRC_DIR}"/tools/legion_prof_rs --all-features --root "${PREFIX}"
echo "Done"
