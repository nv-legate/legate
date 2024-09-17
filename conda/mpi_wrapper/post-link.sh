#!/usr/bin/env bash
# shellcheck disable=SC2154
cat << EOF >> "${PREFIX}/.messages.txt"

To finish configuring the Legate MPI wrapper, activate your environment and run ${CONDA_PREFIX}/mpi-wrapper/build-mpi-wrapper.sh

EOF
