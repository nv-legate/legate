#!/usr/bin/env bash

# Copyright (c) 2021-2023, NVIDIA CORPORATION.
# This script is a wrapper for cmakelang that may be used with pre-commit. The
# wrapping is necessary because RAPIDS libraries split configuration for
# cmakelang linters between a local config file and a second config file that's
# shared across all of RAPIDS via rapids-cmake. In order to keep it up to date
# this file is only maintained in one place (the rapids-cmake repo) and
# pulled down during builds. We need a way to invoke CMake linting commands
# without causing pre-commit failures (which could block local commits or CI),
# while also being sufficiently flexible to allow users to maintain the config
# file independently of a build directory.
#
# This script can be invoked directly anywhere within the project repository.
# Alternatively, it may be invoked as a pre-commit hook via
# `pre-commit run (cmake-format)|(cmake-lint)`.
#
# Usage:
# bash run-cmake-format.sh {cmake-format,cmake-lint} infile [infile ...]
set -ou pipefail

if [[ "${LEGATE_CORE_CMAKE_FORMAT_FILE:-}" == '' ]]; then
  LEGATE_CORE_CMAKE_FORMAT_FILE="${LEGATE_CORE_DIR}/scripts/pre-commit/cmake-format-legate-core.json"
fi

retcode=0
case "${1}" in
  'cmake-format')
    # We cannot pass multiple input files because of a bug in cmake-format. See:
    # https://github.com/cheshirekow/cmake_format/issues/284
    for cmake_file in "${@:2}"; do
      cmake-format \
        --in-place \
        --first-comment-is-literal \
        --config-files \
          "${LEGATE_CORE_CMAKE_FORMAT_FILE}" \
          "${LEGATE_CORE_DIR}/scripts/pre-commit/cmake_config_format.json" \
        -- \
        "${cmake_file}"
      status=$?

      # Keep looping on error, we want to collect as many errors in the output as possible
      if [[ "${status}" != '0' ]]; then
        retcode=${status}
      fi
    done
    ;;
  'cmake-lint')
    # Since the pre-commit hook is verbose, we have to be careful to only
    # present cmake-lint's output (which is quite verbose) if we actually
    # observe a failure.
    # shellcheck disable=SC2068
    output=$(cmake-lint \
               --config-files \
                 "${LEGATE_CORE_CMAKE_FORMAT_FILE}" \
                 "${LEGATE_CORE_DIR}/scripts/pre-commit/cmake_config_format.json" \
                 "${LEGATE_CORE_DIR}/scripts/pre-commit/cmake_config_lint.json" \
               -- \
               ${@:2})
    retcode=$?

    if [[ "${retcode}" != '0' ]]; then
      echo "ERROR"
      echo "${output}"
    fi
    ;;
  *)
    echo "Unknown command: ${1}"
    retcode=1
    ;;
esac
exit ${retcode}
