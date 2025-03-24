#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0
set -eou pipefail

run_make_check()
{
  CMAKE="${CMAKE:-cmake}"
  PYTHON="${PYTHON:-python3}"

  local check_name
  check_name="$1"

  shift

  local -a check_args
  check_args=(--color --debug --timeout=20)
  check_args+=("${@}")

  local log_file
  # We set -u above, so we want this to fail if undefined
  # shellcheck disable=SC2154
  log_file="${LEGATE_DIR}/${LEGATE_ARCH}/make_check.log"

  ${CMAKE} -E rm -f -- "${log_file}"
  ${CMAKE} -E echo_append "Running ${check_name} check..."

  set +e
  ${PYTHON} "${LEGATE_DIR}/test.py" "${check_args[@]}" > "${log_file}"
  rc=$?
  set -e

  if [[ ${rc} == 0 ]]; then
    ${CMAKE} -E echo 'success'
  else
    ${CMAKE} -E echo '' # to get a newline
    ${CMAKE} -E cat -- "${log_file}"
    ${CMAKE} -E false # force a failure
  fi
}

run_make_check "$@"
