#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025-2025 & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
set -eou pipefail

if [[ $# -eq 0  || "${1}" == '-h' || "${1}" == '--help' || "${1}" == '-help' ]]; then
  echo "usage: $0 file1 [file2 ... fileN]" >&2
  exit 2
fi

CUR_YEAR="$(date '+%Y')"

for filename in "${@}"; do
  if [[ -f "${filename}" ]]; then
    year="$(git log --follow --format=%ad --date='format:%Y' "${filename}" | tail -n 1)"
    if [[ "${year}" == '' ]]; then
      year="${CUR_YEAR}"
    fi
    echo "${filename} -> ${year}"
  fi
done
