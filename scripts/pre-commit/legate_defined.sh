#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TODO: The [^D] at the end of LEGATE_ is to not match LEGATE_DEFINED() itself. I would
# have used negative lookahead: LEGATE_(?!DEFINED), but grep does not support this out of
# the box. This requires PCRE2 regex (via -P) flag, but that is not supported on all
# platforms.
#
# Long term, this should really be rewritten in Python, or maybe awk.
output=$(
  grep -E \
       -n \
       -H \
       -C 1 \
       --color=always \
       -e '#\s*if[n]?def\s+LEGATE_[^D]\w+' \
       -e '#(\s*if\s+)?[!]?defined\s*\(\s*LEGATE_[^D]\w+' \
       -e '#.*defined\s*\(\s*LEGATE_[^D]\w+' \
       -e '#\s*elif\s+LEGATE_[^D]\w+' \
       -- \
       "$@"
      )
rc=$?
if [[ ${rc} -eq 1 ]]; then
  # no matches found, that's a good thing
  exit 0
elif [[ ${rc} -eq 0 ]]; then
  echo "x ===------------------------------------------------------------------=== x"
  echo "${output}"
  echo ""
  echo "Instances of preprocessor ifdef/ifndef/if defined found, use"
  echo "LEGATE_DEFINED() instead:"
  echo ""
  echo "- #ifdef LEGATE_USE_FOO"
  echo "- #include \"foo.h\""
  echo "- #endif"
  echo "+ #if LEGATE_DEFINED(LEGATE_USE_FOO)"
  echo "+ #include \"foo.h\""
  echo "+ #endif"
  echo ""
  echo "- #ifdef LEGATE_USE_FOO"
  echo "- x = 2;"
  echo "- #endif"
  echo "+ if (LEGATE_DEFINED(LEGATE_USE_FOO)) {"
  echo "+   x = 2;"
  echo "+ }"
  echo "x ===------------------------------------------------------------------=== x"
  exit 1
else
  exit "${rc}"
fi
