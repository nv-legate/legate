#!/bin/bash

cd build
REALM_BACKTRACE=1 LEGATE_TEST=1 LEGION_DEFAULT_ARGS="-ll:cpu 4" ctest --output-on-failure "$@"
