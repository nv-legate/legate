#!/bin/bash

if [ $# -eq 0 ]
  then
    REALM_BACKTRACE=1 LEGATE_TEST=1 python run.py --cpus 4
elif [ $# -ge 1 ]  && [ "$1" = "ctest" ]
  then
    echo "Using ctest"
    cd build
    REALM_BACKTRACE=1 LEGATE_TEST=1 ctest --output-on-failure "$@"
else
    echo "Invalid arguments"
fi
