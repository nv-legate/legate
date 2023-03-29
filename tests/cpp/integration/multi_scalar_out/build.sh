#!/bin/bash

rm -rf build
cmake -B build -S . -D legate_core_ROOT=../../../../build -D CMAKE_BUILD_TYPE=Debug
cmake --build build -j 4
