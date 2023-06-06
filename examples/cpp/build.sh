#!/bin/bash

rm -rf build
legate_root=`python -c 'import legate.install_info as i; from pathlib import Path; print(Path(i.libpath).parent.resolve())'`
cunumeric_root=`python -c 'import cunumeric.install_info as i; from pathlib import Path; print(Path(i.libpath).parent.resolve())'`
echo "Using Legate at $legate_root"
echo "Using cuNumeric at $cunumeric_root"
cmake -B build -S . -D cunumeric_DIR=$cunumeric_root -D legate_core_ROOT=$legate_root -D CMAKE_BUILD_TYPE=Debug
cmake --build build -j 8
