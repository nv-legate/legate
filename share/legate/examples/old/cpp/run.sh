#!/bin/bash

cd build || exit
ctest --output-on-failure
