#!/usr/bin/env bash
echo -e "\n\n--------------------- CONDA/GASNET_WRAPPER/PREUNLINK.SH -----------------------\n"

set -eo pipefail
echo "Remove the built artifacts"
rm -rf "${CONDA_PREFIX}/gex-wrapper/lib*"
