#!/usr/bin/env bash
echo -e "\n\n--------------------- CONDA/GASNET_WRAPPER/DEACTIVATE.SH -----------------------\n"

set -eox pipefail
unset REALM_GASNETEX_WRAPPER
unset GASNET_OFI_SPAWNER
unset FI_CXI_RDZV_THRESHOLD
set +x
