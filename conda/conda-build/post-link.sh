#!/usr/bin/env bash

# shellcheck disable=SC2154
echo "By downloading and using the ${PKG_NAME} conda package, you accept the terms and conditions of the NVIDIA Legate Evaluation License Agreement - https://docs.nvidia.com/legate/eula.pdf" > "${PREFIX}/.messages.txt"
