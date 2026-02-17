# SPDX-FileCopyrightText: Copyright (c) 2024-2026 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import StrEnum
from typing import Final

# Environment variable constants
WORKER_PEERS_INFO: Final[str] = "WORKER_PEERS_INFO"
WORKER_SELF_INFO: Final[str] = "WORKER_SELF_INFO"
BOOTSTRAP_P2P_PLUGIN: Final[str] = "BOOTSTRAP_P2P_PLUGIN"
REALM_UCP_BOOTSTRAP_MODE: Final[str] = "REALM_UCP_BOOTSTRAP_MODE"


class BootstrapPluginKind(StrEnum):
    UCP = "realm_ucp_bootstrap_p2p.so"


class BootstrapMode(StrEnum):
    P2P = "p2p"
