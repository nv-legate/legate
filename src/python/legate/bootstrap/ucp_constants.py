# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from enum import Enum
from typing import Final

# Environment variable constants
WORKER_PEERS_INFO: Final[str] = "WORKER_PEERS_INFO"
WORKER_SELF_INFO: Final[str] = "WORKER_SELF_INFO"
BOOTSTRAP_P2P_PLUGIN: Final[str] = "BOOTSTRAP_P2P_PLUGIN"
REALM_UCP_BOOTSTRAP_MODE: Final[str] = "REALM_UCP_BOOTSTRAP_MODE"


class BootstrapPluginKind(str, Enum):
    UCP = "realm_ucp_bootstrap_p2p.so"


class BootstrapMode(str, Enum):
    P2P = "p2p"
