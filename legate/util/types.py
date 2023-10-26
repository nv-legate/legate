# SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Provide types that are useful throughout the test driver code.

"""
from __future__ import annotations

from dataclasses import Field, dataclass
from pathlib import Path
from typing import Any, Dict, List, Protocol, Tuple, Type, TypeVar, Union

from typing_extensions import Literal, TypeAlias

from .ui import kvtable

__all__ = (
    "ArgList",
    "Command",
    "CommandPart",
    "CPUInfo",
    "DataclassMixin",
    "DataclassProtocol",
    "EnvDict",
    "GPUInfo",
    "LauncherType",
    "LegatePaths",
    "LegionPaths",
    "RunMode",
    "object_to_dataclass",
)


@dataclass(frozen=True)
class CPUInfo:
    """Encapsulate information about a single CPU"""

    #: IDs of hypterthreading sibling cores for a given physscal core
    ids: tuple[int, ...]


@dataclass(frozen=True)
class GPUInfo:
    """Encapsulate information about a single CPU"""

    #: ID of the GPU to specify in test shards
    id: int

    #: The total framebuffer memory of this GPU
    total: int


#: Define the available launcher for the driver to use
LauncherType: TypeAlias = Union[
    Literal["mpirun"], Literal["jsrun"], Literal["srun"], Literal["none"]
]


#: Represent command line arguments
ArgList = List[str]


#: Represent str->str environment variable mappings
EnvDict: TypeAlias = Dict[str, str]


#: Represent part of a command-line command to execute
CommandPart: TypeAlias = Tuple[str, ...]


#: Represent all the parts of a command-line command to execute
Command: TypeAlias = Tuple[str, ...]


#: Represent how to run the application -- as python script or binary
RunMode: TypeAlias = Literal["python", "exec"]


# This seems like it ought to be in stdlib
class DataclassProtocol(Protocol):
    """Afford better type checking for our dataclasses."""

    __dataclass_fields__: dict[str, Field[Any]]


class DataclassMixin(DataclassProtocol):
    """A mixin for automatically pretty-printing a dataclass."""

    def __str__(self) -> str:
        return kvtable(self.__dict__)


T = TypeVar("T", bound=DataclassProtocol)


def object_to_dataclass(obj: object, typ: Type[T]) -> T:
    """Automatically generate a dataclass from an object with appropriate
    attributes.

    Parameters
    ----------
    obj: object
        An object to pull values from (e.g. an argparse Namespace)

    typ:
        A dataclass type to generate from ``obj``

    Returns
    -------
        The generated dataclass instance

    """
    kws = {name: getattr(obj, name) for name in typ.__dataclass_fields__}
    return typ(**kws)


@dataclass(frozen=True)
class LegatePaths(DataclassMixin):
    """Collect all the filesystem paths relevant for Legate."""

    legate_dir: Path
    legate_build_dir: Path | None
    bind_sh_path: Path
    legate_lib_path: Path


@dataclass(frozen=True)
class LegionPaths(DataclassMixin):
    """Collect all the filesystem paths relevant for Legate."""

    legion_bin_path: Path
    legion_lib_path: Path
    realm_defines_h: Path
    legion_defines_h: Path
    legion_spy_py: Path
    legion_python: Path
    legion_prof: Path
    legion_module: Path | None
    legion_jupyter_module: Path | None
