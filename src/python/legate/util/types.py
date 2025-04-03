# SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES.
#                         All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Provide types that are useful throughout the test driver code."""

from __future__ import annotations

# ruff: noqa: A005
from dataclasses import Field, dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, TypeAlias, TypeVar

from .ui import table

if TYPE_CHECKING:
    from pathlib import Path

    from rich.table import Table

__all__ = (
    "ArgList",
    "CPUInfo",
    "Command",
    "CommandPart",
    "DataclassMixin",
    "DataclassProtocol",
    "EnvDict",
    "GPUInfo",
    "LauncherType",
    "LegatePaths",
    "RunMode",
    "object_to_dataclass",
)


@dataclass(frozen=True)
class CPUInfo:
    """Encapsulate information about a single CPU."""

    #: IDs of hypterthreading sibling cores for a given physical core
    ids: tuple[int, ...]


@dataclass(frozen=True)
class GPUInfo:
    """Encapsulate information about a single CPU."""

    #: ID of the GPU to specify in test shards
    id: int

    #: The total framebuffer memory of this GPU
    total: int


#: Define the available launcher for the driver to use
LauncherType: TypeAlias = Literal["mpirun", "jsrun", "srun", "dask", "none"]


#: Represent command line arguments
ArgList = list[str]


#: Represent str->str environment variable mappings
EnvDict: TypeAlias = dict[str, str]


#: Represent part of a environment variable to build
EnvPart: TypeAlias = tuple[str, ...]


#: Represent part of a command-line command to execute
CommandPart: TypeAlias = tuple[str, ...]


#: Represent all the parts of a command-line command to execute
Command: TypeAlias = tuple[str, ...]


#: Represent how to run the application -- as python script or binary
RunMode: TypeAlias = Literal["python", "exec"]


# This seems like it ought to be in stdlib
class DataclassProtocol(Protocol):
    """Afford better type checking for our dataclasses."""

    __dataclass_fields__: dict[str, Field[Any]]


class DataclassMixin(DataclassProtocol):
    """A mixin for automatically pretty-printing a dataclass."""

    @property
    def ui(self) -> Table:  # noqa: D102
        return table(self.__dict__, justify="left")


T = TypeVar("T", bound=DataclassProtocol)


def object_to_dataclass(obj: object, typ: type[T]) -> T:
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

    bind_sh_path: Path
    legate_lib_path: Path
