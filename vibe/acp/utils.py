from __future__ import annotations

from enum import StrEnum
from typing import Literal, cast

from acp.schema import PermissionOption


class ToolOption(StrEnum):
    ALLOW_ONCE = "allow_once"
    ALLOW_ALWAYS = "allow_always"
    REJECT_ONCE = "reject_once"
    REJECT_ALWAYS = "reject_always"


TOOL_OPTIONS = [
    PermissionOption(
        optionId=ToolOption.ALLOW_ONCE,
        name="Allow once",
        kind=cast(Literal["allow_once"], ToolOption.ALLOW_ONCE),
    ),
    PermissionOption(
        optionId=ToolOption.ALLOW_ALWAYS,
        name="Allow always",
        kind=cast(Literal["allow_always"], ToolOption.ALLOW_ALWAYS),
    ),
    PermissionOption(
        optionId=ToolOption.REJECT_ONCE,
        name="Reject once",
        kind=cast(Literal["reject_once"], ToolOption.REJECT_ONCE),
    ),
]
