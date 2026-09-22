"""Claude Code coding-agent interface.

Runs `claude -p --output-format stream-json --verbose` and tails its JSONL
stdout line by line, forwarding each event to a callback WHILE the agent works
— the same streaming contract agent_pi.py provides, so agents.execute() cannot
tell the two apart.

Two things differ from Pi and drive most of this module:

  * `--session-id` CREATES and fails if the id exists; continuing needs
    `--resume`. Pi's one flag does both. `CodingAgentRequest.resume` carries
    the distinction (spec §2).
  * `--tools` is the availability filter, `--allowedTools` is permission
    pre-approval. They are not interchangeable, and an unknown name handed to
    `--tools` is silently dropped (spec §6).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable, Optional

from .data_types import CodingAgentRequest, CodingAgentResult, UsageBreakdown
from .utils import (ARG_VALUE_CHARS, RESULT_SNIPPET_CHARS, claude_code_env,
                    clip, kill_tree, now_iso, stderr_warnings, tool_label)

CLAUDE_PATH = os.environ.get("CLAUDE_PATH", "claude")

# Claude Code's documented reasoning ladder. `off` and `minimal` are absent:
# the CLI accepts them at parse time but warns on stderr at runtime and falls
# back to its OWN default effort, which is neither what the roster asked for
# nor the lowest setting. Clamping to `low` is the closest honouring of intent
# the CLI actually offers (spec §5).
EFFORT_LEVELS = ("low", "medium", "high", "xhigh", "max")
EFFORT_CLAMP = {"off": "low", "minimal": "low"}

# Roster (Pi) tool names -> Claude Code tool names.
TOOL_MAP = {
    "read": "Read", "bash": "Bash", "edit": "Edit",
    "write": "Write", "grep": "Grep", "find": "Glob",
}
# `ls` is deliberately absent: Claude Code has no Ls tool, and mapping it to
# Bash would grant shell access to an agent that only asked to list a
# directory — exactly the boundary `writes` exists to keep.
TOOL_DROPPED = {"ls"}

# What `tools: None` resolves to. NOT "omit --tools": that yields 28 tools
# including CronCreate (schedules work outliving the run), RemoteTrigger and
# PushNotification (reach off the machine), Workflow (spawns agent fleets),
# Skill (can reach the sssf skill and recurse into the factory) and
# EnterWorktree (moves cwd, breaking BOTH permissions.enforce and session
# resumption). A roster must mean the same capability on both harnesses.
PI_EQUIVALENT_TOOLS = ["Read", "Bash", "Edit", "Write", "Grep", "Glob"]

# Exact Claude Code tool names a roster may name directly.
CC_TOOL_NAMES = {
    "Task", "Bash", "Edit", "Glob", "Grep", "NotebookEdit", "Read",
    "Skill", "ToolSearch", "WebFetch", "WebSearch", "Write",
}


def resolve_model(pattern: str) -> str:
    """`anthropic/claude-opus-5` -> `claude-opus-5`.

    Rejecting a non-anthropic provider HERE, from validate(), is the point:
    a roster that points a claude_code agent at openai/ must fail before any
    phase opens rather than mid-chain (hard rule 1).
    """
    provider, _, model_id = pattern.partition("/")
    if not model_id or provider != "anthropic":
        raise ValueError(
            f"model {pattern!r} must be written as anthropic/<model-id> when "
            f"coding_agent is claude_code (e.g. anthropic/claude-opus-5)")
    return model_id


def map_effort(thinking: str) -> tuple[str, str]:
    """Returns (effort_level, warning). An empty warning means a clean map."""
    if thinking in EFFORT_LEVELS:
        return thinking, ""
    if thinking in EFFORT_CLAMP:
        mapped = EFFORT_CLAMP[thinking]
        return mapped, (f"thinking {thinking!r} is not a Claude Code effort "
                        f"level; using {mapped!r}")
    raise ValueError(f"thinking {thinking!r} is not a known level "
                     f"(off|minimal|{'|'.join(EFFORT_LEVELS)})")


def map_tools(tools: Optional[list[str]]) -> tuple[list[str], list[str]]:
    """Returns (claude_code_tool_names, warnings).

    Raises on an unknown name rather than passing it through, because the CLI
    drops an unrecognised --tools entry SILENTLY — the run succeeds and the
    tool is simply never offered to the model.
    """
    if tools is None:
        return list(PI_EQUIVALENT_TOOLS), []
    if not tools:
        raise ValueError("tools: [] is a tool-less agent, and it will stall — "
                         "omit the key for the default set, or name the tools")
    mapped, warnings, unknown = [], [], []
    for name in tools:
        if name in TOOL_DROPPED:
            warnings.append(f"tool {name!r} has no Claude Code equivalent and "
                            f"was dropped (Read on a directory and Glob cover it)")
        elif name in TOOL_MAP:
            mapped.append(TOOL_MAP[name])
        elif name in CC_TOOL_NAMES:
            mapped.append(name)
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(
            f"tool(s) {unknown} are not Claude Code tools. --tools drops an "
            f"unknown name silently, so this would cost a run. Known roster "
            f"names: {sorted(TOOL_MAP) + sorted(TOOL_DROPPED)}; or name a "
            f"Claude Code tool exactly: {sorted(CC_TOOL_NAMES)}")
    if not mapped:
        raise ValueError("no usable tools after mapping — the agent will stall")
    return mapped, warnings
