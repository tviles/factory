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
import threading
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


# Fixed namespace so an sssf session id always maps to the same Claude Code
# uuid. Deterministic beats a stored uuid4: it is reproducible even if
# agent_map.json is lost, and an --adw-id rejoin needs no extra state. Also
# recorded directly in agent_start's payload and agent_map.json (spec §1b) —
# the containment regression §1b accepts (transcripts live outside data_dir)
# was explicitly traded for recording where they went, not for making the
# operator re-derive this uuid by hand.
CC_NAMESPACE = uuid.UUID("6f1f5b1e-3d0a-5e7c-9a2b-7c4d8e0f1a23")


def cc_session_uuid(sssf_session_id: str) -> str:
    """sssf-<adw_id>-<agent>-<rand4> -> a stable uuid `--session-id` accepts."""
    return str(uuid.uuid5(CC_NAMESPACE, sssf_session_id))


# `projectsDirectory` from the last `claude auth status` reading (see
# preflight_auth/remember_projects_directory), so execute() can record where
# THIS process's transcripts land without shelling out again — validate()
# already runs the CLI once for the whole roster (spec §7a); this is what
# lets every claude_code agent's agent_start payload and agent_map.json entry
# know it too (spec §1b's mitigation for the containment regression).
_preflight_projects_directory: str = ""


def remember_projects_directory(auth_status: dict) -> None:
    """Cache `projectsDirectory` out of a `claude auth status` reading.

    Called from agents.validate() with whatever `preflight_auth()` (real or,
    in a test, monkeypatched) returned — validate() previously parsed
    `projectsDirectory` out of this dict and then threw it away.
    """
    global _preflight_projects_directory
    _preflight_projects_directory = (auth_status or {}).get("projectsDirectory") or ""


def projects_directory() -> str:
    """The transcript root the last validate()-time preflight observed.

    '' when no claude_code preflight has run yet in this process (a pi-only
    chain, or a claude_code send issued before validate()) — callers must
    treat that as "unknown", not fail.
    """
    return _preflight_projects_directory


def parse_auth_status(raw: str, inherit_api_key: bool) -> dict:
    """Validate `claude auth status` output. Raises ValueError with the fix.

    Checking `init.apiKeySource == "none"` alone is NOT enough: "none" means
    "no API key in use", and an unauthenticated session reports it too. This
    is the positive check — that a usable credential exists AND that it is the
    subscription rather than an API key.
    """
    try:
        info = json.loads(raw)
    except json.JSONDecodeError as error:
        raise ValueError(f"could not parse `claude auth status` output: {error}"
                         f"\n{raw[:400]}") from error
    if not info.get("loggedIn"):
        raise ValueError("Claude Code is not logged in — run `claude auth login`. "
                         "(`claude auth status` reports loggedIn: false)")
    if not inherit_api_key:
        # `claude auth status` OMITS apiKeySource entirely when no API key is
        # in use, so a falsy check is correct for THIS surface. The stream's
        # `init` event instead reports the literal string "none" for the same
        # state — if the two ever converge, a bare truthy check would reject
        # every legitimate subscription user. Treat "none" as absent too;
        # do not "simplify" this back to one check across both surfaces.
        if info.get("apiKeySource") and info.get("apiKeySource") != "none":
            raise ValueError(
                f"Claude Code would bill the API, not your subscription: "
                f"apiKeySource={info['apiKeySource']!r}. Unset it, or set "
                f"defaults.claude_code.inherit_api_key: true to allow it.")
        if info.get("authMethod") != "claude.ai":
            raise ValueError(
                f"expected subscription auth (authMethod 'claude.ai'), got "
                f"{info.get('authMethod')!r}. Set "
                f"defaults.claude_code.inherit_api_key: true to allow it.")
    # subscriptionType is RECORDED, never gated on — it is an Anthropic-
    # controlled plan name, and requiring a value would fail a Pro seat, a
    # Team/Enterprise seat, or any future rename.
    return info


def preflight_auth(inherit_api_key: bool = False) -> dict:
    """Run `claude auth status` under the stripped child env. No model call.

    Belongs in validate(): hard rule 1 says nothing spawns against a
    half-valid config, and an unauthenticated CLI is exactly that.
    """
    try:
        result = subprocess.run(
            [CLAUDE_PATH, "auth", "status"], capture_output=True, text=True,
            timeout=30, stdin=subprocess.DEVNULL,
            env=claude_code_env(inherit_api_key), check=False)
    except FileNotFoundError as error:
        raise ValueError(f"the `claude` CLI was not found on PATH "
                         f"(CLAUDE_PATH={CLAUDE_PATH!r})") from error
    except subprocess.TimeoutExpired as error:
        raise ValueError("`claude auth status` timed out after 30s") from error
    if result.returncode != 0:
        raise ValueError(f"`claude auth status` exited {result.returncode}: "
                         f"{result.stderr.strip()[:400]}")
    return parse_auth_status(result.stdout, inherit_api_key)


class CodingAgentError(RuntimeError):
    """The coding agent failed in a way re-prompting cannot fix.

    `rate_limit`, when set, is the `rate_limit_info` reading that caused this
    failure. Carried on the EXCEPTION, not just left in a local variable,
    because a raise from `run()` propagates straight out of `agents.execute()`
    — `agent_end` only fires after gates pass and permissions.enforce()
    succeeds — so this is the only way the one reading most likely to trip
    `agents.py`'s headroom guard (a rejected/blocked window) can ever reach
    the trace `validate()` reads back. See `agents.py`'s `send()`.
    """

    def __init__(self, message: str, rate_limit: Optional[dict] = None) -> None:
        super().__init__(message)
        self.rate_limit: dict = rate_limit or {}


class RateLimited(CodingAgentError):
    """Subscription window exhausted. NOT a JSON problem, NOT retryable."""


class NotAuthenticated(CodingAgentError):
    """The CLI is not logged in. NOT a JSON problem."""


class OverageRefused(CodingAgentError):
    """The subscription fell through to paid overage and we refuse to spend."""


NOT_AUTH_SIGNATURES = ("not logged in", "please run /login", "invalid api key",
                       "authentication_error")


def _result_text(block_content) -> str:
    """tool_result content is `str | list[block]`; normalise both."""
    if isinstance(block_content, str):
        return block_content
    if isinstance(block_content, list):
        return "".join(part.get("text", "") for part in block_content
                       if isinstance(part, dict) and part.get("type") == "text")
    return ""


def usage_from_result(ev: dict) -> UsageBreakdown:
    """Fold the `result` event's authoritative totals into UsageBreakdown.

    `result.usage` is the total for the WHOLE send across every API call it
    made. Per-message usage must not be summed — assistant events repeat one
    message's usage object per content block.

    Per-component COSTS stay 0: Claude Code reports only `total_cost_usd`.
    They are unavailable, not zero, which is why the UI is changed to say so
    rather than render four $0.0000 rows (spec §3).
    """
    u = ev.get("usage") or {}
    details = u.get("output_tokens_details") or {}
    usage = UsageBreakdown()
    usage.input_tokens = u.get("input_tokens") or 0
    usage.output_tokens = u.get("output_tokens") or 0
    usage.cache_read_tokens = u.get("cache_read_input_tokens") or 0
    usage.cache_write_tokens = u.get("cache_creation_input_tokens") or 0
    usage.reasoning_tokens = details.get("thinking_tokens") or 0
    # Pi's convention: cache reads count — cached prompt is still prompt.
    usage.total_tokens = (usage.input_tokens + usage.output_tokens
                          + usage.cache_read_tokens + usage.cache_write_tokens)
    usage.total_cost = ev.get("total_cost_usd") or 0.0
    return usage


def context_window_from_result(ev: dict, model: str) -> int:
    """The model's ceiling, straight from modelUsage. 0 = unknown."""
    entries = ev.get("modelUsage") or {}
    entry = entries.get(model) or next(iter(entries.values()), {})
    return int(entry.get("contextWindow") or 0)


def _raise_if_overage(rl: dict, on_overage: str) -> None:
    """The `OverageRefused` half of `classify()`, pulled out so the stream
    loop in `_run_once` can call it too.

    Spec §8: the adapter must raise the MOMENT it observes
    `isUsingOverage: true`, not after the whole send finishes — a send that
    flips into paid overage at turn 2 must not be left to run to completion
    (up to `timeout_seconds`) before refusing. `_run_once` calls this right
    where `rate_limit_event` is observed, which is the primary, EARLY check;
    `classify()` below also calls it as a backstop for a result event whose
    rate-limit info a future CLI surfaces some other way — by the time
    `classify()` runs, this has almost always already fired from the loop.
    """
    # Fail CLOSED: `on_overage` is typed Literal["fail","warn"] upstream in
    # ClaudeCodeDefaults, but CodingAgentRequest.on_overage is a plain str, so
    # a direct construction bypasses that check. This guard's only job is
    # refusing to spend money the user did not opt into, so an unrecognised
    # value (a typo, a future default) must still refuse — not silently pass.
    if rl.get("isUsingOverage") and on_overage != "warn":
        raise OverageRefused(
            f"the subscription is using PAID overage "
            f"(utilization={rl.get('utilization')}). Refusing to spend. Set "
            f"defaults.claude_code.on_overage: warn to allow it, or disable "
            f"extra usage in your Anthropic account settings.", rate_limit=rl)


def classify(ev: dict, last_rate_limit: Optional[dict], on_overage: str) -> None:
    """Raise if this `result` event is a failure re-prompting cannot fix.

    Ordered by specificity, and checked BEFORE the envelope text ever reaches
    agents._extract_json. That ordering is the whole point: a rate-limit or
    not-logged-in message is perfectly good TEXT, so left to fall through it
    parses as "bad JSON", burns both correction sends against the same wall,
    and kills the run reporting a prompt-engineering problem.
    """
    rl = last_rate_limit or {}
    _raise_if_overage(rl, on_overage)
    if rl.get("status") in ("rejected", "blocked"):
        raise RateLimited(
            f"Claude Code rate limit reached: {rl.get('rateLimitType')} window "
            f"at utilization={rl.get('utilization')}, resets at "
            f"{rl.get('resetsAt')}. Not retried — a seven_day reset can be days "
            f"away, and falling back to a per-token provider is not automatic.",
            rate_limit=rl)
    if not ev.get("is_error"):
        return
    text = str(ev.get("result") or "").lower()
    if any(sig in text for sig in NOT_AUTH_SIGNATURES):
        raise NotAuthenticated(
            f"Claude Code is not authenticated: {ev.get('result')!r}. "
            f"Run `claude auth status` to check, `claude auth login` to fix.")
    raise CodingAgentError(
        f"claude exited with is_error=true "
        f"(subtype={ev.get('subtype')!r}, terminal_reason={ev.get('terminal_reason')!r}, "
        f"api_error_status={ev.get('api_error_status')!r}): "
        f"{str(ev.get('result'))[:500]}")


class ToolCallTracker:
    """Folds Claude Code's stream into ONE normalized record per tool call.

    A call appears as a `tool_use` block on an assistant message and closes as
    a `tool_result` block on a user message, joined by id. Only the result
    carries the outcome, so that is where a record is emitted — one trace event
    per real tool call, the moment it returns.

    Also tracks context occupancy as a side effect, because the only place the
    per-turn usage appears is on the assistant messages this already walks.

    `observe` returns a LIST, not `Optional[dict]`: Claude Code can batch
    parallel tool calls into one `user` event carrying several `tool_result`
    blocks, and a single-value return can only ever surface the first one —
    every later block in that event would be silently dropped from the trace
    AND leak its `_open` entry forever (review Task 6 I-2). `agent_pi`'s
    tracker shares this contract; its results really are one-at-a-time, so it
    just wraps its single record as `[record]`.
    """

    def __init__(self) -> None:
        self._open: dict[str, dict] = {}
        self._seen_messages: set[str] = set()
        self.context_tokens = 0

    def observe(self, event: dict) -> list[dict]:
        etype = event.get("type")
        if etype == "assistant":
            self._on_assistant(event)
            return []
        if etype == "user":
            return self._on_user(event)
        return []

    def _on_assistant(self, event: dict) -> None:
        message = event.get("message") or {}
        for block in message.get("content") or []:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                self._announce(block.get("id"), block.get("name"),
                               block.get("input") or {})
        # Occupancy, deduped: every content block of one message re-reports the
        # SAME usage object, so keying on message id is what stops a
        # three-block message counting its tokens three times.
        message_id = message.get("id")
        usage = message.get("usage") or {}
        if not message_id or message_id in self._seen_messages or not usage:
            return None
        self._seen_messages.add(message_id)
        if message.get("stop_reason") in ("error", "aborted"):
            return None              # an errored turn reports usage you can't trust
        self.context_tokens = ((usage.get("input_tokens") or 0)
                               + (usage.get("cache_creation_input_tokens") or 0)
                               + (usage.get("cache_read_input_tokens") or 0)
                               + (usage.get("output_tokens") or 0))
        return None

    def _on_user(self, event: dict) -> list[dict]:
        # A list, not "return on first match": Claude Code CAN close several
        # parallel tool calls in ONE user event (one tool_result block each),
        # and returning early left every block after the first neither
        # emitted nor popped from `self._open` — a silently dropped trace row
        # AND a leaked open-call entry for the life of the run.
        records = []
        for block in (event.get("message") or {}).get("content") or []:
            if not isinstance(block, dict) or block.get("type") != "tool_result":
                continue
            call_id = str(block.get("tool_use_id") or "")
            opened = self._open.pop(call_id, {})
            tool = str(opened.get("tool") or "tool")
            args = opened.get("args") or {}
            record = {
                "tool": tool,
                "tool_call_id": call_id,
                "args": {k: clip(v, ARG_VALUE_CHARS) if isinstance(v, str) else v
                         for k, v in args.items()},
                "ok": not block.get("is_error", False),
                "label": tool_label(tool, args),
                "result_snippet": clip(_result_text(block.get("content")),
                                       RESULT_SNIPPET_CHARS),
                "ended_at": now_iso(),
                "started_at": opened.get("started_at") or now_iso(),
                "duration_ms": int((time.monotonic() - opened["clock"]) * 1000)
                               if opened.get("clock") else 0,
            }
            if event.get("parent_tool_use_id"):
                record["parent_tool_use_id"] = event["parent_tool_use_id"]
            records.append(record)
        return records

    def _announce(self, call_id, tool, args) -> None:
        if not call_id:
            return
        self._open[str(call_id)] = {
            "tool": tool or "tool", "args": args or {},
            "started_at": now_iso(),          # wall clock, for the row
            "clock": time.monotonic(),        # monotonic, for duration
        }


# Linux caps a SINGLE argv entry at MAX_ARG_STRLEN (128KB) regardless of how
# much ARG_MAX headroom there is. Stay under it with room to spare; above this
# the prompt goes to stdin instead (spec §1b).
ARGV_SPILL_THRESHOLD = 96_000


def build_command(request: CodingAgentRequest) -> tuple[list[str], list[str]]:
    """Returns (argv, warnings). Pure — no subprocess, no filesystem."""
    model = resolve_model(request.model)
    effort, effort_warning = map_effort(request.thinking)
    tools, warnings = map_tools(request.tools)
    if effort_warning:
        warnings.append(effort_warning)

    cmd = [CLAUDE_PATH, "-p",
           "--output-format", "stream-json",
           "--verbose",                       # mandatory with stream-json
           "--model", model,
           "--effort", effort]

    # --session-id CREATES and fails if the id exists; --resume continues.
    # Passing both is a CLI error without --fork-session.
    session_uuid = cc_session_uuid(request.session_id)
    cmd += ["--resume", session_uuid] if request.resume else ["--session-id", session_uuid]

    if request.system_prompt_path:
        cmd += ["--append-system-prompt-file", request.system_prompt_path]
    else:
        cmd += ["--append-system-prompt", request.system_prompt]

    # --tools decides what EXISTS; --allowedTools decides what needs no
    # approval. Comma-joined into ONE argument each: both flags are variadic,
    # so space-separated values before the positional prompt let the parser
    # swallow the prompt as another tool name.
    joined = ",".join(tools)
    cmd += ["--tools", joined, "--allowedTools", joined]

    # Non-blocking without being reckless: edits proceed, granted tools are
    # pre-approved, and anything else that WOULD prompt is denied rather than
    # hanging a headless run forever.
    cmd += ["--permission-mode", "acceptEdits", "--permission-prompts", "none"]

    # The agent runs with cwd at the repo root (permissions.enforce and session
    # resumption both require it), so the repo's own CLAUDE.md, hooks, skills
    # and MCP servers are in reach. Isolate the CONFIGURATION, not the cwd.
    # NOT --bare: it reads auth strictly from ANTHROPIC_API_KEY.
    cmd += ["--setting-sources", "", "--strict-mcp-config", "--disable-slash-commands"]

    if request.restricted:
        cmd.append("--restricted")

    if len(request.prompt) > ARGV_SPILL_THRESHOLD:
        cmd += ["--input-format", "stream-json"]
    else:
        cmd.append(request.prompt)
    return cmd, warnings


def _assert_subscription_auth(init_event: dict, inherit_api_key: bool) -> None:
    """Cross-check, per send, that the CHILD is billing what the preflight predicted.

    `preflight_auth` runs once in validate(); this fires on every run, and it
    is the only per-run verification of the feature's core claim. Between the
    two, a stale shell, a hook, or a bug in `claude_code_env` could still put a
    key in front of the child.

    NOTE the value semantics differ between the two surfaces, and the
    difference is load-bearing — do not "unify" these two checks:

      * `claude auth status` OMITS `apiKeySource` entirely when no key is in
        use, so `parse_auth_status` truthy-checks it.
      * the stream's `init` event reports the STRING `"none"` for subscription
        auth, so this one compares against that string. A truthy check here
        would reject every legitimate subscription run.
    """
    if inherit_api_key:
        return
    source = init_event.get("apiKeySource")
    if source not in (None, "none"):
        raise NotAuthenticated(
            f"Claude Code is billing {source!r}, not your subscription, despite "
            f"the startup preflight passing. Something put a key in front of "
            f"the child after validate() ran. Set "
            f"defaults.claude_code.inherit_api_key: true to allow it.")


def _stderr_tail(stderr_path: Path, offset: int) -> str:
    """This attempt's last 800 bytes of stderr, for a failure message."""
    return stderr_path.read_bytes()[offset:][-800:].decode(errors="replace").strip()


def _run_once(request: CodingAgentRequest,
              on_event: Optional[Callable[[dict], None]] = None,
              on_spawn: Optional[Callable[[int], None]] = None,
              on_exit: Optional[Callable[[int], None]] = None) -> CodingAgentResult:
    """Run one non-interactive Claude Code turn.

    Same contract as agent_pi.run(): stream events to on_event as they happen,
    bracket the child with on_spawn/on_exit so a hung agent is a pid the trace
    can name, and return the shape agents.execute() already consumes.
    """
    cmd, warnings = build_command(request)
    model = resolve_model(request.model)

    raw_path = Path(request.raw_output_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    stderr_path = Path(request.stderr_path or raw_path.with_name("stderr.log"))
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    # One log per agent-phase, appended to — but this send reads only ITS OWN
    # bytes. `send()` is called repeatedly for one agent by design (JSON
    # retries, gate corrections; see agents.py:105), so scanning the whole file
    # would report a previous attempt's warnings and could show a previous
    # attempt's crash tail. Same defect Task 2 fixed in agent_pi.py; do not
    # reintroduce it here.
    stderr_offset = stderr_path.stat().st_size if stderr_path.exists() else 0

    result = CodingAgentResult(session_id=request.session_id, cost_basis="list")
    tracker = ToolCallTracker()
    last_rate_limit: Optional[dict] = None
    result_event: Optional[dict] = None
    saw_init_event = False
    spill = len(request.prompt) > ARGV_SPILL_THRESHOLD

    # stderr to a FILE, never a second pipe: with both piped and a blocking
    # stdout read, a chatty child deadlocks both sides (see agent_pi.py).
    with stderr_path.open("a") as err:
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE if spill else subprocess.DEVNULL,
            stdout=subprocess.PIPE, stderr=err, text=True, bufsize=1,
            cwd=request.cwd, env=claude_code_env(request.inherit_api_key),
            start_new_session=True)      # own process group, so kill_tree works
    if on_spawn:
        on_spawn(process.pid)

    # A watchdog, not the in-loop check this replaced: the earlier version
    # only compared the clock against a deadline WHILE handling a line, so a
    # child that produced zero output (this module's own motivating incident
    # — "sat idle at 0% CPU with an empty raw_output.jsonl") blocked forever
    # in `for line in process.stdout` and the check never ran (review
    # Important #1). A timer fires independently of whether the child ever
    # writes; killing it closes the pipe, the blocking read returns EOF, and
    # the loop ends on its own.
    timed_out = threading.Event()
    watchdog: Optional[threading.Timer] = None
    if request.timeout_seconds:
        def _on_timeout() -> None:
            timed_out.set()
            kill_tree(process.pid)
        watchdog = threading.Timer(request.timeout_seconds, _on_timeout)
        watchdog.daemon = True
        watchdog.start()

    try:
        if spill:
            # The prompt was too large for argv, so it travels as one
            # stream-json user message. Closing stdin is what ends the turn.
            # Inside `try` on purpose (review Important #2): a
            # BrokenPipeError here — the ordinary outcome when the child
            # rejects a flag and exits before ever reading stdin — used to
            # escape with on_spawn already fired but no kill_tree, no wait(),
            # and no on_exit, leaking the pid into Task 8's live_children
            # forever.
            assert process.stdin is not None
            try:
                process.stdin.write(json.dumps({
                    "type": "user",
                    "message": {"role": "user",
                                "content": [{"type": "text", "text": request.prompt}]},
                }) + "\n")
                process.stdin.close()
            except (BrokenPipeError, OSError) as error:
                raise CodingAgentError(
                    f"failed writing the spilled prompt to claude's stdin: "
                    f"{error}") from error

        with raw_path.open("a") as raw:
            assert process.stdout is not None
            for line in process.stdout:
                raw.write(line)
                raw.flush()                  # events land on disk as they happen
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("subtype") == "init":
                    saw_init_event = True
                    _assert_subscription_auth(event, request.inherit_api_key)
                elif event.get("type") == "rate_limit_event":
                    last_rate_limit = event.get("rate_limit_info") or {}
                    # The moment it is observed (spec §8), not end-of-send: a
                    # send that flips into paid overage at turn 2 must not run
                    # to completion first. Raising here lands inside this
                    # try/except/finally, so it still goes through the same
                    # kill_tree / process.wait() / on_exit path as any other
                    # BaseException from this loop.
                    _raise_if_overage(last_rate_limit, request.on_overage)
                elif event.get("type") == "result":
                    result_event = event
                tracker.observe(event)
                if on_event:
                    on_event(event)
    except BaseException:
        kill_tree(process.pid)
        raise
    finally:
        if watchdog:
            watchdog.cancel()
        if process.poll() is None:
            kill_tree(process.pid)
        result.returncode = process.wait()
        # process.stdout is the read end of the PIPE opened above. Popen never
        # closes it for us, and nothing else in this function does either —
        # confirmed by `-W error::ResourceWarning` reporting an unclosed-pipe
        # warning per call before this line existed. Same reasoning as
        # agent_pi.run()'s matching close: send() re-enters for one
        # agent-phase (parse-fix and gate-correction retries), so a chain of
        # sends in one ADW process accumulates leaked fds instead of each
        # being reclaimed by GC soon after. Closed HERE, after wait() so the
        # child is already reaped, and guarded so a redundant/already-closed
        # pipe cannot itself raise and mask the real exception this `finally`
        # may be unwinding.
        try:
            if process.stdout is not None:
                process.stdout.close()
        except OSError:
            pass
        # The spill path (above) opens stdin as a SECOND pipe and closes it
        # itself right after the write — but only on the write's happy path.
        # A BrokenPipeError there is caught and re-raised as CodingAgentError
        # BEFORE that close() runs, so that write end leaked too: verified by
        # `-W error::ResourceWarning` still reporting one unclosed pipe on
        # test_run_still_fires_on_exit_when_the_spill_write_breaks_the_pipe
        # even after the stdout close above was added. Same guard, same
        # reasoning — idempotent against the already-closed happy-path case.
        try:
            if process.stdin is not None:
                process.stdin.close()
        except OSError:
            pass
        if on_exit:
            on_exit(process.pid)

    if timed_out.is_set():
        raise CodingAgentError(
            f"claude exceeded timeout_seconds="
            f"{request.timeout_seconds} and was killed")

    result.warnings = warnings + stderr_warnings(stderr_path, offset=stderr_offset)
    if not saw_init_event:
        # `_assert_subscription_auth` is the only PER-RUN verification of the
        # "never bills the API key" claim; `preflight_auth` only covers the
        # static validate()-time case. A CLI that stops emitting `init`,
        # renames the field, or moves it would otherwise skip this check by
        # never running it — fail open silently. Surface it instead of hard
        # failing: a legitimate run against a future CLI should not die
        # because the check could not be performed, but it must be loud
        # about not having been performed (review Minor #6).
        result.warnings.append(
            "claude never sent an init event — the per-run subscription-"
            "billing check could not run; only the static validate()-time "
            "check (preflight_auth) covered this send.")

    if result_event is None:
        raise CodingAgentError(
            f"claude exited {result.returncode} without a result event: "
            f"{_stderr_tail(stderr_path, stderr_offset)}")

    # Classify BEFORE handing any text upstream: a rate limit or a logged-out
    # CLI is perfectly good text, and left to fall through it would parse as
    # "bad JSON" and burn the correction budget against the same wall.
    classify(result_event, last_rate_limit, request.on_overage)

    result.text = str(result_event.get("result") or "")
    result.usage = usage_from_result(result_event)
    result.tokens = result.usage.total_tokens
    result.cost = result.usage.total_cost
    result.context_tokens = tracker.context_tokens
    result.context_window = context_window_from_result(result_event, model)
    if last_rate_limit:
        result.rate_limit = last_rate_limit
    # `--permission-prompts none` (build_command) converts every would-be
    # prompt into a SILENT automatic denial — this is the only record of what
    # got refused. Surfaced as a warning too (not just the payload field) so
    # an operator watching a live run sees it the same way they see an
    # --effort clamp, rather than just debugging their prompt (spec §7c).
    result.permission_denials = result_event.get("permission_denials") or []
    if result.permission_denials:
        result.warnings.append(
            f"claude denied {len(result.permission_denials)} permission "
            f"prompt(s) under --permission-prompts none: "
            f"{result.permission_denials}")
    if result.returncode != 0 and not result.text:
        # Same tail as the "no result event" raise above (review Minor #4):
        # without it the retry wrapper cannot match a session signature on
        # this branch, and the operator got "claude exited 1" with
        # result.warnings computed and then thrown away with the exception.
        tail = _stderr_tail(stderr_path, stderr_offset)
        detail = f": {tail}" if tail else ""
        if result.warnings:
            detail += f" (warnings: {'; '.join(result.warnings)})"
        raise CodingAgentError(f"claude exited {result.returncode}{detail}")
    return result


SESSION_EXISTS = "already in use"
SESSION_MISSING = ("no conversation found", "session not found")


def run(request: CodingAgentRequest,
        on_event: Optional[Callable[[dict], None]] = None,
        on_spawn: Optional[Callable[[int], None]] = None,
        on_exit: Optional[Callable[[int], None]] = None) -> CodingAgentResult:
    """One turn, with a single create/resume correction.

    Our belief about whether the session exists comes from agent_map.json,
    which is only written after a phase SUCCEEDS — so a phase that created a
    session and then failed leaves the map saying "new" about an id Claude
    Code already knows, and a cleared ~/.claude leaves it saying "existing"
    about one that is gone. Both are one retry, never a loop.
    """
    try:
        return _run_once(request, on_event, on_spawn, on_exit)
    except CodingAgentError as error:
        text = str(error).lower()
        if request.resume and any(sig in text for sig in SESSION_MISSING):
            flipped = request.model_copy(update={"resume": False})
        elif not request.resume and SESSION_EXISTS in text:
            flipped = request.model_copy(update={"resume": True})
        else:
            raise
        return _run_once(flipped, on_event, on_spawn, on_exit)
