"""Pi coding agent interface.

Runs `pi -p --mode json` and tails its JSONL stdout line by line, forwarding
each event to a callback WHILE the agent works (the streaming crack, solved
by construction). `--session-id` creates-or-continues, so running and
continuing an agent are the same call: same session id = same context window.
"""

from __future__ import annotations

import json
import os
import subprocess
import time
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Callable, Optional

from .data_types import PiRequest, PiResult
from .utils import (ARG_VALUE_CHARS, LABEL_CHARS, PRIMARY_ARGS,
                    RESULT_SNIPPET_CHARS, clip as _clip, kill_tree, now_iso,
                    operator_env, stderr_warnings, tool_label as _label)

PI_PATH = os.environ.get("PI_PATH", "pi")
MODELS_JSON = os.environ.get("PI_MODELS_PATH",
                             str(Path.home() / ".pi" / "agent" / "models.json"))


def _count(value: str) -> int:
    """Parse pi's compact model-list counts (`272K`, `1.0M`)."""
    suffixes = {"K": 1_000, "M": 1_000_000}
    suffix = value[-1:].upper()
    if suffix in suffixes:
        return int(float(value[:-1]) * suffixes[suffix])
    return int(value)


@lru_cache(maxsize=1)
def _pi_catalog() -> list[tuple[str, str, int]]:
    """Read pi's merged catalog, including built-in providers and custom models."""
    try:
        result = subprocess.run(
            [PI_PATH, "--list-models"], capture_output=True, text=True,
            timeout=30, env=operator_env(), check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    if result.returncode != 0:
        return []
    rows = []
    for line in result.stdout.splitlines()[1:]:
        columns = line.split()
        if len(columns) < 3:
            continue
        try:
            rows.append((columns[0], columns[1], _count(columns[2])))
        except ValueError:
            continue
    return rows


def resolve_model(pattern: str) -> tuple[str, str]:
    """Resolve a model pattern to an explicit ``(provider, model_id)`` pair.

    Pi's catalog merges built-in models with ``~/.pi/agent/models.json``. Using
    that same merged view lets SSSF target direct providers such as
    ``openai/gpt-5.6-terra`` without re-registering built-in models locally.
    """
    catalog = [(provider, model_id) for provider, model_id, _ in _pi_catalog()]
    if "/" in pattern:
        provider, model_id = pattern.split("/", 1)
        if (provider, model_id) in catalog:
            return provider, model_id
    matches = [(provider, model_id) for provider, model_id in catalog
               if pattern == model_id or pattern in model_id]
    exact = [match for match in matches
             if match[1] == pattern or match[1].endswith("/" + pattern)]
    if len(exact) == 1:
        return exact[0]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"model pattern {pattern!r} not found in pi --list-models — "
                         "authenticate/register it or fix the config")
    raise ValueError(f"model pattern {pattern!r} is ambiguous: {matches}")


def _context_tokens(usage: dict) -> int:
    """Tokens occupying the window after a turn.

    Mirrors pi's own `calculateContextTokens` (coding-agent
    `core/compaction/compaction.ts`), which is what pi compacts against and
    shows in its footer: prefer the provider's `totalTokens`, else sum the
    parts. Cache reads count — cached prompt is still prompt.
    """
    total = usage.get("totalTokens") or 0
    if total:
        return int(total)
    return int(sum(usage.get(part) or 0
                   for part in ("input", "output", "cacheRead", "cacheWrite")))


def _warn_if_models_path_was_explicit(detail: str) -> None:
    """Stay quiet for the default MODELS_JSON — a fresh `pi` install has no
    custom-models file, and that absence is normal, not an error. But an
    operator who set PI_MODELS_PATH themselves and pointed it somewhere
    broken made a mistake, not an absence, and silently falling back to only
    the built-in catalog would hide it. `warnings.warn` rather than `print`:
    this module has no `run`/console handle to log through, and modules here
    never print.
    """
    if "PI_MODELS_PATH" in os.environ:
        warnings.warn(
            f"PI_MODELS_PATH={MODELS_JSON!r}: {detail}; falling back to "
            "pi's built-in catalog only",
            RuntimeWarning, stacklevel=3)


def context_window(provider: str, model_id: str) -> int:
    """The model's context ceiling from pi's merged model catalog.

    MODELS_JSON is for CUSTOM models only — a fresh `pi` install has no such
    file, and `pi` itself works fine without one. Confirmed on a real box: a
    stamped scratch repo's pi run died here with FileNotFoundError before pi
    ever launched, on a machine where `pi --list-models` returns the built-in
    catalog correctly. So a missing file, a non-UTF-8 or malformed-JSON file,
    or a file that parses to valid JSON that isn't an object (`null`, `[]`,
    `"x"`) is not an error here either — it just means this run has no
    (readable) custom models, and the real fallback (the `_pi_catalog()` scan
    below, which already knows the built-ins) is what answers instead.

    That silence is scoped to the DEFAULT path only. An operator who set
    PI_MODELS_PATH explicitly and it turns out unreadable or malformed gets a
    RuntimeWarning instead of a silent fallback — losing the default file is
    normal, but losing a path someone deliberately configured is a mistake
    they need to know about.
    """
    registry: object = {}
    try:
        registry = json.loads(Path(MODELS_JSON).read_text())
    except (OSError, ValueError) as exc:
        # OSError: missing file, permission denied, etc. ValueError: covers
        # both json.JSONDecodeError (malformed JSON) and UnicodeDecodeError
        # (non-UTF-8 bytes) — both are ValueError subclasses, so this one
        # except clause catches every "unreadable as JSON" shape.
        _warn_if_models_path_was_explicit(f"could not be read as JSON ({exc})")
        registry = {}
    if not isinstance(registry, dict):
        _warn_if_models_path_was_explicit("parsed but is not a JSON object")
        registry = {}
    for model in registry.get("providers", {}).get(provider, {}).get("models", []):
        if model.get("id") == model_id:
            return int(model.get("contextWindow") or 0)
    for listed_provider, listed_model, window in _pi_catalog():
        if listed_provider == provider and listed_model == model_id:
            return window
    return 0


def _text_of(container: dict) -> str:
    """Join the text blocks of anything pi shapes as {content: [...]} — a
    message or a tool result."""
    return "".join(part.get("text", "") for part in container.get("content", []) or []
                   if isinstance(part, dict) and part.get("type") == "text")


class ProviderError(RuntimeError):
    """The PROVIDER refused/errored on a turn — not a malformed-JSON problem.

    Real incident: a run against google/gemini-3.6-flash did real work (ls,
    read x3, find, write — it even wrote its findings file), then the
    provider returned an HTTP 429 (quota exhausted) on the final turn, three
    times (initial send + both JSON-correction retries). Each errored turn's
    `message_end` carried empty text, so the OLD code treated it as malformed
    JSON, burned both corrections against a wall retrying could not fix, and
    finally raised "scout never produced valid GenericOutput JSON: no JSON
    object found in the response" — blaming the agent's formatting for a
    provider outage. pi already reports `stopReason: "error"` plus a
    populated `errorMessage` on exactly this event; it was just unused.

    Mirrors agent_cc.classify()/CodingAgentError in spirit — raised BEFORE
    the response text ever reaches agents._extract_json, so a 429 (or any
    other provider-side refusal) is reported as what it is instead of being
    misdiagnosed as bad JSON. Not the same machinery: agent_cc classifies off
    a dedicated `rate_limit_event`/`result` event; pi has no equivalent
    stream shape, so this reads the same signal off the assistant
    `message_end` pi already emits.
    """


def _provider_error_text(raw: str) -> str:
    """pi's `errorMessage` is often itself a JSON-encoded provider error body
    — e.g. `{"error":{"message":"...429 ... You exceeded your current
    quota..."}}`. Extract the inner message so the operator sees the
    provider's own words, not a wrapped JSON blob; fall back to the raw
    string when it is not that shape (or not JSON at all)."""
    text = (raw or "").strip()
    if not text:
        return ""
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return text
    if isinstance(parsed, dict):
        inner = parsed.get("error")
        if isinstance(inner, dict) and inner.get("message"):
            return str(inner["message"])
    return text


class ToolCallTracker:
    """Folds pi's tool stream into ONE normalized record per completed call.

    pi announces a call as a `toolCall` content block, then emits
    tool_execution_start / _update / _end for it. Only the end carries the
    result, so that is where a record is emitted — one trace event per real
    tool call, the moment it returns, instead of three shapeless ones.

    The record carries the call's real span (`started_at`/`ended_at`), which the
    tracer writes to columns so the UI can lay tool calls on a time axis without
    parsing every payload.
    """

    def __init__(self) -> None:
        self._open: dict[str, dict] = {}

    def observe(self, event: dict) -> list[dict]:
        """Returns every record a finished tool call in this event completed.

        List, not `Optional[dict]`, to share one contract with
        `agent_cc.ToolCallTracker` — whose stream CAN close several parallel
        tool calls in one event, so its version genuinely needs multiple
        results per `observe()` call. Pi's own events only ever complete one
        call each, so this side just wraps that single record.
        """
        etype = event.get("type", "")
        if etype == "message_end":
            for block in event.get("message", {}).get("content", []) or []:
                if isinstance(block, dict) and block.get("type") == "toolCall":
                    self._announce(block.get("id"), block.get("name"),
                                   block.get("arguments"))
            return []
        if etype == "tool_execution_start":
            self._announce(event.get("toolCallId"), event.get("toolName"),
                           event.get("args"))
            return []
        if etype != "tool_execution_end":
            return []

        call_id = str(event.get("toolCallId") or "")
        opened = self._open.pop(call_id, {})
        tool = str(event.get("toolName") or opened.get("tool") or "tool")
        args = event.get("args") or opened.get("args") or {}
        record = {
            "tool": tool,
            "tool_call_id": call_id,
            "args": {key: _clip(value, ARG_VALUE_CHARS) if isinstance(value, str) else value
                     for key, value in args.items()},
            "ok": not event.get("isError", False),
            "label": _label(tool, args),
        }
        result_text = _text_of(event.get("result") or {})
        if result_text:
            record["result_snippet"] = _clip(result_text, RESULT_SNIPPET_CHARS)
        record["ended_at"] = now_iso()
        if opened.get("clock"):
            record["duration_ms"] = int((time.monotonic() - opened["clock"]) * 1000)
        if opened.get("started_at"):
            record["started_at"] = opened["started_at"]
        return [record]

    def _announce(self, call_id, tool, args) -> None:
        """First sighting starts the clock; a later sighting only fills gaps."""
        if not call_id:
            return
        known = self._open.get(str(call_id), {})
        self._open[str(call_id)] = {
            "tool": tool or known.get("tool", ""),
            "args": args or known.get("args", {}),
            "started_at": known.get("started_at") or now_iso(),   # wall clock, for the row
            "clock": known.get("clock") or time.monotonic(),      # monotonic, for duration
        }


def run(request: PiRequest, on_event: Optional[Callable[[dict], None]] = None,
        on_spawn: Optional[Callable[[int], None]] = None,
        on_exit: Optional[Callable[[int], None]] = None) -> PiResult:
    """Run one non-interactive pi turn.

    `on_spawn(pid)` and `on_exit(pid)` bracket the child process so the caller
    can record it as killable — a hung coding agent is otherwise a pid you have
    to hunt for in `ps` while the run sits there.
    """
    provider, model_id = resolve_model(request.model)
    cmd = [
        PI_PATH, "-p", "--mode", "json",
        "--provider", provider, "--model", model_id,
        "--thinking", request.thinking,
        "--session-id", request.session_id,
        "--session-dir", request.session_dir,
        "--system-prompt", request.system_prompt,
    ]
    if request.tools:
        cmd += ["--tools", ",".join(request.tools)]
    for extension in request.extensions:
        cmd += ["-e", extension]
    cmd.append(request.prompt)

    raw_path = Path(request.raw_output_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    result = PiResult(session_id=request.session_id,
                      context_window=context_window(provider, model_id))
    # The last assistant turn's stopReason/errorMessage, so a run that ends
    # with no usable text can tell "the provider refused" (stopReason ==
    # "error", ProviderError below) from "the model just wrote something
    # unparseable" (any other reason, left to agents._parse_with_retries
    # exactly as before).
    last_stop_reason = ""
    last_error_message = ""
    # stdin is DEVNULL, deliberately. The prompt travels in argv, so the child
    # never needs stdin — but inheriting the parent's means pi sees a non-TTY
    # and can sit forever waiting for piped input that will never arrive or
    # EOF. That failure is silent and total: no request goes out, no bytes come
    # back, and the ADW blocks on a read loop with nothing to read. Observed as
    # a run that sat idle at 0% CPU with an empty raw_output.jsonl.
    # stderr goes to a FILE, not a pipe. With both as pipes and a blocking
    # stdout read, a child that fills the ~64KB stderr buffer blocks writing
    # stderr, stops producing stdout, and both sides wait forever — the same
    # silent 0%-CPU hang the stdin comment below describes, through the other
    # pipe. A file has no fixed-size buffer, so it cannot happen.
    stderr_path = Path(request.stderr_path) if request.stderr_path else \
        raw_path.with_name("stderr.log")
    stderr_path.parent.mkdir(parents=True, exist_ok=True)
    # Retries re-enter the SAME send() for one agent-phase (agents.py's parse
    # fixes and gate corrections keep the same pi session, appending to this
    # same log), so the size before THIS attempt's Popen is the offset that
    # separates "what this attempt wrote" from what an earlier attempt left
    # behind. Without it, a later attempt's report could show a stale cause.
    stderr_offset = stderr_path.stat().st_size if stderr_path.exists() else 0
    with stderr_path.open("a") as err:
        process = subprocess.Popen(cmd, stdin=subprocess.DEVNULL,
                                   stdout=subprocess.PIPE, stderr=err,
                                   text=True, bufsize=1, cwd=request.cwd,
                                   env=operator_env(), start_new_session=True)
    if on_spawn:
        on_spawn(process.pid)
    # `start_new_session=True` (above) takes pi out of this ADW's own process
    # group so `kill_tree` can reach it as a group leader — but that same
    # detachment means the terminal's Ctrl-C no longer reaches it directly,
    # and nothing else was left to clean it up. Mirrors agent_cc._run_once's
    # guard (agent_cc.py:603-610): any exception unwinding this loop — most
    # realistically a locked-sqlite write inside tracer.event(), called from
    # on_event — must still kill the child and let on_exit fire, or the pid
    # is stuck "alive" in run.live_children forever with nothing able to stop
    # it.
    try:
        with raw_path.open("a") as raw:
            assert process.stdout is not None
            for line in process.stdout:
                raw.write(line)
                raw.flush()                      # events land on disk as they happen
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "message_end":
                    message = event.get("message", {})
                    if message.get("role") == "assistant":
                        text = _text_of(message)
                        if text:
                            result.text = text   # last assistant message wins
                        usage = message.get("usage", {}) or {}
                        turn = _context_tokens(usage)
                        result.tokens += turn
                        result.usage.add_turn(usage, turn)
                        # Occupancy is read off the last VALID assistant turn, the
                        # way pi does it — an aborted or errored turn reports usage
                        # you can't trust, so it must not overwrite a good reading.
                        stop_reason = message.get("stopReason") or ""
                        if turn and stop_reason not in ("aborted", "error"):
                            result.context_tokens = turn
                        result.cost += (usage.get("cost", {}) or {}).get("total", 0.0) or 0.0
                        # Tracks the LAST assistant turn seen, error or not —
                        # ProviderError below only fires when that last turn
                        # errored AND left no usable text.
                        last_stop_reason = stop_reason
                        if stop_reason == "error":
                            last_error_message = message.get("errorMessage") or last_error_message
                if on_event:
                    on_event(event)
    except BaseException:
        kill_tree(process.pid)
        raise
    finally:
        if process.poll() is None:
            kill_tree(process.pid)
        result.returncode = process.wait()
        # process.stdout is the read end of the PIPE opened above. Popen never
        # closes it for us, and it is never closed anywhere else in this
        # function either — confirmed by `-W error::ResourceWarning` reporting
        # an unclosed-pipe warning per call before this line existed. A single
        # send() leaking one fd is invisible; agents.py calls send() repeatedly
        # for one agent-phase (the first prompt, then JSON-correction and gate
        # retries — see agents.py's `latest`/`spent` comment), so a long chain
        # in one ADW process accumulates them instead of each being reclaimed
        # by GC soon after. Closed HERE, after wait() so the child is already
        # reaped, and guarded so a redundant/already-closed pipe cannot itself
        # raise and mask the real exception this `finally` may be unwinding.
        try:
            if process.stdout is not None:
                process.stdout.close()
        except OSError:
            pass
        if on_exit:
            on_exit(process.pid)

    # Both, never `or`: the docstring's own motivating example — a benign
    # `Warning: Unknown --effort value 'off'` — exits 0 and would, under `or`,
    # suppress the tail entirely. A log holding that warning AND a fatal
    # traceback (whose lines match none of stderr_warnings' four prefixes)
    # would then report only the harmless line, discarding the real cause.
    tail = stderr_path.read_bytes()[stderr_offset:][-800:].decode(errors="replace")
    stderr = "\n".join(stderr_warnings(stderr_path, offset=stderr_offset) + [tail])
    # Checked BEFORE the returncode branch below, and independent of
    # returncode: the motivating incident's `pi` process exited 0 — nothing
    # crashed, one turn was just refused by the provider. Left unchecked, an
    # empty result.text here flows into agents._extract_json, which reports
    # "no JSON object found in the response" and burns both JSON-correction
    # attempts against a wall retrying cannot fix (see ProviderError's
    # docstring). Only fires when there is genuinely NO usable text; a run
    # whose last turn errored but which still produced text from an earlier
    # turn is left alone, same as today.
    if not result.text.strip() and last_stop_reason == "error":
        detail = _provider_error_text(last_error_message)
        raise ProviderError(
            "pi's provider turn ended in error rather than a response"
            f"{': ' + detail if detail else ''} — this is a provider-side "
            f"failure, not malformed JSON; not retried as one")
    if result.returncode != 0 and not result.text:
        raise RuntimeError(f"pi exited {result.returncode}: {stderr.strip()[-800:]}")
    return result
