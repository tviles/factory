"""Config loading/validation and agent execution.

Every ADW validates its agents before running (fail fast, nothing spawns
against a half-valid config). Every agent call parses against a concrete
output type; parse failures and gate violations re-prompt the SAME session
with a correction — context intact, bounded retries. Agent proposes, code
disposes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import yaml

from . import agent_cc, agent_pi, permissions, prompts
from .data_types import (AgentCall, AgentConfig, CodingAgentRequest,
                         CodingAgentResult, EnvelopeBase, EventRecord,
                         GateCheck, GateReport, Phase, SSSFConfig,
                         UsageBreakdown)
from .utils import new_id

JSON_FIX_ATTEMPTS = 2      # continue-with-correction attempts for malformed JSON

# The dispatch point v1 never had: agents.py imported agent_pi directly and
# called it unconditionally, so `coding_agent` was recorded in the trace and
# then ignored at the call site.
ADAPTERS = {"pi": agent_pi, "claude_code": agent_cc}


class GateFailure(RuntimeError):
    pass


# ── config ───────────────────────────────────────────────────────────────────

def load_config(path: str = "adws/adw_sssf_config/sssf.config.yaml") -> SSSFConfig:
    raw = yaml.safe_load(Path(path).read_text()) or {}
    defaults = raw.get("defaults", {}) or {}
    for agent in raw.get("agents", []) or []:
        for key in ("coding_agent", "model", "thinking", "color", "tools", "writes"):
            if key in defaults:
                agent.setdefault(key, defaults[key])
        agent.setdefault("harness_engineering", defaults.get("harness_engineering", []))
    return SSSFConfig(**raw)


def resolve(cfg: SSSFConfig, name: str) -> AgentConfig:
    for agent in cfg.agents:
        if agent.name == name:
            return agent
    raise SystemExit(f"agent {name!r} is not defined in the config — "
                     f"available: {[a.name for a in cfg.agents]}")


def validate(cfg: SSSFConfig, required: list[str]) -> None:
    """Fail fast: every required name must resolve to a usable agent.

    Dispatches per agent. The v1 version called `agent_pi.resolve_model` for
    EVERY agent, which shells out to `pi --list-models` — a claude_code agent
    would have failed for the wrong reason on a machine without pi, and no
    Anthropic model is in pi's catalog anyway.
    """
    problems: list[str] = []
    needs_cc_preflight = False
    for name in required:
        try:
            agent = resolve(cfg, name)
        except SystemExit as e:
            problems.append(str(e))
            continue
        adapter = ADAPTERS.get(agent.coding_agent)
        if adapter is None:
            problems.append(f"agent {name!r}: unknown coding_agent "
                            f"{agent.coding_agent!r} (known: {sorted(ADAPTERS)})")
            continue
        for label, ref in (("system", agent.prompt_engineering.system),
                           ("user", agent.prompt_engineering.user)):
            if not Path(ref).is_file():
                problems.append(f"agent {name!r}: {label} prompt not found: {ref}")
        try:
            adapter.resolve_model(agent.model)
        except ValueError as e:
            problems.append(f"agent {name!r}: {e}")
        if agent.coding_agent != "claude_code":
            continue

        needs_cc_preflight = True
        # Everything below is checkable without spawning anything, so it
        # belongs here rather than surfacing mid-chain (hard rule 1).
        try:
            agent_cc.map_effort(agent.thinking)
        except ValueError as e:
            problems.append(f"agent {name!r}: {e}")
        try:
            agent_cc.map_tools(agent.tools)
        except ValueError as e:
            problems.append(f"agent {name!r}: {e}")
        if agent.harness_engineering:
            problems.append(
                f"agent {name!r}: harness_engineering entries "
                f"{agent.harness_engineering} are Pi extensions and cannot load "
                f"under coding_agent: claude_code. Claude Code has a built-in "
                f"Task tool — drop the entries and the four subagent_* tools, "
                f"and add 'Task' to this agent's tools.")

    # One CLI call for the whole roster, not one per agent.
    if needs_cc_preflight and not problems:
        try:
            agent_cc.preflight_auth(cfg.defaults.claude_code.inherit_api_key)
        except ValueError as e:
            problems.append(str(e))

    if problems:
        raise SystemExit("config validation failed:\n- " + "\n- ".join(problems))


# ── execution ────────────────────────────────────────────────────────────────

def execute(run, phase: Phase, call: AgentCall) -> EnvelopeBase:
    """One agent call: render prompts -> pi run -> typed parse -> gates -> envelope."""
    agent = resolve(run.cfg, phase.params.owner)
    agent_dir = run.session_dir / agent.name
    agent_dir.mkdir(parents=True, exist_ok=True)

    variables = {
        "prompt": call.prompt,
        "previous_envelope": call.previous.model_dump_json(indent=2) if call.previous else "(none)",
        "context_handoff_dir": str(run.context_handoff_dir),
    }
    system_text = prompts.render(agent.prompt_engineering.system, variables)
    user_text = prompts.render(agent.prompt_engineering.user, variables)
    prompts.save(agent_dir / "prompts", "system.md", system_text)
    prompts.save(agent_dir / "prompts", "user.md", user_text)

    session_id, reused = _agent_session_id(run, agent)
    adapter = ADAPTERS[agent.coding_agent]
    run.tracer.event(EventRecord(adw_id=run.adw_id, phase_id=phase.phase_id,
                                 type="agent_start", name=agent.name,
                                 payload={"model": agent.model, "thinking": agent.thinking,
                                          "color": agent.color,
                                          "session_id": session_id,
                                          "coding_agent": agent.coding_agent,
                                          "purpose": agent.purpose,
                                          "tools": agent.tools,  # None = all tools
                                          "harness_engineering": agent.harness_engineering}))
    run.console.agent_started(agent.name, agent.model, session_id)

    # Parse retries and gate corrections re-enter the SAME session, so the
    # last send is the one whose context occupancy is current — while spend is
    # the opposite: every send costs, so usage accumulates across all of them.
    latest: CodingAgentResult | None = None
    spent = UsageBreakdown()

    # Claude Code's --session-id CREATES and errors if the id exists, so the
    # adapter must be told which send this is. Only execute() knows: send #1
    # continues iff we rejoined a prior session, and every send after it —
    # JSON corrections, gate corrections — is by definition a continuation.
    # Pi ignores the flag; its one flag already does both.
    resumed = reused

    def send(prompt_text: str) -> CodingAgentResult:
        nonlocal latest, resumed
        request = CodingAgentRequest(
            prompt=prompt_text,
            system_prompt=system_text,
            system_prompt_path=str((agent_dir / "prompts" / "system.md").resolve()),
            model=agent.model,
            thinking=agent.thinking,
            session_id=session_id,
            resume=resumed,
            # absolute: these are read by the coding-agent subprocess, which
            # runs in repo_root
            session_dir=str((agent_dir / "pi_sessions").resolve()),
            raw_output_path=str((agent_dir / "raw_output.jsonl").resolve()),
            stderr_path=str((agent_dir / "stderr.log").resolve()),
            tools=agent.tools,
            extensions=agent.harness_engineering,
            cwd=str(run.repo_root),
            restricted=agent.writes == [],
            inherit_api_key=run.cfg.defaults.claude_code.inherit_api_key,
            on_overage=run.cfg.defaults.claude_code.on_overage,
            timeout_seconds=run.cfg.defaults.claude_code.timeout_seconds,
        )
        def _on_spawn(pid: int) -> None:
            # agent_pi.py spawns pi with start_new_session=True, which takes
            # it out of this ADW's foreground process group — Ctrl-C on the
            # ADW no longer reaches pi directly. Tracking the pid here is
            # what makes session.py's signal handler still able to reap it.
            run.live_children.add(pid)
            run.tracer.process_start(
                run.adw_id, "agent", agent.name, pid,
                f"{agent.coding_agent} {agent.name} {agent.model}")

        def _on_exit(pid: int) -> None:
            run.live_children.discard(pid)
            run.tracer.process_end(run.adw_id, pid)

        result = adapter.run(
            request,
            on_event=_event_forwarder(run, phase, agent.name, adapter),
            on_spawn=_on_spawn,
            on_exit=_on_exit)
        resumed = True                      # every later send continues
        for warning in getattr(result, "warnings", []):
            run.tracer.event(EventRecord(adw_id=run.adw_id, phase_id=phase.phase_id,
                                         type="log", name="coding_agent_warning",
                                         payload={"agent": agent.name,
                                                  "message": warning}))
        run.add_usage(result.tokens, result.cost)
        spent.merge(result.usage)
        latest = result
        return result

    # What the tree looked like before this agent got its hands on it. Every
    # send in this phase — first prompt, JSON retries, gate corrections — is
    # measured against this one baseline.
    tree_before = permissions.snapshot(run)

    result = send(user_text)
    envelope, attempt = _parse_with_retries(run, phase, call, result, send)

    # claim gates — violations flow back into the SAME session as corrections
    for gate_attempt in range(1, max(1, phase.params.retries + 1) + 1):
        violations = []
        for gate in call.gates:
            report = _as_report(gate(envelope, run))
            found = report.violations
            run.tracer.gate_row(phase, gate.__name__, report, gate_attempt)
            run.tracer.event(EventRecord(
                adw_id=run.adw_id, phase_id=phase.phase_id,
                type="gate_fail" if found else "gate_pass", name=gate.__name__,
                payload={"attempt": gate_attempt, "violations": found,
                         "checks": [c.model_dump() for c in report.checks]}))
            run.console.gate_result(gate.__name__, report)
            violations.extend(found)
        if not violations:
            break
        if gate_attempt > phase.params.retries:
            raise GateFailure(f"{agent.name} failed gates after {gate_attempt} attempt(s):\n- "
                              + "\n- ".join(violations))
        phase.attempt = gate_attempt
        run.console.retry(agent.name, gate_attempt, phase.params.retries,
                          f"{len(violations)} gate violation(s)")
        correction = ("Your previous response failed validation:\n- "
                      + "\n- ".join(violations)
                      + "\n\nFix these problems, then re-emit ONLY your Report JSON.")
        result = send(correction)
        envelope, attempt = _parse_with_retries(run, phase, call, result, send)

    # Permission is checked after every send is done, and before the envelope is
    # accepted: an agent does not get to report success on a phase in which it
    # wrote somewhere it was not allowed to.
    try:
        touched = permissions.enforce(run, phase, agent, tree_before)
    except permissions.PermissionBreach as breach:
        run.tracer.event(EventRecord(adw_id=run.adw_id, phase_id=phase.phase_id,
                                     type="error", name="permission_breach",
                                     payload={"agent": agent.name, "error": str(breach),
                                              "writes": agent.writes,
                                              "protected_files": run.cfg.defaults.protected_files}))
        raise
    if touched:
        run.tracer.event(EventRecord(adw_id=run.adw_id, phase_id=phase.phase_id,
                                     type="log", name="paths_touched",
                                     payload={"agent": agent.name, "paths": touched}))

    _persist_envelope(run, phase, agent.name, call, envelope, attempt, valid=True)
    run.console.envelope_summary(envelope)
    context = latest or result
    run.tracer.agent_session_row(run.adw_id, agent, session_id,
                                 context_tokens=context.context_tokens,
                                 context_window=context.context_window)
    run.save_agent_map(agent.name, {"session_id": session_id, "model": agent.model,
                                    "coding_agent": agent.coding_agent})
    run.tracer.event(EventRecord(adw_id=run.adw_id, phase_id=phase.phase_id,
                                 type="handoff", name=agent.name,
                                 payload={"artifacts": envelope.artifacts,
                                          "summary": envelope.summary}))
    run.tracer.event(EventRecord(adw_id=run.adw_id, phase_id=phase.phase_id,
                                 type="agent_end", name=agent.name,
                                 # Phase totals, not the last send's: a retried
                                 # phase paid for every attempt.
                                 tokens=spent.total_tokens,
                                 payload={"cost": spent.total_cost,
                                          "usage": spent.model_dump(),
                                          "context_tokens": context.context_tokens,
                                          "context_window": context.context_window}))
    run.console.agent_finished(agent.name, spent.total_tokens, spent.total_cost)
    if envelope.status != "success":
        raise RuntimeError(f"{agent.name} reported status={envelope.status!r}: {envelope.summary}")
    return envelope


# ── internals ────────────────────────────────────────────────────────────────

def _as_report(result) -> GateReport:
    """Accept a GateReport, or a legacy gate that returned a violations list."""
    if isinstance(result, GateReport):
        return result
    return GateReport(checks=[GateCheck(item=str(v), ok=False) for v in (result or [])])


def _agent_session_id(run, agent: AgentConfig) -> tuple[str, bool]:
    """Returns (session_id, reused).

    `reused` is the one place "did we rejoin a prior session?" gets decided —
    execute() reads it once to seed `resumed` rather than re-deriving this
    same agent_map lookup a second time, which would risk the two conditions
    drifting apart under a later edit.
    """
    entry = run.agent_map.get(agent.name)
    if entry and entry.get("model") == agent.model:
        return entry["session_id"], True     # rejoin the existing context window
    return f"sssf-{run.adw_id}-{agent.name}-{new_id(4)}", False


def _event_forwarder(run, phase: Phase, agent_name: str, adapter):
    """One tool_call event per real tool call, with its exact args and result."""
    tracker = adapter.ToolCallTracker()

    def forward(event: dict) -> None:
        # observe() returns a LIST: one event can close several parallel tool
        # calls at once (agent_cc's tracker shares this contract for exactly
        # that reason — see agent_cc.ToolCallTracker's docstring), so every
        # record it hands back must be traced, not just the first.
        for record in tracker.observe(event):
            # The call's span rides the columns; duration_ms stays in the payload as
            # pi's own authoritative number.
            run.tracer.event(EventRecord(adw_id=run.adw_id, phase_id=phase.phase_id,
                                         type="tool_call", name=record.pop("label"),
                                         started_at=record.pop("started_at", None),
                                         ended_at=record.pop("ended_at", None),
                                         payload={**record, "agent": agent_name}))
    return forward


def _extract_json(text: str) -> dict:
    candidate = text
    if "```" in text:
        for block in text.split("```")[1::2]:
            block = block.removeprefix("json").strip()
            if block.startswith("{"):
                candidate = block
                break
    start, end = candidate.find("{"), candidate.rfind("}")
    if start == -1 or end <= start:
        raise ValueError("no JSON object found in the response")
    return json.loads(candidate[start:end + 1])


def _parse_with_retries(run, phase: Phase, call: AgentCall, result, send):
    """Parse the final response against the declared output type; on failure,
    continue the SAME session with a correction (bounded)."""
    for attempt in range(1, JSON_FIX_ATTEMPTS + 2):
        try:
            payload = _extract_json(result.text)
            return call.output_type.model_validate(payload), attempt
        except Exception as error:
            _persist_envelope(run, phase, phase.params.owner, call, None, attempt,
                              valid=False, raw=result.text)
            if attempt > JSON_FIX_ATTEMPTS:
                raise RuntimeError(
                    f"{phase.params.owner} never produced valid "
                    f"{call.output_type.__name__} JSON: {error}") from error
            run.console.retry(phase.params.owner, attempt, JSON_FIX_ATTEMPTS,
                              f"invalid {call.output_type.__name__} JSON: {error}")
            fields = ", ".join(call.output_type.model_fields.keys())
            result = send(
                f"Your response was not valid JSON for the required structure "
                f"({error}). Respond again with ONLY a JSON object with these "
                f"fields: {fields}. No prose, no code fences.")


def _persist_envelope(run, phase: Phase, agent_name: str, call: AgentCall,
                      envelope: Optional[EnvelopeBase], attempt: int,
                      valid: bool, raw: str = "") -> None:
    payload_json = envelope.model_dump_json(indent=2) if envelope else json.dumps({"raw": raw[-2000:]})
    run.tracer.envelope_row(phase, agent_name, call.output_type.__name__,
                            payload_json, valid, attempt)
    if envelope:
        record = {"agent_name": agent_name, "purpose": resolve(run.cfg, agent_name).purpose,
                  "output_type": call.output_type.__name__, "attempt": attempt,
                  **envelope.model_dump()}
        (run.session_dir / agent_name / "envelope.json").write_text(json.dumps(record, indent=2))
