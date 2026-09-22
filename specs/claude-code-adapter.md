# Claude Code Adapter for SSSF (`agent_cc.py`)

Make `coding_agent: claude_code` real, so roster agents run through headless
Claude Code (`claude -p`) on a Max subscription instead of per-token providers.

All edits land in the skill templates under
`.claude/skills/sssf/templates/adws/...` — never in a stamped copy.

Verified against **Claude Code 2.1.278** on this machine (`claude --help` plus
live `claude -p --output-format stream-json --verbose` probes). Every claim
below marked **[probed]** was observed, not remembered.

---

## Phase 1 findings — the contract the adapter must satisfy

### 1.1 `agent_pi.py`, end to end

`run(request: PiRequest, on_event, on_spawn, on_exit) -> PiResult` is the whole
public surface. It:

1. `resolve_model(request.model)` → `(provider, model_id)` against pi's merged
   catalog (`pi --list-models` + `~/.pi/agent/models.json`).
2. Builds argv: `pi -p --mode json --provider P --model M --thinking T
   --session-id S --session-dir D --system-prompt <text>`, then
   `--tools a,b,c` (comma-joined, only if `request.tools`), then one `-e <path>`
   per `request.extensions`, then the user prompt as the final positional arg.
   **Both prompts travel in argv**, never on stdin.
3. `Popen(..., stdin=DEVNULL, stdout=PIPE, stderr=PIPE, text=True, bufsize=1,
   cwd=request.cwd, env=operator_env())`. `stdin=DEVNULL` is load-bearing and
   documented in-file: an inherited non-TTY stdin made pi hang forever at 0% CPU.
4. `on_spawn(pid)` immediately after spawn.
5. Tails `process.stdout` line by line. Each raw line is appended to
   `raw_output.jsonl` and flushed, then parsed as JSON and passed to
   `on_event(event)` — streaming by construction, no buffering.
6. On `message_end` with `role == "assistant"`: last text wins into
   `result.text`; `result.tokens += _context_tokens(usage)`;
   `result.usage.add_turn(...)`; `result.context_tokens = turn` **only when the
   turn is non-zero and `stopReason` is not `aborted`/`error`**; `result.cost +=
   usage.cost.total`.
7. `process.wait()` → `result.returncode`; `on_exit(pid)`.
8. Raises `RuntimeError` only when `returncode != 0 AND not result.text`.

**Returned object — `PiResult`, the shape the adapter must reproduce exactly:**

| Field | Meaning |
|---|---|
| `text` | final assistant text; `agents._extract_json` parses the envelope out of it |
| `returncode` | child exit code |
| `session_id` | echoed back for `agent_map.json` |
| `tokens` | billing total, summed over every turn in this send |
| `cost` | dollars, summed over every turn |
| `usage` | `UsageBreakdown` (input/output/cache_read/cache_write/reasoning + per-component cost) |
| `context_tokens` | window **occupancy after the last valid turn** — not a sum |
| `context_window` | the model's ceiling; `0` means unknown |

### 1.2 The three-events-to-one-row fold

`ToolCallTracker` exists because pi announces a tool call up to three times
(`message_end` `toolCall` block → `tool_execution_start` → `tool_execution_end`)
and only the end carries the result. `_announce()` is idempotent-ish: first
sighting starts both `started_at` (wall clock, for the row) and `clock`
(monotonic, for duration); a later sighting only fills gaps. `observe()` returns
a record **only** on `tool_execution_end`, keyed by `toolCallId`.

`agents._event_forwarder` then pops `label` → `EventRecord.name`, pops
`started_at`/`ended_at` → columns, and writes the rest as payload plus `agent`.
Required payload keys: **`{tool, tool_call_id, args, result_snippet, ok,
duration_ms, agent}`**. Clip limits: `RESULT_SNIPPET_CHARS = 20_000`,
`ARG_VALUE_CHARS = 20_000`, `LABEL_CHARS = 80`. `_label()` picks the first
non-empty string among `PRIMARY_ARGS = (command, path, file_path, pattern,
query, url)`, else any string arg.

### 1.3 Where `coding_agent` is read and dispatched

It is **read in four places and dispatched in none**:

- `data_types.AgentConfig.coding_agent` / `ConfigDefaults.coding_agent` —
  `Literal["pi", "claude_code"]`, default `pi`.
- `agents.load_config` merges `defaults.coding_agent` into each agent entry.
- `agents.validate` (`agents.py:61`) rejects anything but `"pi"`.
- `agents.execute` records it on the `agent_start` event, in
  `tracer.agent_session_row`, and in `run.save_agent_map` — then calls
  `agent_pi.run(...)` unconditionally (`agents.py:127`).

`agent_cc.py` is 15 lines: a module docstring and `def run(*args, **kwargs)`
that raises `NotImplementedError`. Nothing imports it.

**So there is no dispatch table to extend — one must be introduced.**

Second, quieter bug: `agents.validate` calls `agent_pi.resolve_model(agent.model)`
for *every* agent (`agents.py:69`). That shells out to `pi --list-models`. A
`claude_code` agent on a machine without pi would fail validation for the wrong
reason, and no Anthropic model is in pi's catalog anyway. **Validation must
dispatch per-agent too.**

### 1.4 Session handling

- `session.ensure(cfg, adw_id)` mints or pins `adw_id`, builds the `Run`,
  registers the ADW's own pid, and installs a SIGTERM/SIGINT handler that calls
  `tracer.session_finish(ok=False)` (which closes every open `processes` row)
  and re-raises as `SystemExit`. **It does not kill child processes.**
- `Run.__init__` loads `{data_dir}/sessions/{adw_id}/agent_map.json` if present.
  `run.save_agent_map(name, entry)` rewrites the whole file.
- `agents._agent_session_id(run, agent)` (`agents.py:228`):
  ```python
  entry = run.agent_map.get(agent.name)
  if entry and entry.get("model") == agent.model:
      return entry["session_id"]          # rejoin the existing context window
  return f"sssf-{run.adw_id}-{agent.name}-{new_id(4)}"
  ```
  Changing an agent's model silently invalidates its session — by design.
- `agent_map.json` entry shape: `{"session_id", "model", "coding_agent"}`,
  written **once, after gates pass**, at the end of `execute()`.

**`ph.call()` → `agents.execute()` sends more than once against that one id:**

1. `send(user_text)` — the real task.
2. `_parse_with_retries` — up to `JSON_FIX_ATTEMPTS = 2` corrections
   ("Respond again with ONLY a JSON object with these fields: ...").
3. The gate loop — up to `phase.params.retries` corrections
   ("Your previous response failed validation: ... re-emit ONLY your Report JSON"),
   **each followed by its own `_parse_with_retries`**.

Every one of those re-enters the same session id. The comment at `agents.py:106`
states the invariant precisely: *the last send is the one whose context occupancy
is current, while spend accumulates across all of them* — hence `latest` for
`context_tokens` and `spent: UsageBreakdown` for the `agent_end` totals.

With pi this is free: `--session-id` **creates or continues**, identically.
**Claude Code does not work that way.** See §2 below — this is the single
largest design delta.

Cross-workflow resumption works the same way: `--adw-id` re-joins the session
dir, `agent_map.json` is reloaded, and an agent whose `model` is unchanged gets
its previous `session_id` back.

### 1.5 `tracer.py` and the event schema

Events the adapter's caller must produce (types are free-form `TEXT`, but the
visualizer and `EventRecord`'s docstring fix the vocabulary):
`phase_start · agent_start · tool_call · handoff · gate_pass · gate_fail · log ·
agent_end · phase_end · error`.

`agents.execute` already emits `agent_start`, `handoff`, `agent_end`, gate rows,
envelope rows, and permission events. **The only event the adapter is
responsible for is `tool_call`** (via `on_event` → `_event_forwarder`), with the
payload contract in §1.2.

Token accounting paths:
- `run.add_usage(result.tokens, result.cost)` after every send →
  `tracer.session_add_usage` → `UPDATE sessions SET total_tokens = total_tokens + ?,
  total_cost = total_cost + ?`.
- `agent_end` carries `tokens=spent.total_tokens` and
  `payload={cost, usage, context_tokens, context_window}`.
- `tracer.agent_session_row(...)` upserts `agent_sessions` with
  `context_tokens` / `context_window`.

`processes` table: `process_start(ProcessRecord(adw_id, kind, name, pid,
command))` on spawn, `process_end(adw_id, pid)` on exit,
`processes_end_all(adw_id)` from `session_finish`. `kind` is `'adw'` or
`'agent'`; `command` is stored so a recycled pid is not killed by mistake.

### 1.6 `agents.validate()` and the config schema

```python
def validate(cfg, required):          # fail fast, nothing spawns
    for name in required:
        resolve(cfg, name)            # must exist in the roster
        agent.coding_agent == "pi"    # else "not implemented in v1"
        Path(system).is_file() and Path(user).is_file()
        agent_pi.resolve_model(agent.model)
    raise SystemExit("config validation failed:\n- " + ...)
```

Schema (`data_types.AgentConfig`): `name`, `coding_agent`
(`Literal["pi","claude_code"]`), `model` (`provider/model-id`), `thinking`
(free string; documented ladder `off|minimal|low|medium|high|xhigh|max`),
`color`, `purpose`, `prompt_engineering{system,user}`, `harness_engineering:
list[str]` (pi `-e` paths), `tools: Optional[list[str]]` (`None` = all),
`writes: Optional[list[str]]` (`None` = unrestricted · `[]` = read-only ·
`[...]` = only these).

### 1.7 The `writes` boundary is harness-agnostic — **confirmed**

`permissions.snapshot(run)` fingerprints the working tree with
`git diff HEAD --numstat` plus `git ls-files --others --exclude-standard`.
`enforce()` re-snapshots after the agent's last send and diffs the two
fingerprint maps. It never observes the agent, only the repo.

Nothing in `permissions.py` imports or mentions pi. It takes `run`, `phase`,
`agent: AgentConfig`, and `before: dict[str,str]`. `agents.execute` takes
`tree_before` *before* the first send and enforces *after* the last one, so it
already covers the whole multi-send correction loop.

**It works unchanged for Claude Code**, with one precondition worth stating
explicitly: the child must run with `cwd == run.repo_root`. That rules out
isolating the agent into a scratch cwd (see §7b).

---

## Phase 1 findings — the real Claude Code CLI

`claude --version` → **2.1.278**. Everything below is from `--help` and live runs.

### Flags that matter

| Flag | Verified behaviour |
|---|---|
| `-p, --print` | non-interactive; required for everything below |
| `--output-format stream-json` | **[probed] requires `--verbose`** — without it: `Error: When using --print, --output-format=stream-json requires --verbose` |
| `--session-id <uuid>` | **[probed] must be a valid UUID** (`Error: Invalid session ID. Must be a valid UUID.`) and **[probed] create-only** — reusing one exits 1 with `Error: Session ID <id> is already in use.` |
| `--resume <id>` | **[probed]** continues with full context; keeps the same session id in `init` and `result` (no `--fork-session`) |
| `--session-id` + `--resume` | **[probed]** `Error: --session-id can only be used with --continue or --resume if --fork-session is also specified.` |
| `--model` | alias (`opus`, `sonnet`, `fable`, `haiku`) or full id (`claude-opus-5`) |
| `--system-prompt` | replaces the default system prompt entirely |
| `--append-system-prompt` | appends to the default system prompt |
| `--tools <names...>` | **[probed] the availability filter** — `--tools Read,Bash` yielded exactly `['Bash','Read']` in `init`. `""` = no tools, `"default"` = all. **[probed] an unknown name is silently dropped** (`BogusTool` produced no error) |
| `--allowedTools` | **[probed] NOT an availability filter** — `--allowedTools Read` left all 28 tools available. It is a *permission* pre-approval list, and accepts patterns like `Bash(git *)` |
| `--disallowedTools` | permission denylist, same grammar |
| `--permission-mode` | `acceptEdits · auto · bypassPermissions · manual · dontAsk · plan` |
| `--permission-prompts none` | **anything that would prompt is denied automatically** instead of hanging; the permission mode still decides everything else |
| `--effort <level>` | documented `low · medium · high · xhigh · max`. **Not choice-validated at parse time** — `--effort off` and `--effort minimal` produce no parse error, so whether they are honoured is unverified |
| `--max-turns <n>` | undocumented in `--help` but **[probed] accepted** (a genuinely unknown flag errors with `unknown option`) |
| `--max-budget-usd <amount>` | print-mode spend ceiling |
| `--setting-sources <user,project,local>` | **[probed] `""` disabled everything measurable**: hook events 12 → 0, skills 0, slash commands 53 → 0, MCP 0 |
| `--strict-mcp-config` | ignore all MCP config except `--mcp-config` |
| `--disable-slash-commands` | disables all skills |
| `--json-schema <schema>` | native structured-output validation |
| `--agents <json>` | define custom subagents inline |
| `--add-dir` | extra allowed directories |
| `--bare` | **DO NOT USE** — help states *"Anthropic auth is strictly ANTHROPIC_API_KEY or apiKeyHelper (OAuth and keychain are never read)"*, i.e. it defeats the entire point of this work |
| `--safe-mode` | disables CLAUDE.md/skills/plugins/hooks/MCP while *"auth, model selection, built-in tools, and permissions work normally"* — the viable blunt alternative to `--setting-sources ""` |
| `--restricted` | removes Bash/WebFetch unless `--tools` names them, ignores user/project/local settings, confines file tools to the working dirs, refuses `bypassPermissions` |

### Event shapes [probed]

One real run (`claude -p "Read README.md and tell me its contents in 3 words."
--output-format stream-json --verbose --session-id <uuid> --model
claude-haiku-4-5-20251001 --system-prompt ... --allowedTools Read
--permission-mode acceptEdits --setting-sources "" --strict-mcp-config`)
produced, in order:

```
system/init → assistant(thinking) → assistant(tool_use) → rate_limit_event
→ user(tool_result) → system/thinking_tokens ×2 → assistant(thinking)
→ assistant(text) → result/success
```

**`system` / `init`** — keys include `session_id`, `model`, `cwd`, `tools`,
`agents`, `skills`, `slash_commands`, `plugins`, `mcp_servers`, `memory_paths`,
`permissionMode`, `claude_code_version`, and critically **`apiKeySource`**
(`"none"` on subscription OAuth, `"ANTHROPIC_API_KEY"` when that var is set).

**`assistant`** — `{type, message{id, role, content[], usage, stop_reason, model},
session_id, request_id, parent_tool_use_id, uuid, timestamp}`.
Content blocks seen: `thinking`, `tool_use` (`{id, name, input, caller}`), `text`.
⚠ **The same `usage` object repeats across every event belonging to one API
message** — two consecutive assistant events carried byte-identical usage.
Summing per event double-counts; dedupe by `message.id` or ignore per-event
usage entirely.

**`user`** — `{type, message{role, content:[{tool_use_id, type:"tool_result",
content}]}, tool_use_result, parent_tool_use_id, session_id, uuid}`.
`content` was a plain string here (`"1\thello\n2\t"`), but the block form is
`str | list[block]`. The top-level `tool_use_result` carries a richer structured
result. Errors are flagged by `is_error` on the tool_result block.

**`result` / `success`** — the envelope source:
```jsonc
{ "type":"result", "subtype":"success", "is_error":false, "num_turns":2,
  "result":"Just says hello.",                       // ← final assistant text
  "session_id":"0cc3121a-…", "total_cost_usd":0.0325828,
  "duration_ms":4178, "duration_api_ms":4145, "stop_reason":"end_turn",
  "terminal_reason":"completed", "permission_denials":[],
  "usage":{ "input_tokens":18, "cache_creation_input_tokens":14881,
            "cache_read_input_tokens":14678, "output_tokens":267,
            "output_tokens_details":{"thinking_tokens":141} },
  "modelUsage":{ "claude-haiku-4-5-20251001":{
      "contextWindow":200000, "maxOutputTokens":32000,
      "costUSD":0.0325828, "costBasis":"list", "provider":"firstParty" } } }
```
`result.usage` is the **authoritative per-send total** (18 = 10 + 8 across two
API calls). `modelUsage[model].contextWindow` supplies `context_window` for free
— no `models.json` equivalent is needed.

Also on the stream: `rate_limit_event` (subscription utilisation — this account
showed `seven_day` at 0.88), `system/thinking_tokens`, and
`system/hook_started` / `hook_response` (12 of them at repo root with default
settings; 0 with `--setting-sources ""`).

---

## Phase 2 — the plan

### 1. Command construction

**Naming.** `PiRequest` / `PiResult` are renamed to `CodingAgentRequest` /
`CodingAgentResult` — `agent_cc.run()` returning a `PiResult` would be absurd,
and `cookbooks/update_modules.md` names both types in its module table, so it is
updated in the same edit. `PiRequest = CodingAgentRequest` and
`PiResult = CodingAgentResult` aliases are kept so nothing upstream breaks.
`CodingAgentRequest` gains `resume: bool` (§2) and `timeout_seconds` (§8); the
returned shape is otherwise **identical** to what `agent_pi.run()` returns
today (§1.1), which is what lets `agents._extract_json`, `_parse_with_retries`
and the gate loop stay harness-blind.

Field-by-field from `AgentConfig` to argv:

| Config / call-site field | Claude Code flag | Notes |
|---|---|---|
| — | `-p --output-format stream-json --verbose` | always; `--verbose` is mandatory |
| `model` (`anthropic/claude-opus-5`) | `--model claude-opus-5` | §4 |
| `thinking` | `--effort <level>` | §5 |
| `session_id` (+ `resume` flag) | `--session-id <uuid>` \| `--resume <uuid>` | §2 — mutually exclusive |
| `system_prompt` (rendered `system.md`) | `--append-system-prompt-file {agent_dir}/prompts/system.md` | **decided:** append, not replace (§7d), and **by file, not argv** (§1b) |
| `tools` | `--tools <mapped,names>` **and** `--allowedTools <mapped names>` | §6 — availability *and* pre-approval |
| `harness_engineering` | `--mcp-config` / `--agents` / `--settings`, or fail | §6 |
| — | `--permission-mode acceptEdits --permission-prompts none` | plus `--restricted` when `writes == []`. §7c |
| — | `--setting-sources "" --strict-mcp-config --disable-slash-commands` | §7b |
| `cwd` (`run.repo_root`) | `Popen(cwd=...)` | required by `permissions.enforce` |
| `prompt` (rendered `user.md`) | final positional arg, or stdin above ~96KB | §1b |

Unchanged from `agent_pi.run`: `stdin=subprocess.DEVNULL` (same hang class —
and worse here, since Claude Code explicitly waits 3s for stdin and warns),
line-buffered stdout tail, raw JSONL appended+flushed to
`{agent_dir}/raw_output.jsonl`, `on_spawn` / `on_exit` brackets.

#### 1b. Argv limits — why the system prompt leaves argv

pi puts both prompts in argv. That inherits a platform limit this spec must name:
macOS here reports `ARG_MAX = 1048576` (argv **+ environ** combined), but **Linux
additionally caps a single argument at `MAX_ARG_STRLEN` = 128KB**, which `ARG_MAX`
headroom does not protect you from. (Stated as a documented Linux constraint —
not testable on this macOS box.) The growth vector is `{{previous_envelope}}`,
rendered at `agents.py:87` as `model_dump_json(indent=2)`: `ChangesOutput.stat`
is `git diff --stat` **verbatim** and `BuildOutput.changed_files` is unbounded,
so a thousand-file refactor puts the user prompt into six figures of bytes —
fine on a Mac, `E2BIG` on a Linux CI box, mid-chain.

Two probed facts shape the fix:

- **`--append-system-prompt-file` exists** (undocumented in the main `--help`; it
  errors `Append system prompt file not found: …`, not `unknown option`).
  **`--prompt-file` does not** — `unknown option`.
- **`agents.py:91–92` already writes both rendered prompts to disk** before the
  call: `prompts.save(agent_dir / "prompts", "system.md", system_text)`.

**Decided (Q6-5):**

1. **System prompt → `--append-system-prompt-file {agent_dir}/prompts/system.md`**,
   the file `execute()` has already written. This removes the larger fixed
   argument *and* is a provenance upgrade: today the audit copy and the bytes
   actually sent are two separate copies nothing proves identical. Pointing the
   flag at the audit copy makes the prompt in the session dir **be** the prompt
   that ran.
2. **User prompt stays in argv**, with a **stdin spill above ~96KB** (under
   Linux's 128KB cap): switch to `--input-format stream-json`, write one user
   message, close stdin. [probed] stdin EOF terminates the process cleanly
   (`rc=0`). This is harness-local and does not touch the output-contract triad.
   Note it partially reintroduces stdin — but `agent_pi.py`'s warning concerns
   *inheriting* the parent's stdin and waiting forever, which is a different
   thing from writing one message and closing.
3. **Noted, not built:** the architecturally correct fix is to bound the envelope
   itself, following the precedent `changes.py` already sets (full diff →
   `context_handoff/changes.diff`, envelope carries `diff_path`). That helps pi
   identically but changes how envelopes chain and touches the triad, so it is
   separate work.

Session storage: Claude Code keeps transcripts under
`<projectsDirectory>/<slug-of-cwd>/<uuid>.jsonl`, not under a `--session-dir`.

**This is a containment regression, not merely an unused field (Q6-6).** pi's
sessions live *inside* `data_dir` by construction — `agents.py:121` sets
`session_dir=str((agent_dir / "pi_sessions").resolve())` — so an archived
`sessions/<adw_id>/` is self-sufficient. Claude Code's do not.

**[probed] `CLAUDE_CONFIG_DIR` cannot buy the containment back.** Relocating it
does move transcripts, but the run then fails with
`"Not logged in · Please run /login"` in the transcript — auth does not follow.

**Decided: accept and document.** `raw_output.jsonl` already captures every
event the adapter saw — that *is* the archival record, and it is what
`tracer.py`'s "files are the raw record" already commits to. The transcript's
only unique content is resume state, which is machine-local and worthless in an
archive (you cannot resume into another machine's `~/.claude`). Copying it would
double the largest artifact in the run and buy back no recoverable capability.

So: record `projectsDirectory` (taken from `claude auth status` — authoritative,
free, no cwd-slug derivation) plus the session uuid in the `agent_start` payload
and in `agent_map.json`. Add a data-residency line to `env.sample`: a
`claude_code` roster writes conversation transcripts **outside the repo**, and
they contain whatever the agent read. The failure this exposes — `~/.claude`
cleared between ADWs in a chain — is already covered by the create/resume
fallback above.

### 2. Session semantics

**The delta.** Pi: one flag, creates-or-continues. Claude Code: `--session-id`
creates and **errors if the id exists**; `--resume` continues. So the adapter
must know, per send, which it is.

**Deterministic UUID mapping.** sssf ids are `sssf-<adw_id>-<agent>-<rand4>` —
not UUIDs, and `--session-id` requires one. Derive rather than store a second id:

```python
CC_NAMESPACE = uuid.UUID("…fixed constant in agent_cc.py…")
def cc_session_uuid(sssf_session_id: str) -> str:
    return str(uuid.uuid5(CC_NAMESPACE, sssf_session_id))
```

Deterministic beats a stored `uuid4` for three reasons: `agent_map.json` keeps
its current three-key shape; the mapping is reproducible from the trace alone
when you need to `claude --resume` a dead agent by hand; and an `--adw-id`
rejoin recomputes the same uuid from the same sssf id without extra state.

**Create-vs-continue decision.** Add `resume: bool` to `CodingAgentRequest`.
`agents.execute` owns it, because only `execute` knows the send sequence:

```python
reused = bool(entry and entry.get("model") == agent.model)   # from _agent_session_id
resumed = reused                    # send #1 continues only if we rejoined a prior session
def send(prompt_text):
    nonlocal resumed
    request = CodingAgentRequest(..., session_id=session_id, resume=resumed)
    result = ADAPTERS[agent.coding_agent].run(request, ...)
    resumed = True                  # every later send in this phase continues
    return result
```

`agent_pi.run` ignores `resume` entirely — its one flag already does both — so
the field is inert for pi and the two adapters stay behind one signature.

**Fallback.** A stale `agent_map.json` (transcript deleted, or `~/.claude`
cleared) makes `--resume` fail. The adapter catches a non-zero exit whose stderr
matches "No conversation found" / "not found", emits one `log` event, and
retries once as a create. The reverse — "already in use" on a create — means the
transcript survived but our state said otherwise; retry once as a resume. Both
are one-shot, never a loop.

**Precondition to document:** Claude Code scopes sessions by cwd. `cwd` must
stay `run.repo_root` for the whole chain or resumption silently breaks — which
is the same constraint §1.7 already imposes for `writes`.

### 3. Event translation

New `ToolCallTracker` in `agent_cc.py`, same interface as pi's
(`observe(event) -> Optional[dict]`), joining on `tool_use_id`:

| Stream event | Action |
|---|---|
| `system` / `init` | one `log` event: `{model, tools, session_id, apiKeySource, cwd, version}`. **Assert `apiKeySource == "none"`** (§7a) |
| `assistant` → content `tool_use` | `_announce(block.id, block.name, block.input)` — start wall clock + monotonic clock. Handles N parallel blocks in one message |
| `user` → content `tool_result` | close by `tool_use_id`; emit the record |
| `assistant` → content `text` | ignored for the envelope — `result.result` is authoritative and avoids the repeated-usage trap |
| `assistant` → content `thinking` | ignored |
| `result` | envelope text, usage, cost, context window (§ below) |
| `rate_limit_event` | one `log` event — subscription utilisation is exactly what a Max-billed factory wants in the trace |
| `system` / `thinking_tokens` | ignored (`result.usage.output_tokens_details.thinking_tokens` is authoritative) |
| `system` / `hook_*` | ignored; should be absent under `--setting-sources ""` |
| anything with `parent_tool_use_id` set | tag the payload with `parent_tool_use_id` and let it through — a `Task` subagent's own calls. Without `--forward-subagent-text` you only get the one `Task` tool_use/tool_result pair, which is the right granularity |

Record built on close, matching §1.2 exactly:

```python
{ "tool": name, "tool_call_id": tool_use_id,
  "args": {k: _clip(v, ARG_VALUE_CHARS) if isinstance(v,str) else v …},
  "ok": not block.get("is_error", False),
  "label": _label(tool, args),            # popped into EventRecord.name
  "result_snippet": _clip(_text_of(content), RESULT_SNIPPET_CHARS),
  "started_at": …, "ended_at": …,         # popped into columns
  "duration_ms": int((monotonic() - clock) * 1000) }
```

`_text_of` must handle `content` as **`str | list[block]`** — pi's version only
handles the list form. `_label`'s `PRIMARY_ARGS` already covers Claude Code's
arg names (`command` for Bash, `file_path` for Read/Write/Edit, `pattern` for
Grep, `query` for WebSearch, `url` for WebFetch) — no change needed.

`_clip`, `_label`, `PRIMARY_ARGS` and the three char limits are duplicated
today; move them to `utils.py` so both adapters share one definition and the
tool_call payload cannot drift between harnesses.

**Token accounting from `result`:**

```python
u = ev["usage"]
usage = UsageBreakdown(
    input_tokens       = u["input_tokens"],
    output_tokens      = u["output_tokens"],
    cache_read_tokens  = u["cache_read_input_tokens"],
    cache_write_tokens = u["cache_creation_input_tokens"],
    reasoning_tokens   = (u.get("output_tokens_details") or {}).get("thinking_tokens", 0),
    total_tokens       = input + output + cache_read + cache_write,   # pi's convention: cache reads count
    total_cost         = ev.get("total_cost_usd", 0.0))
result.tokens  = usage.total_tokens
result.cost    = usage.total_cost
result.text    = ev["result"]
result.context_window = ev["modelUsage"][model]["contextWindow"]
```

Per-component *costs* stay 0 — Claude Code reports only a total. `reasoning_tokens`
keeps its documented meaning (a share of output, never added to it).

**`context_tokens`** (occupancy, not a sum): track the last assistant message's
usage, deduped by `message.id`, and sum
`input + cache_creation + cache_read + output`. In the probe that is
`8 + 203 + 14678 + 4 = 14893` against a 200 000 window. Guard it the way pi does
— do not overwrite a good reading from a turn whose `stop_reason` is an error.

**Cost — decided (Q6-3): keep the number, add a real column, fix the UI.**

`modelUsage[…].costBasis` was `"list"` [probed]. On a Max subscription
`total_cost_usd` is notional list price, not money spent. An earlier draft of
this spec said "write it to `sessions.total_cost` and put `cost_basis` in the
payload so nobody misreads it" — that is a comment, not a mechanism, and three
things in this codebase contradict it:

- `StatChip.vue:25` renders the tooltip *"Cost — dollars billed for this run,
  all agents combined."* on both `SessionCard.vue:245` and
  `SessionTrace.vue:438`. Nothing reads the payload.
- `PhaseDetail.vue:103–121` builds a per-component cost table from
  `input_cost` / `output_cost` / `cache_read_cost` / `cache_write_cost` — all
  zero here. `money()` at `:130` returns `'$0'` for a falsy value, so the table
  renders **four `$0` rows under a non-zero total**, which reads as "these were
  free" rather than "these were not reported". The thinking row at `:109`
  computes `(output_cost × reasoning_tokens) / output_tokens`, so it reads `$0`
  too. `UsageRow.cost` is a required `number`, so the fix makes it optional and
  renders an em dash.
- `references/observability.md:30` states the invariant *"their costs sum to
  `total_cost`"*, which this breaks.

Resolution:

1. **`sessions.total_cost` keeps the list-price figure.** It is the only number
   comparable across a mixed roster, and a `claude_code` planner → `pi` builder
   chain has to sum to something meaningful. Zeroing it would hide real pi spend.
2. **`cost_basis` becomes a real column**, not a payload note —
   `agent_sessions.cost_basis TEXT` via the existing `MIGRATIONS` list
   (`tracer.py:94`), the additive pattern already in place. Values: `billed`
   (pi) / `list` (claude_code).
3. **Component costs stay 0, and `PhaseDetail` says so** — render "list price;
   per-component costs not reported by this harness" instead of a table of
   `$0.0000`. Component *tokens* are real and still render; only the dollars are
   unavailable.
4. **Amend `StatChip`'s tooltip** to wording true under both bases, and amend
   `observability.md:30` to state the invariant holds for pi and not for
   claude_code.

⚠ **The preflight proves which credential, not which meter** (§7a). A
subscription past its limits can fall through to overage billing —
`rate_limit_event.isUsingOverage` is the runtime signal, and it is recorded
alongside `utilization` (§8).

### 4. Model mapping

```python
def resolve_model(pattern: str) -> str:
    provider, _, model_id = pattern.partition("/")
    if not model_id or provider != "anthropic":
        raise ValueError(f"model {pattern!r} must be written as "
                         f"anthropic/<model-id> when coding_agent is claude_code")
    return model_id            # claude-opus-5 · claude-sonnet-5 · opus · sonnet · haiku · fable
```

`anthropic/claude-opus-5` → `--model claude-opus-5`. Aliases pass through
untouched, so `anthropic/opus` works. Claude Code resolves the rest.

**`agents.validate` must dispatch**, which also fixes the latent bug in §1.3:

```python
ADAPTERS = {"pi": agent_pi, "claude_code": agent_cc}
...
adapter = ADAPTERS.get(agent.coding_agent)
if adapter is None:
    problems.append(f"agent {name!r}: unknown coding_agent {agent.coding_agent!r}")
    continue
try:
    adapter.resolve_model(agent.model)
except ValueError as e:
    problems.append(f"agent {name!r}: {e}")
```

A non-anthropic provider on a `claude_code` agent is now a `SystemExit` from
`validate()` **before any phase opens**, never a mid-chain surprise — hard rule 1.
Same pass also validates thinking (§5) and tool names (§6), since a bad tool name
is otherwise silently dropped.

`agent_cc.validate_binary()` additionally checks `claude --version` resolves and
is ≥ 2.1 (the `--permission-prompts` flag is the floor), so a missing CLI fails
at startup too.

### 5. Thinking-level mapping

| sssf `thinking` | `--effort` | |
|---|---|---|
| `off` | `low` | clamped ⚠ |
| `minimal` | `low` | clamped ⚠ |
| `low` / `medium` / `high` / `xhigh` / `max` | same | 1:1 |

**[probed] correction to an earlier claim in this spec.** `--effort` is not
validated at *parse* time, but it *is* validated at runtime: `--effort off`
emits on **stderr**

```
Warning: Unknown --effort value 'off' — ignoring it and using the default
effort. Valid values: low, medium, high, xhigh, max.
```

and then exits 0 with `is_error: false`, `subtype: success`. So the earlier
"fails silently" framing was wrong — it fails *loudly, on a stream nothing in
this system currently reads* (§8, stderr handling).

That changes the rationale but not the decision. Left unclamped, `off`/`minimal`
fall back to **the CLI's own default effort**, which is neither `off` nor
whatever the roster asked for. Clamping to `low` is therefore a **deliberate
behaviour change toward the roster's intent**, not a no-op — the closest
honouring of "as little reasoning as possible" that the CLI actually offers. The
adapter emits one `log` event per clamped agent (`thinking 'off' is not a Claude
Code effort level; using 'low'`).

⚠ **Unresolved and worth knowing:** whether `--effort` has *any* measurable
effect is **inconclusive, leaning negative**. A single-sample A/B on
`output_tokens_details.thinking_tokens` gave low=837, minimal=961, off=867,
**max=755** — no dose-response, `max` *below* `low`, all within ±13%. There is
also **no oracle**: no effort field appears in `init`, `usage.iterations`, or
`modelUsage`. Treat the five levels as unvalidated until a multi-sample test
says otherwise; the mapping is harmless either way.

**Decided (Q2): clamp and log.** Rejected alternative — failing validation on
`off`/`minimal` — is more honest but would mean any roster sharing
`defaults.thinking: off|minimal` between a pi agent and a claude_code agent
stops validating until every entry is qualified. Clamping keeps mixed rosters
working, and the `log` event makes the substitution visible in the trace.

Also: strip `CLAUDE_EFFORT` from the child env (§7a) — it is set in a nested
Claude Code environment and would fight the flag.

### 6. Tool mapping

`--tools` is the availability filter [probed], the true analogue of `pi --tools`.
`--allowedTools` is pre-approval and is used *in addition*, so the agent's
granted tools never hit a permission prompt (§7c).

| roster | Claude Code | note |
|---|---|---|
| `read` | `Read` | |
| `bash` | `Bash` | |
| `edit` | `Edit` | |
| `write` | `Write` | |
| `grep` | `Grep` | [probed] resolves under `--tools` even though it is deferred from the default set |
| `find` | `Glob` | [probed] same |
| `ls` | — | no equivalent tool; `Read` on a directory and `Glob` cover it |

`ls` handling: drop it and emit one `log` event, rather than silently mapping it
to `Bash` (which would hand a `writes: []` agent shell access it was not given).

An unmapped name is **not** passed through: `--tools BogusTool` is silently
dropped by the CLI [probed], so a typo would cost a run. `validate()` fails on
any roster tool name that is neither in the table nor an exact Claude Code tool
name (`Task`, `WebFetch`, `WebSearch`, `NotebookEdit`, `Skill`, `ToolSearch`…).

**`tools: None` does NOT mean "omit `--tools`" — decided (Q6-1).** On pi, `None`
means **7 tools**. On Claude Code, omitting `--tools` yields **28** [probed],
including `CronCreate` (schedules work that outlives the run), `RemoteTrigger`
and `PushNotification` (reach off the machine), `SendMessage`, `Workflow`
(spawns agent fleets), `Skill` (can reach the sssf skill itself and recurse into
the factory), and `EnterWorktree` — which **moves cwd**, breaking both
`permissions.enforce` and session resumption (§2) at once. The `writes` rollback
catches none of that; it only diffs the git tree.

This is not a hypothetical path: `references/config.md:196` actively recommends
dropping `tools` and leaving `defaults.tools` unset as the escape hatch when an
extension tool gets filtered out.

So for `coding_agent: claude_code`, `tools: None` resolves to the
**pi-equivalent set** — `Read, Bash, Edit, Write, Grep, Glob` — and anything
beyond it must be named explicitly. A roster then means the same capability on
both harnesses. `validate()` additionally rejects an agent whose resolved tool
list is **empty**, mirroring `references/config.md`'s "a tool-less agent will
stall".

**`harness_engineering`.** `subagents.ts` is a pi extension — TypeScript against
`@mariozechner/pi-coding-agent` — with no Claude Code equivalent, and the four
`subagent_*` tool names it registers do not exist there. `references/config.md`
already reserves the field for *"MCP config and hooks"* on Claude Code.

**Decided (Q4): a non-empty `harness_engineering` on a `claude_code` agent is a
validation error.** No mapping, no pass-through, no silent ignore.

```
config validation failed:
- agent 'planner': harness_engineering entry 'subagents.ts' is a Pi extension
  and cannot load under coding_agent: claude_code. Claude Code has a built-in
  Task tool — drop the entry and the four subagent_* tools, and add 'Task' to
  this agent's tools.
```

Failing beats ignoring-with-a-warning because the failure mode of ignoring is
exactly the one `references/config.md` already warns about: *"the extension still
loads, the run still succeeds, and the tool the extension exists to provide is
simply never offered to the model — you find out by noticing the agent never
called it."* A hard startup error is hard rule 1.

Consequence to accept up front: the shipped starter roster gives `subagents.ts`
to **both planner and scout**, so switching either to `claude_code` is a
two-line config edit (drop the extension, drop the four `subagent_*` tool names,
add `Task`) before the roster will validate. The `claude_code` example agent
stamped into `sssf.config.yaml` ships already in that shape.

**Deferred, not dropped:** routing `.json` entries by shape (`mcpServers` →
`--mcp-config`, `agents` → `--agents`, `hooks`/`permissions` → `--settings`) is
the natural way to make this field mean something on Claude Code. It is a
follow-up, not part of this change; until then `references/config.md` keeps
describing the field as *reserved* for Claude Code.

### 7. Environment and isolation

#### (a) `ANTHROPIC_API_KEY` → wrong billing

**Confirmed, twice over.** With the var set, `init` reported
`apiKeySource: "ANTHROPIC_API_KEY"`; without it, `"none"` (subscription OAuth).
Worse, a *bad* key made `claude -p` **hang past 120 s with no `init` event at
all** rather than failing fast — the silent-stall failure class `agent_pi.py`'s
`stdin=DEVNULL` comment already documents for pi.

`utils.operator_env()` copies `os.environ` wholesale, so today the key would
flow straight through.

Add to `utils.py`:

```python
CC_STRIPPED_ENV = (
    "ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL",
    "ANTHROPIC_CUSTOM_HEADERS", "ANTHROPIC_MODEL", "ANTHROPIC_SMALL_FAST_MODEL",
    "CLAUDE_CODE_USE_BEDROCK", "CLAUDE_CODE_USE_VERTEX", "CLAUDE_CODE_USE_FOUNDRY",
    "CLAUDE_EFFORT",
)
def claude_code_env(inherit_api_key: bool = False) -> dict[str, str]:
    env = operator_env()
    if not inherit_api_key:
        for key in CC_STRIPPED_ENV:
            env.pop(key, None)
    # Nested-launch contamination: an ADW launched from inside Claude Code
    # inherits a live session's identity. Observed set on this machine:
    # CLAUDECODE, CLAUDE_CODE_ENTRYPOINT, CLAUDE_CODE_SESSION_ID,
    # CLAUDE_CODE_BRIDGE_SESSION_ID, CLAUDE_CODE_MESSAGING_SOCKET/TOKEN,
    # CLAUDE_CODE_CHILD_SESSION, CLAUDE_PID, CLAUDE_TRANSCRIPT_PATH, …
    for key in [k for k in env if k.startswith("CLAUDE_CODE_") or k in
                ("CLAUDECODE", "CLAUDE_PID", "CLAUDE_TRANSCRIPT_PATH")]:
        env.pop(key, None)
    return env
```

Configurable via a new `defaults.claude_code.inherit_api_key: false`. The second
loop is not optional and not configurable — it exists because this very
exploration ran inside Claude Code with all of those set, and an ADW launched
from a Claude Code terminal is the normal case for this repo.

**Startup preflight, not just a strip — decided (Q6-6).**

An earlier draft asserted only that `init.apiKeySource == "none"`. That check is
**one-sided and would pass a completely unauthenticated run**: `"none"` means
*no API key in use*, not *authenticated*. The `CLAUDE_CONFIG_DIR` probe reported
`apiKeySource: "none"` while the transcript read `"Not logged in · Please run
/login"`.

**[probed] `claude auth status` is a free, non-interactive, zero-model-call
oracle**, and it belongs in `validate()` — hard rule 1, before any phase opens:

```jsonc
// subscription, no key in env
{ "loggedIn": true, "authMethod": "claude.ai", "apiProvider": "firstParty",
  "projectsDirectory": "/Users/…/.claude/projects",
  "configDirectory": "/Users/…/.claude", "subscriptionType": "max" }

// same machine, ANTHROPIC_API_KEY set
{ "loggedIn": true, "authMethod": "claude.ai",
  "apiKeySource": "ANTHROPIC_API_KEY", "subscriptionType": null, "email": null }
```

Run it **under the stripped child env** (`claude_code_env()`), so it reports
what the child will actually see. The bar:

- `loggedIn == true` — always.
- when `inherit_api_key` is false: `authMethod == "claude.ai"` **and** no
  `apiKeySource` key.
- **`subscriptionType` is recorded, never gated on.** It is a plan-name string
  Anthropic controls; hard-requiring it would fail a Pro seat, a Team/Enterprise
  seat, a `setup-token` auth, or any renamed plan for no real reason.
  `authMethod == "claude.ai"` with no `apiKeySource` already proves the thing
  that matters: not billing per token.
- `projectsDirectory` is captured here and reused for the transcript path (§2),
  authoritative and free — no cwd-slug derivation.

Keep the `init.apiKeySource` read as a **cheap cross-check** that the child saw
what the preflight predicted.

⚠ **The preflight proves which credential, not which meter.** A subscription
past its limits can fall through to overage billing; `rate_limit_event`'s
`isUsingOverage` is the runtime signal, recorded per §8.

#### (b) cwd at repo root loads the repo's own config

**Confirmed:** 12 `SessionStart` hook events fired at repo root with default
settings, plus 53 slash commands, skills, and plugins. And this repo's `.claude/`
contains the sssf skill itself — an agent could invoke `/sssf` and recurse into
the factory that spawned it.

**Do not isolate cwd.** `permissions.enforce` diffs the git tree at
`run.repo_root` (§1.7) and Claude Code scopes sessions by cwd (§2). Moving cwd
breaks the `writes` boundary and session resumption together.

Isolate the *configuration* instead:

```
--setting-sources ""        # [probed] hooks 12→0, skills→0, slash→0, mcp→0
--strict-mcp-config         # no MCP unless harness_engineering asked for it
--disable-slash-commands    # belt and braces; the sssf skill must not be reachable
```

`--bare` is **rejected**: its help text says OAuth and keychain are never read,
which defeats the subscription requirement. `--safe-mode` is the fallback if
`--setting-sources ""` proves insufficient — it keeps auth normal.

Residue seen even with `--setting-sources ""`: `plugins: 1` still loaded, and
`memory_paths.auto` still pointed at the project memory dir. Add
`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` to the child env for the latter.

**[probed] A root `CLAUDE.md` is NOT injected, and this was the spec's most
misdiagnosed risk.** Under the exact chosen config
(`--append-system-prompt` + `--setting-sources ""`), a sentinel placed in a
scratch repo's root `CLAUDE.md` appeared **zero times** in the entire raw JSONL
stream — absent from the system prompt and from any injected user message.

The probe that settles it has to be built carefully: asking the model "do you
know X?" tests **reachability, not injection**. With file tools granted, the
agent simply *reads* `CLAUDE.md` and answers — which looks identical to a leak.
The sound test is `--tools ""` (no file access at all) plus a `grep` over the
raw stream. Under that test the sentinel is absent even with `--system-prompt`
full replacement, which independently disproves the system-prompt path.

Also settled: **`memory_paths` being populated is not evidence of injection.**
In the valid run `memory_paths` was populated *and* the sentinel was absent — it
points at the agent-memory directory, a different mechanism entirely.

Consequences:
- No lever is needed for CLAUDE.md. `--safe-mode` and
  `--exclude-dynamic-system-prompt-sections` remain available but are
  **unnecessary for this purpose**.
- `CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` is still worth setting, but relabel what
  it is for: **agent memory, not CLAUDE.md**. (`--restricted` nulls
  `memory_paths` outright, so `writes: []` agents get it for free — §7c.)
- **Residual real risk, unchanged by any flag:** any agent holding
  `read` / `grep` / `bash` can still *read* the target repo's `CLAUDE.md` of its
  own accord. That is agent behaviour governed by `--tools`, not harness
  injection, and no isolation flag addresses it.

#### (c) Headless permissions

```
--permission-mode acceptEdits      # file edits proceed
--allowedTools <the agent's mapped tools>   # its granted tools are pre-approved
--permission-prompts none          # anything else that WOULD prompt is denied, not hung
```

This is the layering the two flags were built for: `--tools` decides what
exists, `--allowedTools` decides what needs no approval, `--permission-prompts
none` turns every remaining prompt into an automatic denial instead of a hang.
`result.permission_denials` (empty in the probe) then names exactly what was
refused, and the adapter writes it into the `agent_end` payload so a
mysteriously-stalled agent is diagnosable from the trace.

`bypassPermissions` is **not** used. The `writes` rollback is the backstop, and a
backstop that is also the only guard is not a backstop.

**Decided (Q5): add `--restricted` when `agent.writes == []`.** Reviewer and
scout get defence in depth — settings files ignored, file tools confined to the
working directories, `bypassPermissions` refused outright — on top of the
`writes` rollback that already makes their read-only claim true.

**[probed] the grant survives — resolved, ships as written.** With
`--restricted --tools Read,Bash,Grep,Glob,Write`, the `init` event reported
`tools == ["Bash","Glob","Grep","Read","Write"]` — exactly the five named,
nothing stripped (confirmed across two runs; the unrestricted baseline is 28).
A `Bash` call **executed**: `tool_result is_error=false content="RESTRICTED_OK"`,
with `permission_denials == []`, `permissionMode` preserved as `acceptEdits`,
empty stderr, and `apiKeySource == "none"` (auth intact despite `--restricted`
ignoring user/project/local settings).

Bonus: `--restricted` also sets `memory_paths: null`, so it subsumes
`CLAUDE_CODE_DISABLE_AUTO_MEMORY=1` for the agents that get it.

Untested, in the interest of honesty: we never ran `--restricted` *without*
`--tools` naming Bash, so the documented stripping behaviour itself is
unverified. It does not affect the adapter, which always passes `--tools` built
from the roster.

#### (d) System-prompt mode — **decided: append**

`--append-system-prompt <rendered system.md>`, not `--system-prompt`.

This deliberately diverges from `agent_pi.py`, which replaces. Keeping Claude
Code's default system prompt means its tool-use guidance, output conventions and
file-editing discipline stay intact, which is most of what makes `Read`/`Edit`/
`Bash` work well headlessly. The agent's `system.md` lands on top as its
identity and output contract.

One consequence to watch (the second, CLAUDE.md leakage, was probed and struck):

1. **Envelope discipline.** Every `system.md` ends in a "respond with ONLY valid
   JSON" contract, and Claude Code's defaults nudge toward prose and
   explanation. If `_parse_with_retries` fires noticeably more often on
   `claude_code` agents than on pi ones, that is the cause. Escalation path, in
   order: sharpen the `## Report` section → `--json-schema` (native structured
   output, already in the CLI) → fall back to `--system-prompt`.
   **Test #9 measures this directly** — it counts `envelopes` rows with
   `valid=0`, so the regression is visible rather than anecdotal.
2. ~~CLAUDE.md leakage~~ — **struck. [probed] it does not leak**; see §7b.

Recorded as a per-agent escape hatch in the config (`system_prompt_mode:
append | replace`, default `append`) only if #1 actually bites; not built
speculatively.

### 8. Process handling

- `on_spawn(process.pid)` / `on_exit(process.pid)` keep the existing
  `tracer.process_start` / `process_end` contract. `command` is recorded as
  `f"claude_code {agent.name} {model}"`, matching pi's format.
- **Spawn with `start_new_session=True`** so the child leads its own process
  group. Claude Code spawns real grandchildren (Bash tool commands, MCP servers)
  that a bare `kill(pid)` orphans. Record the pgid alongside the pid in the
  `processes` row's `command` field.
- **Kill children first.** Add `utils.kill_tree(pid, grace=5.0)`: SIGTERM the
  process group, wait, SIGKILL the survivors. On the adapter's own timeout path
  it is called directly; `session._finalize_when_killed` is extended to call it
  for every open `processes` row of the run before `session_finish`. Today that
  handler closes trace rows but leaves the actual coding agent running — a gap
  that affects pi equally, so this is a shared fix, not a claude_code one.
- **Hang guard.** A wall-clock `timeout_seconds` on `CodingAgentRequest`
  (default ~1800, configurable via `defaults.claude_code.timeout_seconds`),
  enforced by a watchdog on the read loop. On expiry: `kill_tree`, emit an
  `error` event, raise. The phase context manager marks the phase `fail` and
  `run.finish()` settles the session, so the trace never claims a hung run is
  still running. `--max-turns` and `--max-budget-usd` are available as
  belt-and-braces ceilings inside the child.
#### stderr goes to a FILE, not a pipe — and this fixes `agent_pi.py` too

`agent_pi.py:243–280` opens **both** `stdout=PIPE` and `stderr=PIPE`, then reads
them sequentially: the loop blocks on `for line in process.stdout`, and
`process.stderr.read()` is only reached after stdout hits EOF. If the child
fills the ~64KB stderr pipe buffer while the parent is blocked on stdout, the
child blocks writing stderr, stops producing stdout, and the parent waits
forever for a line that will never come.

**Reproduced** with that exact topology (200KB to stderr, 2 lines to stdout):
parent still blocked after 10s at 0% CPU. It is the same failure class the file
already documents two comments earlier for `stdin` — *"a run that sat idle at 0%
CPU with an empty raw_output.jsonl."*

pi is quiet on stderr, so it has not bitten. Claude Code is chatty there (the
`--effort` warning in §5), and — the nasty part — **`-d/--debug` writes a lot to
stderr, so the flag you would reach for to diagnose a hang is itself a way to
cause one.**

Fix, in both adapters:

```python
stderr_path = agent_dir / "stderr.log"
with stderr_path.open("a") as err:
    process = subprocess.Popen(cmd, stdin=DEVNULL, stdout=PIPE, stderr=err, ...)
```

A file has no fixed-size buffer, so the deadlock is structurally impossible
rather than merely unlikely, and it matches how `raw_output.jsonl` already
works. After exit: read the tail, emit any `Warning:` / `Error:` lines as `log`
events (the trace is currently blind to them — §5's `--effort` warning goes
nowhere today), and reuse the same tail in the `RuntimeError` message.
`stderr.log` is written unconditionally and left even when empty: an
absent-vs-empty ambiguity is worse than a 0-byte file.

Rejected: `stderr=STDOUT` interleaves non-JSON lines into `raw_output.jsonl`,
which the parse loop silently `continue`s past — corrupting the raw record to
fix a hang. A drain thread works but needs a bounded buffer and leaves nothing
on disk to open.

**Decided (Q6-4): fix both adapters.** Same five lines, no happy-path behaviour
change, and `agent_pi.py` stops carrying a latent deadlock.

#### Failure classification — `is_error` must short-circuit the parser

Failure surfacing mirrors pi: raise when `returncode != 0 AND not result.text`.
But an earlier draft added *"additionally raise when `is_error: true` **with no
usable text**"* — and that qualifier is **wrong, twice over**. Both
`"Not logged in · Please run /login"` and a rate-limit message **are** usable
text. Under that rule they flow into `_extract_json`, fail to parse, burn both
JSON correction sends against a wall, and kill the run reporting
`planner never produced valid PlanOutput JSON` — pointing the operator at their
prompt engineering when the cause was an expired login or a weekly quota.

**Rule: `result.is_error == true` hard-fails, always, before the parser ever
sees the text.** Specifically:

| Signal | Raised as | Handling |
|---|---|---|
| last `rate_limit_event.status` in `{rejected, blocked}`, or a rate-limited `result` | `RateLimited(rateLimitType, resetsAt)` | propagates untouched; **never consumes JSON or gate retry budget** |
| `result` text matches a not-logged-in signature | `NotAuthenticated` | names the fix (`claude auth status`, `/login`) |
| any other `is_error: true` | `CodingAgentError` | carries `subtype`, `terminal_reason`, `api_error_status` |

`agents.execute` lets all three propagate. `_parse_with_retries` calls `send()`
inside its `except` block (`agents.py:284`), so a raise there propagates
correctly — the danger was never a retry loop on a *raise*, it was a retry loop
on **error text that parses as "bad JSON"**.

Every `rate_limit_event` also records `utilization` and **`isUsingOverage`** into
the `agent_end` payload.

**`isUsingOverage: true` is a hard stop by default.** The stated requirement for
this whole feature is *"use the Max subscription and not API or extra usage
tokens."* The §7a preflight guarantees the first half (which credential) but
cannot guarantee the second (which meter): an account with extra usage enabled
silently continues on **paid overage** once the subscription window is
exhausted. That is per-token billing arriving through the subscription's front
door.

So the adapter raises `OverageRefused` the moment it observes
`isUsingOverage: true`, configurable via `defaults.claude_code.on_overage:
fail | warn`, default **`fail`**. Killing a chain at 80% is the cheaper mistake
here — an unnoticed overage bill is the expensive one.

⚠ **This is a backstop, not the guarantee.** The authoritative control is
account-side: disable extra usage in the Anthropic account settings, which is
enforced by Anthropic rather than by this adapter. Document that in
`env.sample` next to the data-residency note.

**Explicitly NOT done:** no auto-wait (a `seven_day` `resetsAt` can be days out,
turning an ADW into a zombie) and no auto-fallback to a pi agent (silently
failing over to a per-token provider is precisely the billing surprise this
feature exists to avoid). Fail, name the reset time, let the operator decide.

Optional `defaults.claude_code.min_headroom`: refuse to *start* a chain when the
last recorded `utilization` exceeds a threshold. Costs nothing (reuses the last
observed value, no extra call). Permissive default; easy to delete.

### 9. Test plan

**Fixture.** Fresh scratch repo (not this one), `git init`, one trivial source
file, then `uv run <skill>/scripts/install.py`. Roster edited so exactly one
agent is `coding_agent: claude_code` with `model: anthropic/claude-sonnet-5`,
and — per §6 — with `subagents.ts` and the four `subagent_*` tools removed from
that agent.

| # | Test | Passes when |
|---|---|---|
| 1 | `validate()` unit: non-anthropic model on a `claude_code` agent | `SystemExit` at startup, message names the agent and the fix; **no process spawns** |
| 2 | `validate()` unit: `harness_engineering: [subagents.ts]` on a `claude_code` agent | `SystemExit`, message names the `Task` replacement (§6) |
| 3 | `validate()` unit: unmapped tool name | `SystemExit` rather than a silently-dropped tool |
| 4 | `just demo` with scout on `claude_code` | both runs green; `adw_prompt` + `adw_scout` envelopes parse; `writes: []` holds — `git status` clean afterwards |
| 5 | Trace shape | `select type, count(*) from events where adw_id=…` shows `tool_call` rows; each payload has all seven of `{tool, tool_call_id, args, result_snippet, ok, duration_ms, agent}`; `started_at`/`ended_at` populated as columns |
| 6 | Token accounting | `sessions.total_tokens > 0`; `agent_sessions.context_window == 200000` (or the model's real ceiling); `context_tokens < context_window` |
| 7 | Process rows | `processes` has an `agent` row with the child pid; `ended_at` set after the run |
| 8 | **Mixed chain** — planner `claude_code`, builder `pi`, via `adw_plan_build` | chain completes; `agent_map.json` holds two entries with different `coding_agent`; planner's `writes: [specs/]` enforced; the commit lands the builder's diff |
| 9 | **Forced bad JSON** — a roster agent whose `user.md` `## Report` section is replaced with "answer in prose, no JSON" | `_parse_with_retries` fires; `envelopes` table shows ≥2 rows for the phase with `valid=0` then `valid=1`; **and the proof of same-session resumption:** `result.session_id` identical across all sends, exactly **one** transcript file under `~/.claude/projects/<slug>/`, its turn count increasing, and the correction send's `cache_read_input_tokens > 0` (a cold start reads no cache) |
| 10 | Gate correction | a phase with `retries=1` and a failing gate produces a second send on the same session id (same assertions as #9) |
| 11 | `ANTHROPIC_API_KEY` set in the environment | run still reports `apiKeySource: "none"`; with `inherit_api_key: true` it reports `ANTHROPIC_API_KEY` and the adapter says so in the trace |
| 12 | Isolation | scratch repo given a root `CLAUDE.md` with a sentinel and a `SessionStart` hook that touches a file. **Run with `--tools ""`** and `grep` the raw JSONL stream for the sentinel — count must be 0, and the hook's file must not exist. ⚠ **Do NOT ask the model whether it knows the sentinel**: with file tools granted it will simply `Read` the file, which is reachability, not injection, and the test false-positives on every run |
| 13 | Kill | `just plan-build` a long task, `kill <adw pid>` mid-agent: session status `fail`, all `processes` rows closed, **and no orphaned `claude` process in `ps`** |
| 14 | Timeout | `timeout_seconds: 10` against a long task: `error` event, phase `fail`, exit non-zero, no orphan |
| 15 | **`--restricted` composes with `--tools Bash`** — scout (`writes: []`, `tools` includes `bash`) asked to run one `echo`. **[probed green ahead of implementation]** | `Bash` present in `init.tools`; the call executes. ⚠ assert on the **`tool_result` block**, not `result.result` — the final text was just `"Done."` while the echo output lived only in the tool_result, so a probe greping the final text would wrongly conclude failure |
| 16 | Append-mode envelope discipline (§7d) | count `envelopes` rows with `valid=0` across tests 4–10 for a `claude_code` agent vs the same roster on pi. A materially higher retry rate means `--append-system-prompt` is fighting the JSON contract — escalate per §7d |
| 17 | Child stderr is surfaced (§8) | run an agent with `thinking: off`: the CLI's `Warning: Unknown --effort value…` line appears as a `log` event in the trace, not only on a terminal |
| 18 | Billing provenance — **the feature's core claim** | with a clean env: `claude auth status` reports `authMethod: "claude.ai"` and no `apiKeySource`; the run's `init.apiKeySource == "none"`; and a `rate_limit_event` appears carrying subscription `utilization` (API-key calls do not consume subscription windows, so its presence is the proof of subscription metering). With `ANTHROPIC_API_KEY` set and `inherit_api_key: false`, all three still hold |
| 19 | Overage refusal (§8) | synthesise a `rate_limit_event` with `isUsingOverage: true` into the adapter's event handler: run aborts with `OverageRefused`, phase `fail`, no further sends. With `on_overage: warn`, a `log` event instead and the run continues |

**Rate-limit note for whoever runs this:** the probes above reported this
account's `seven_day` window at **0.88 utilisation**. The full matrix is ~30
model calls; run tests 1–3 (no model calls) first, and prefer
`anthropic/claude-haiku-4-5-20251001` for 4–14.

### Output contract triad — unchanged

No new `EnvelopeBase` subclass, no `user.md` `## Report` change, no `output_type=`
change at any call site. `agent_cc.run()` returns the same `PiResult` shape
`agent_pi.run()` does (§1.1), `agents._extract_json` parses `result.text`
identically, and `_parse_with_retries` / the gate loop are harness-blind. The
triad is deliberately untouched: **swapping an agent's harness must not change
what it is asked to emit.**

### Files touched

| File | Change |
|---|---|
| `adw_modules/agent_cc.py` | the adapter: `resolve_model`, `preflight_auth`, `map_tools`, `map_effort`, `cc_session_uuid`, `ToolCallTracker`, `run()`, the three failure classes |
| `adw_modules/agents.py` | `ADAPTERS` dispatch table; per-agent `validate()`; `resume` threading in `send()`; `_event_forwarder` takes the adapter's tracker |
| `adw_modules/data_types.py` | rename `PiRequest`/`PiResult` → `CodingAgentRequest`/`CodingAgentResult` (+ aliases, +`resume`, +`timeout_seconds`); `ClaudeCodeDefaults` under `ConfigDefaults` (`inherit_api_key`, `timeout_seconds`, `min_headroom`) |
| **`adw_modules/agent_pi.py`** | **stderr → file** (§8 deadlock fix, shared); accepts the renamed request type |
| `adw_modules/tracer.py` | `agent_sessions.cost_basis TEXT` appended to `MIGRATIONS` |
| `adw_modules/utils.py` | `claude_code_env()`, `kill_tree()`, shared `_clip` / `_label` / `PRIMARY_ARGS` |
| `adw_modules/session.py` | `_finalize_when_killed` kills the process tree before finalizing |
| `templates/sssf.config.yaml` | commented `claude_code` example agent; `defaults.claude_code` block |
| `references/config.md` | Claude Code rows in the thinking / model / tools / harness_engineering sections; correct the `tools: None` escape hatch at `:196` for claude_code |
| **`references/observability.md`** | `:30` cost-sum invariant qualified to pi only; note `cost_basis` |
| **`apps/visualizer/src/components/StatChip.vue`** | tooltip wording true under both cost bases |
| **`apps/visualizer/src/components/PhaseDetail.vue`** | render "list price; per-component costs not reported" instead of four `$0.0000` rows |
| `cookbooks/update_modules.md` | module table: renamed request/result types, `agent_cc.py` no longer "stubbed" |
| `SKILL.md` | v1-scope paragraph updated |
| `templates/env.sample` | a `claude_code` roster needs no key; `ANTHROPIC_API_KEY` is stripped by default; **data-residency note** — transcripts land outside the repo under `projectsDirectory` and contain whatever the agent read |

---

## Decisions taken

| # | Question | Decision | Where |
|---|---|---|---|
| Q1 | system-prompt mode | **append** (`--append-system-prompt`), keeping Claude Code's default prompt | §7d |
| Q2 | `thinking: off` / `minimal` | **clamp to `low`**, one `log` event per clamped agent | §5 |
| Q3 | roster `ls` | **drop with a logged warning** — no CC equivalent, and mapping it to `Bash` would grant shell access a `writes: []` agent never had | §6 |
| Q4 | `harness_engineering` on a `claude_code` agent | **fail validation**; JSON→MCP/agents/settings routing deferred to a follow-up | §6 |
| Q5 | permission posture | baseline (`acceptEdits` + `--allowedTools` + `--permission-prompts none`) **plus `--restricted` when `writes == []`** | §7c |

### From the design review (Q6 series)

| # | Question | Decision | Where |
|---|---|---|---|
| Q6-1 | what `tools: None` means on Claude Code | **clamp to the six-tool pi-equivalent set**, never "omit `--tools`" (28 tools incl. cron, remote triggers, worktree switching) | §6 |
| Q6-2 | rate-limit handling | classify as a distinct non-retryable failure; **never consume the JSON/gate retry budget**; no auto-wait, no auto-fallback to pi; record `utilization` + `isUsingOverage` | §8 |
| Q6-3 | subscription cost accounting | keep list price in `sessions.total_cost`, add `agent_sessions.cost_basis` as a **real column**, fix the UI tooltip and the all-zeros component table | §3 |
| Q6-4 | stderr pipe deadlock | **stderr → file, in both adapters**; surface `Warning:` lines as `log` events | §8 |
| Q6-5 | argv size limits | system prompt via `--append-system-prompt-file` (the existing audit copy); user prompt argv with **stdin spill above ~96KB**; envelope-bounding noted as the right long-term fix | §1b |
| Q6-7 | overage billing | `isUsingOverage: true` raises `OverageRefused`; `defaults.claude_code.on_overage` defaults to **`fail`**. Backstop only — the real control is disabling extra usage account-side | §8 |
| Q6-6 | auth verification + session containment | `claude auth status` **preflight in `validate()`** (`loggedIn` + `authMethod` + no `apiKeySource`; record but never gate on `subscriptionType`); accept that transcripts live outside `data_dir` and record the path | §7a, §2 |

## Still open

Not blocking — resolved by the first two implementation steps, not by a decision:

1. ~~Does a root `CLAUDE.md` leak in under append mode?~~ **RESOLVED — it does
   not** (§7b). The residual risk is an agent *reading* it of its own accord,
   which is a `--tools` question, not a harness one.
2. **Does `--effort` have any measurable effect?** **Still open, leaning
   negative** (§5). Single-sample A/B showed no dose-response and `max` below
   `low`; there is no oracle field anywhere in the stream. Needs a multi-sample
   test. Does not block — the mapping is harmless either way.
3. ~~Does `--restricted` compose with `--tools Bash`?~~ **RESOLVED — it does**
   (§7c). Probed green before implementation.
4. **Does `CLAUDE_EFFORT` override the flag in `-p` mode?** Inconclusive —
   `CLAUDE_EFFORT=off` triggered no warning where the *flag* did, consistent
   with "not read in print mode" but not proof. The adapter strips it either
   way (§7a), so this is curiosity, not a blocker.
