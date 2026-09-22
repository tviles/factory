def test_clip_leaves_short_text_alone():
    from adw_modules.utils import clip
    assert clip("hello", 10) == "hello"


def test_clip_truncates_with_ellipsis():
    from adw_modules.utils import clip
    assert clip("abcdefghij", 5) == "abcde…"


def test_tool_label_prefers_command_then_file_path():
    from adw_modules.utils import tool_label
    assert tool_label("Bash", {"command": "ls  -la   src"}) == "Bash: ls -la src"
    assert tool_label("Read", {"file_path": "/a/b.py"}) == "Read: /a/b.py"


def test_tool_label_falls_back_to_any_string_then_bare_name():
    from adw_modules.utils import tool_label
    assert tool_label("X", {"weird": "value"}) == "X: value"
    assert tool_label("X", {"n": 3}) == "X"


def test_pi_helpers_are_the_shared_ones():
    """agent_pi must not keep private copies — the payload would drift."""
    from adw_modules import agent_pi, utils
    assert agent_pi._clip is utils.clip
    assert agent_pi._label is utils.tool_label


def test_renamed_types_with_back_compat_aliases():
    from adw_modules.data_types import (CodingAgentRequest, CodingAgentResult,
                                        PiRequest, PiResult)
    assert PiRequest is CodingAgentRequest
    assert PiResult is CodingAgentResult


def test_request_defaults_for_new_fields():
    from adw_modules.data_types import CodingAgentRequest
    r = CodingAgentRequest(prompt="p", system_prompt="s", model="m",
                           session_id="sid", session_dir="d",
                           raw_output_path="raw.jsonl")
    assert r.resume is False
    assert r.system_prompt_path == ""
    assert r.stderr_path == ""
    assert r.timeout_seconds == 1800


def test_claude_code_defaults():
    from adw_modules.data_types import ClaudeCodeDefaults
    d = ClaudeCodeDefaults()
    assert d.inherit_api_key is False
    assert d.on_overage == "fail"
    assert d.timeout_seconds == 1800
    assert d.max_utilization == 1.0


def test_config_defaults_carry_a_claude_code_block():
    from adw_modules.data_types import SSSFConfig
    cfg = SSSFConfig()
    assert cfg.defaults.claude_code.inherit_api_key is False
