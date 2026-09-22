def test_templates_import(templates_dir):
    from adw_modules import agent_cc, agent_pi, agents, data_types, permissions, utils
    assert (templates_dir / "adw_modules" / "agent_cc.py").exists()


def test_roundtrip_fixture_has_the_shapes_we_rely_on(fixture):
    events = fixture("tool_use_roundtrip.jsonl")
    types = [(e.get("type"), e.get("subtype")) for e in events]
    assert ("system", "init") in types
    assert ("result", "success") in types
    tool_uses = [b for e in events if e.get("type") == "assistant"
                 for b in (e.get("message") or {}).get("content", [])
                 if isinstance(b, dict) and b.get("type") == "tool_use"]
    tool_results = [b for e in events if e.get("type") == "user"
                    for b in (e.get("message") or {}).get("content", [])
                    if isinstance(b, dict) and b.get("type") == "tool_result"]
    assert len(tool_uses) == 1 and len(tool_results) == 1
    assert tool_uses[0]["id"] == tool_results[0]["tool_use_id"]


def test_fake_claude_replays_a_fixture(tmp_path, fake_claude, fake_claude_argv):
    import subprocess
    bindir = fake_claude(tmp_path, "tool_use_roundtrip.jsonl", stderr_text="hi\n")
    out = subprocess.run([str(bindir / "claude"), "-p", "x"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert '"type":"result"' in out.stdout.replace(" ", "")
    assert out.stderr == "hi\n"
    assert fake_claude_argv(bindir) == ["-p", "x"]
