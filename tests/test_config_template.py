from pathlib import Path

import yaml


def _template(templates_dir: Path) -> Path:
    return templates_dir.parent / "sssf.config.yaml"


def test_template_config_still_parses_into_the_schema(templates_dir):
    from adw_modules.data_types import SSSFConfig
    raw = yaml.safe_load(_template(templates_dir).read_text())
    cfg = SSSFConfig(**raw)
    assert [a.name for a in cfg.agents]


def test_template_declares_the_claude_code_defaults_block(templates_dir):
    raw = yaml.safe_load(_template(templates_dir).read_text())
    cc = raw["defaults"]["claude_code"]
    assert cc["inherit_api_key"] is False
    assert cc["on_overage"] == "fail"
    assert cc["max_utilization"] == 1.0


def test_default_roster_is_still_pi_so_existing_installs_are_unchanged(templates_dir):
    raw = yaml.safe_load(_template(templates_dir).read_text())
    assert raw["defaults"]["coding_agent"] == "pi"


def test_skill_md_no_longer_calls_claude_code_stubbed(templates_dir):
    skill = (templates_dir.parent.parent / "SKILL.md").read_text()
    assert "stubbed until v2" not in skill
    assert "claude_code" in skill
