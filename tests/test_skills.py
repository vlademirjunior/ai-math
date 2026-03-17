from pathlib import Path

import pytest

from main import discover_skills_source


def test_skills_optional_when_missing(tmp_path: Path) -> None:
    assert discover_skills_source(tmp_path, required=False) == []


def test_skills_required_when_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        discover_skills_source(tmp_path, required=True)


def test_invalid_skill_dir(tmp_path: Path) -> None:
    root = tmp_path / ".agents"
    root.mkdir()
    (root / "bad-skill").mkdir()

    with pytest.raises(ValueError):
        discover_skills_source(tmp_path, required=False)


def test_valid_skills_tree(tmp_path: Path) -> None:
    root = tmp_path / ".agents"
    root.mkdir()
    skill = root / "code-review"
    skill.mkdir()
    (skill / "SKILL.md").write_text("# Skill\n", encoding="utf-8")

    assert discover_skills_source(tmp_path, required=False) == ["/.agents/"]
