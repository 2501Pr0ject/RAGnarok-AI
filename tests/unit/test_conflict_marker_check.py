"""The conflict-marker check used by CI: fails on markers in tracked files, tolerates RST underlines."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "check_conflict_markers.sh"
OPEN = "<" * 7 + " HEAD"
MID = "=" * 7
CLOSE = ">" * 7 + " theirs"

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="git not available")


def make_repo(tmp_path: Path, files: dict[str, str]) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    for name, content in files.items():
        (repo / name).write_text(content, encoding="utf-8")
    subprocess.run(["git", "-C", str(repo), "add", "-A"], check=True)
    return repo


def run_check(repo: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["sh", str(SCRIPT), str(repo)], capture_output=True, text=True, check=False)


def test_clean_repository_passes(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"a.py": "x = 1\n", "doc.rst": "Title\n" + MID + "\n\nText.\n"})
    result = run_check(repo)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "No conflict markers" in result.stdout


def test_rst_underline_alone_is_not_a_marker(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"howto.rst": "Sockets\n" + MID + "\n\nA " + MID + " in the middle is fine too.\n"})
    assert run_check(repo).returncode == 0


def test_markers_in_a_tracked_file_fail_and_are_named(tmp_path: Path) -> None:
    conflicted = "line\n" + OPEN + "\nours\n" + MID + "\ntheirs\n" + CLOSE + "\n"
    repo = make_repo(tmp_path, {"clean.py": "x = 1\n", "CHANGELOG.md": conflicted})
    result = run_check(repo)
    assert result.returncode == 1
    assert "CHANGELOG.md" in result.stdout
    assert "Conflict markers found" in result.stderr
    assert "clean.py" not in result.stdout


def test_untracked_files_are_ignored(tmp_path: Path) -> None:
    repo = make_repo(tmp_path, {"a.py": "x = 1\n"})
    (repo / "scratch.txt").write_text(OPEN + "\n" + MID + "\n" + CLOSE + "\n", encoding="utf-8")
    assert run_check(repo).returncode == 0
