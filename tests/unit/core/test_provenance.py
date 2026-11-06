"""library_commit: the checkout's HEAD without running git; None outside a checkout; not part of the config hash."""

from __future__ import annotations

import subprocess
from typing import TYPE_CHECKING

from ragnarok_ai.core import provenance
from ragnarok_ai.core.provenance import commit_of, library_commit
from ragnarok_ai.judges import JudgeConfig, OrdinalBands

if TYPE_CHECKING:
    from pathlib import Path

    import pytest

SHA_A = "a" * 40
SHA_B = "b" * 40


def test_library_commit_matches_git_in_this_checkout():
    expected = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    assert library_commit() == expected


def test_none_outside_a_checkout(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    package = tmp_path / "site-packages" / "ragnarok_ai"
    package.mkdir(parents=True)
    monkeypatch.setattr(provenance, "_package_root", lambda: package)
    assert library_commit() is None
    assert commit_of(package) is None


def test_symbolic_head_loose_ref_packed_ref_and_detached_head(tmp_path: Path):
    git = tmp_path / ".git"
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n")
    (git / "refs" / "heads" / "main").write_text(SHA_A + "\n")
    package = tmp_path / "src" / "ragnarok_ai"
    package.mkdir(parents=True)
    assert commit_of(package) == SHA_A

    (git / "refs" / "heads" / "main").unlink()
    (git / "packed-refs").write_text(f"# pack-refs with: peeled\n{SHA_B} refs/heads/main\n{SHA_A} refs/tags/v1\n")
    assert commit_of(package) == SHA_B

    (git / "HEAD").write_text(SHA_A + "\n")
    assert commit_of(package) == SHA_A

    (git / "HEAD").write_text("garbage\n")
    assert commit_of(package) is None


def test_worktree_gitdir_file_and_commondir(tmp_path: Path):
    main_repo = tmp_path / "repo" / ".git"
    (main_repo / "refs" / "heads").mkdir(parents=True)
    (main_repo / "refs" / "heads" / "feature").write_text(SHA_B + "\n")
    worktree_git = main_repo / "worktrees" / "wt"
    worktree_git.mkdir(parents=True)
    (worktree_git / "HEAD").write_text("ref: refs/heads/feature\n")
    (worktree_git / "commondir").write_text("../..\n")
    checkout = tmp_path / "wt"
    (checkout / "src" / "ragnarok_ai").mkdir(parents=True)
    (checkout / ".git").write_text(f"gitdir: {worktree_git}\n")
    assert commit_of(checkout / "src" / "ragnarok_ai") == SHA_B


def test_source_commit_is_recorded_but_never_decides_comparability():
    base = JudgeConfig("tests.J", "custom", "m", "faithfulness", OrdinalBands(4, 2), "1.13.0")
    with_a = JudgeConfig("tests.J", "custom", "m", "faithfulness", OrdinalBands(4, 2), "1.13.0", source_commit=SHA_A)
    with_b = JudgeConfig("tests.J", "custom", "m", "faithfulness", OrdinalBands(4, 2), "1.13.0", source_commit=SHA_B)
    assert base.config_hash == with_a.config_hash == with_b.config_hash
    assert with_a.comparable_with(with_b) and with_a != with_b  # equality still sees it
    assert with_a.to_dict()["source_commit"] == SHA_A and base.to_dict()["source_commit"] is None
    other_version = JudgeConfig("tests.J", "custom", "m", "faithfulness", OrdinalBands(4, 2), "1.14.0.dev0")
    assert not base.comparable_with(other_version)
