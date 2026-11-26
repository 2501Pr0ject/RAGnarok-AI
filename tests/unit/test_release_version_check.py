"""scripts/check_release_version.py: what a release publishes is what its tag names."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "scripts" / "check_release_version.py"

spec = importlib.util.spec_from_file_location("check_release_version", SCRIPT)
assert spec is not None and spec.loader is not None
check = importlib.util.module_from_spec(spec)
spec.loader.exec_module(check)


def project(root: Path, pyproject: str, package: str | None = None) -> Path:
    """A minimal project tree carrying the two written versions."""
    (root / "pyproject.toml").write_text(
        f'[build-system]\nrequires = ["hatchling"]\n\n[project]\nname = "ragnarok-ai"\nversion = "{pyproject}"\n\n'
        '[tool.other]\nversion = "9.9.9"\n',
        encoding="utf-8",
    )
    package_dir = root / "src" / "ragnarok_ai"
    package_dir.mkdir(parents=True, exist_ok=True)
    (package_dir / "__init__.py").write_text(
        f'"""Package."""\n\n__version__ = "{pyproject if package is None else package}"\n', encoding="utf-8"
    )
    return root


def run(root: Path, *args: str) -> tuple[int, str]:
    done = subprocess.run(
        [sys.executable, str(SCRIPT), "--root", str(root), *args], capture_output=True, text=True, check=False
    )
    return done.returncode, done.stdout + done.stderr


class TestWrittenVersions:
    def test_reads_the_project_table_only(self, tmp_path: Path):
        project(tmp_path, "1.14.0")
        assert check.pyproject_version(tmp_path) == "1.14.0"  # not the 9.9.9 of [tool.other]
        assert check.package_version(tmp_path) == "1.14.0"

    def test_the_repository_itself_is_consistent(self):
        assert check.pyproject_version(REPO) == check.package_version(REPO)
        assert check.main([]) == 0

    def test_two_different_versions_are_refused_and_both_are_named(self, tmp_path: Path):
        project(tmp_path, "1.14.0", package="1.13.1")
        code, out = run(tmp_path)
        assert code == 1
        assert "pyproject.toml says '1.14.0' but ragnarok_ai.__version__ says '1.13.1'" in out

    @pytest.mark.parametrize("version", ["1.14.0", "1.14.0rc1", "1.12.1.post1", "1.14.0.dev0", "2.0"])
    def test_expected_forms(self, version: str):
        assert check.check_versions(version, version) == []

    @pytest.mark.parametrize("version", ["v1.14.0", "1.14.0-rc1", "1.14.0.dev", "1", "latest", ""])
    def test_other_forms_are_refused(self, version: str):
        assert any("not in an expected form" in problem for problem in check.check_versions(version, version))

    def test_missing_version_is_an_error_not_a_crash(self, tmp_path: Path):
        project(tmp_path, "1.14.0")
        (tmp_path / "pyproject.toml").write_text('[project]\nname = "ragnarok-ai"\n', encoding="utf-8")
        code, out = run(tmp_path)
        assert code == 1 and "pyproject.toml has no version in its [project] table" in out


class TestTag:
    @pytest.mark.parametrize(
        ("version", "tag"), [("1.14.0", "v1.14.0"), ("1.14.0rc1", "v1.14.0rc1"), ("1.12.1.post1", "v1.12.1.post1")]
    )
    def test_a_matching_tag_passes_for_releases_candidates_and_post_releases(
        self, tmp_path: Path, version: str, tag: str
    ):
        project(tmp_path, version)
        code, out = run(tmp_path, "--tag", tag)
        assert code == 0, out
        assert f"OK: version {version}, tag {tag}" in out

    @pytest.mark.parametrize("tag", ["v1.13.2", "1.13.1", "v1.13.1.post1", "V1.13.1", "v1.13.1 "])
    def test_a_tag_that_is_not_v_plus_the_version_is_refused_and_both_are_named(self, tmp_path: Path, tag: str):
        project(tmp_path, "1.13.1")
        code, out = run(tmp_path, "--tag", tag)
        assert code == 1
        assert f"tag {tag!r} does not match the version '1.13.1' of pyproject.toml: expected the tag 'v1.13.1'" in out

    def test_a_development_version_is_refused_under_a_tag_even_a_matching_one(self, tmp_path: Path):
        # main after a release: tagging it by mistake must not publish 1.14.0.dev0
        project(tmp_path, "1.14.0.dev0")
        code, out = run(tmp_path, "--tag", "v1.14.0.dev0")
        assert code == 1
        assert "version '1.14.0.dev0' is a development version and cannot be released" in out
        assert "does not match" not in out  # the tag itself matches: one problem, the right one

    def test_a_development_version_is_accepted_without_a_tag(self, tmp_path: Path):
        # the Lint job of the CI and a dry run of publish.yml: main carries X.Y.0.dev0 between two releases
        project(tmp_path, "1.14.0.dev0")
        code, out = run(tmp_path)
        assert code == 0, out
        assert out.strip() == "OK: version 1.14.0.dev0"
        assert check.check_versions("1.14.0.dev0", "1.14.0.dev0") == []

    def test_without_a_tag_the_two_files_must_still_agree(self, tmp_path: Path):
        project(tmp_path, "1.14.0.dev0", package="1.13.1")
        assert run(tmp_path)[0] == 1

    def test_every_problem_is_reported_at_once(self, tmp_path: Path):
        project(tmp_path, "1.14.0.dev0", package="1.13.1")
        code, out = run(tmp_path, "--tag", "v1.13.1")
        assert code == 1 and out.count("::error::") == 3


class TestDist:
    @staticmethod
    def _dist(tmp_path: Path, *names: str) -> Path:
        dist = tmp_path / "dist"
        dist.mkdir()
        for name in names:
            (dist / name).write_bytes(b"")
        return dist

    def test_one_wheel_and_one_archive_of_the_version(self, tmp_path: Path):
        project(tmp_path, "1.13.1")
        dist = self._dist(tmp_path, "ragnarok_ai-1.13.1-py3-none-any.whl", "ragnarok_ai-1.13.1.tar.gz")
        code, out = run(tmp_path, "--tag", "v1.13.1", "--dist", str(dist))
        assert code == 0, out

    def test_the_gitignore_that_uv_build_writes_is_not_a_distribution(self, tmp_path: Path):
        project(tmp_path, "1.13.1")
        dist = self._dist(tmp_path, "ragnarok_ai-1.13.1-py3-none-any.whl", "ragnarok_ai-1.13.1.tar.gz", ".gitignore")
        code, out = run(tmp_path, "--dist", str(dist))
        assert code == 0, out

    def test_a_development_build_is_accepted_without_a_tag(self, tmp_path: Path):
        project(tmp_path, "1.14.0.dev0")
        dist = self._dist(tmp_path, "ragnarok_ai-1.14.0.dev0-py3-none-any.whl", "ragnarok_ai-1.14.0.dev0.tar.gz")
        assert run(tmp_path, "--dist", str(dist))[0] == 0

    def test_files_of_another_version_are_refused(self, tmp_path: Path):
        project(tmp_path, "1.13.1")
        dist = self._dist(tmp_path, "ragnarok_ai-1.13.0-py3-none-any.whl", "ragnarok_ai-1.13.0.tar.gz")
        code, out = run(tmp_path, "--dist", str(dist))
        assert code == 1
        assert "holds 0 wheel(s) of version '1.13.1', expected exactly 1" in out
        assert "does not hold the source archive 'ragnarok_ai-1.13.1.tar.gz'" in out
        assert "holds files that are not of version '1.13.1'" in out

    def test_a_leftover_file_next_to_the_right_ones_is_refused(self, tmp_path: Path):
        project(tmp_path, "1.13.1")
        dist = self._dist(
            tmp_path,
            "ragnarok_ai-1.13.1-py3-none-any.whl",
            "ragnarok_ai-1.13.1.tar.gz",
            "ragnarok_ai-1.13.1.post1-py3-none-any.whl",
        )
        code, out = run(tmp_path, "--dist", str(dist))
        assert code == 1 and "ragnarok_ai-1.13.1.post1-py3-none-any.whl" in out
        assert "holds 0 wheel" not in out and "holds 2 wheel" not in out  # the post-release wheel is not this version's

    def test_a_missing_folder_is_refused(self, tmp_path: Path):
        project(tmp_path, "1.13.1")
        assert run(tmp_path, "--dist", str(tmp_path / "nowhere"))[0] == 1


class TestBranch:
    @pytest.mark.parametrize(
        ("containing", "ok"),
        [
            (["main"], True),
            (["feat/x", "main"], True),
            (["release/1.13.1"], True),
            (["release/1.13.1", "feat/x"], True),
            (["feat/x"], False),
            (["releases/1.13.1"], False),
            (["maintenance"], False),
            (["my/main"], False),
            ([], False),
        ],
    )
    def test_main_or_a_release_branch(self, containing: list[str], ok: bool):
        problems = check.check_branch(containing, ["main", "release/*"], "0123456789abcdef")
        assert (problems == []) is ok
        if not ok:
            assert "commit 0123456789ab is on no branch matching main, release/* of origin" in problems[0]

    def test_with_real_repositories(self, tmp_path: Path):
        def git(cwd: Path, *args: str) -> str:
            done = subprocess.run(
                ["git", "-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false", *args],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )
            return done.stdout.strip()

        origin = tmp_path / "origin"
        origin.mkdir()
        git(origin, "init", "-q", "-b", "main")
        project(origin, "1.13.1")
        git(origin, "add", "-A")
        git(origin, "commit", "-q", "-m", "release")
        on_main = git(origin, "rev-parse", "HEAD")
        git(origin, "switch", "-q", "-c", "release/1.13.2")
        (origin / "a.txt").write_text("a", encoding="utf-8")
        git(origin, "add", "-A")
        git(origin, "commit", "-q", "-m", "on a release branch")
        on_release = git(origin, "rev-parse", "HEAD")
        git(origin, "switch", "-q", "-c", "feat/stray", "main")
        (origin / "b.txt").write_text("b", encoding="utf-8")
        git(origin, "add", "-A")
        git(origin, "commit", "-q", "-m", "stray")
        stray = git(origin, "rev-parse", "HEAD")
        git(origin, "switch", "-q", "main")

        clone = tmp_path / "clone"
        git(tmp_path, "clone", "-q", str(origin), str(clone))
        assert check.branches_containing(on_main, clone) == ["feat/stray", "main", "release/1.13.2"]
        assert check.branches_containing(stray, clone) == ["feat/stray"]

        for commit, expected in ((on_main, 0), (on_release, 0), (stray, 1)):
            code, out = run(clone, "--tag", "v1.13.1", "--commit", commit, "--on-branch", "main", "release/*")
            assert code == expected, out
        assert "is on no branch matching main, release/* of origin (branches containing it: feat/stray)" in out
