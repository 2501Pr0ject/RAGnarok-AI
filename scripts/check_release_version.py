#!/usr/bin/env python3
"""Check that what a release publishes is what its tag names.

``publish.yml`` publishes the version written in ``pyproject.toml`` at the
tagged commit, whatever the tag is called, and a release on PyPI cannot be
replaced. This script fails before anything is built when the two disagree.

Checks, in this order:

1. ``pyproject.toml`` and ``ragnarok_ai.__version__`` carry the same
   version, in one of the forms the project uses (``1.14.0``,
   ``1.14.0rc1``, ``1.12.1.post1``, ``1.14.0.dev0``). Always.
2. With ``--tag``: the tag is exactly ``v`` followed by that version, and
   the version is not a development one (``.devN``). Release candidates and
   post-releases are accepted. Without ``--tag`` a development version is
   accepted: ``main`` carries one between two releases, and the check then
   only says that the two files agree.
3. With ``--on-branch``: the commit belongs to a remote branch matching one
   of the patterns (``main``, ``release/*``), so that a tag pushed on a
   stray commit does not publish it.
4. With ``--dist``: the folder holds exactly one wheel and one source
   archive, both named after that version.

Standard library only, so it runs before anything is installed. Exit status
0 when every requested check passes, 1 otherwise, with one line per problem
naming both values.
"""

from __future__ import annotations

import argparse
import fnmatch
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DISTRIBUTION = "ragnarok_ai"
# Releases (1.14.0), release candidates (1.14.0rc1), post-releases (1.12.1.post1), development versions (1.14.0.dev0)
VERSION = re.compile(r"\d+(\.\d+)+((a|b|rc)\d+)?(\.post\d+)?(\.dev\d+)?")
DEVELOPMENT = re.compile(r"\.dev\d+$")
REMOTE = "origin"


def pyproject_version(root: Path) -> str:
    """The ``version`` of the ``[project]`` table (read without a TOML parser: Python 3.10 has none)."""
    table = ""
    for raw in (root / "pyproject.toml").read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if line.startswith("["):
            table = line
        elif table == "[project]":
            found = re.fullmatch(r"""version\s*=\s*["']([^"']+)["']\s*(#.*)?""", line)
            if found:
                return found.group(1)
    msg = "pyproject.toml has no version in its [project] table"
    raise ValueError(msg)


def package_version(root: Path) -> str:
    """``__version__`` as written in ``src/ragnarok_ai/__init__.py`` (read, not imported)."""
    source = (root / "src" / DISTRIBUTION / "__init__.py").read_text(encoding="utf-8")
    found = re.search(r"""^__version__\s*=\s*["']([^"']+)["']""", source, re.MULTILINE)
    if not found:
        msg = f"src/{DISTRIBUTION}/__init__.py has no __version__"
        raise ValueError(msg)
    return found.group(1)


def check_versions(pyproject: str, package: str) -> list[str]:
    """Problems with the two written versions; empty when they agree and are well formed."""
    problems = []
    if pyproject != package:
        problems.append(
            f"pyproject.toml says {pyproject!r} but {DISTRIBUTION}.__version__ says {package!r}: they must be equal"
        )
    if not VERSION.fullmatch(pyproject):
        problems.append(
            f"version {pyproject!r} in pyproject.toml is not in an expected form "
            "(1.14.0, 1.14.0rc1, 1.12.1.post1, 1.14.0.dev0)"
        )
    return problems


def check_tag(tag: str, version: str) -> list[str]:
    """Problems with releasing *version* under *tag*."""
    problems = []
    if tag != f"v{version}":
        problems.append(
            f"tag {tag!r} does not match the version {version!r} of pyproject.toml: expected the tag 'v{version}'"
        )
    if DEVELOPMENT.search(version):
        problems.append(
            f"version {version!r} is a development version and cannot be released: "
            "tag the release commit, not a commit made after main moved to the next .dev0"
        )
    return problems


def branches_containing(commit: str, root: Path) -> list[str]:
    """Remote branches of ``origin`` that contain *commit*, without the remote's name."""
    done = subprocess.run(
        ["git", "branch", "--remotes", "--contains", commit, "--format=%(refname:short)"],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    prefix = f"{REMOTE}/"
    return sorted(
        name[len(prefix) :]
        for name in (line.strip() for line in done.stdout.splitlines())
        if name.startswith(prefix) and name != f"{prefix}HEAD"
    )


def check_branch(containing: list[str], patterns: list[str], commit: str) -> list[str]:
    """A problem unless one of the branches *containing* the commit matches one of *patterns*."""
    if any(fnmatch.fnmatchcase(branch, pattern) for branch in containing for pattern in patterns):
        return []
    found = ", ".join(containing) if containing else "none"
    return [
        f"commit {commit[:12]} is on no branch matching {', '.join(patterns)} of {REMOTE} (branches containing it: "
        f"{found}): a release is tagged on main or on a release branch"
    ]


def check_dist(dist: Path, version: str) -> list[str]:
    """Problems unless *dist* holds exactly one wheel and one source archive of *version*.

    Hidden files are not distributions and are left out: ``uv build`` writes a ``.gitignore`` in the
    folder, and neither the artifact upload nor the publication takes hidden files.
    """
    names = sorted(p.name for p in dist.iterdir() if not p.name.startswith(".")) if dist.is_dir() else []
    sdist = f"{DISTRIBUTION}-{version}.tar.gz"
    wheels = [name for name in names if name.startswith(f"{DISTRIBUTION}-{version}-") and name.endswith(".whl")]
    problems = []
    if len(wheels) != 1:
        problems.append(f"{dist} holds {len(wheels)} wheel(s) of version {version!r}, expected exactly 1")
    if sdist not in names:
        problems.append(f"{dist} does not hold the source archive {sdist!r}")
    extra = [name for name in names if name not in wheels and name != sdist]
    if extra:
        problems.append(f"{dist} holds files that are not of version {version!r}: {', '.join(extra)}")
    return problems


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument("--tag", help="the pushed tag (vX.Y.Z); without it only the agreement of the files is checked")
    parser.add_argument(
        "--on-branch", nargs="+", metavar="PATTERN", help="the commit must be on a remote branch matching one of these"
    )
    parser.add_argument("--commit", default="HEAD", help="the commit --on-branch is about (default: HEAD)")
    parser.add_argument("--dist", type=Path, help="a folder of built distributions to check against the version")
    parser.add_argument("--root", type=Path, default=ROOT, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    try:
        version = pyproject_version(args.root)
        problems = check_versions(version, package_version(args.root))
    except (OSError, ValueError) as exc:
        print(f"::error::{exc}")
        return 1
    if args.tag is not None:
        problems += check_tag(args.tag, version)
    if args.on_branch:
        problems += check_branch(branches_containing(args.commit, args.root), args.on_branch, args.commit)
    if args.dist is not None:
        problems += check_dist(args.dist, version)

    for problem in problems:
        print(f"::error::{problem}")
    if problems:
        return 1
    checked = [f"version {version}"]
    if args.tag is not None:
        checked.append(f"tag {args.tag}")
    if args.on_branch:
        checked.append(f"commit on {' or '.join(args.on_branch)}")
    if args.dist is not None:
        checked.append(f"distributions in {args.dist}")
    print("OK: " + ", ".join(checked))
    return 0


if __name__ == "__main__":
    sys.exit(main())
