"""Where the running code comes from: the git commit of a source checkout, when there is one.

An installed wheel has no commit; a checkout does. ``library_commit()`` reads
it from the ``.git`` directory above the package without running git, so a
judge configuration can record which revision produced a run (provenance,
not comparability: the commit is not part of ``JudgeConfig.config_hash``).

It never reports another project's commit: the walk up stops at a
``site-packages`` (or ``dist-packages``) folder, since a wheel installed in
some other repository's virtual environment sits below that repository's
``.git``, and the repository found must be ragnarok-ai itself, its
``pyproject.toml`` naming the project and holding ``src/ragnarok_ai``.
"""

from __future__ import annotations

import re
from pathlib import Path

_SHA = re.compile(r"^[0-9a-f]{40}$")
_PROJECT_NAME = re.compile(r'^name\s*=\s*"ragnarok-ai"\s*$', re.MULTILINE)
_INSTALL_FOLDERS = frozenset({"site-packages", "dist-packages"})


def _package_root() -> Path:
    import ragnarok_ai

    return Path(ragnarok_ai.__file__).resolve().parent


def _is_ragnarok_checkout(folder: Path) -> bool:
    """*folder* is a ragnarok-ai source checkout: ``pyproject.toml`` names it and ``src/ragnarok_ai`` is there."""
    pyproject = folder / "pyproject.toml"
    if not pyproject.is_file() or not (folder / "src" / "ragnarok_ai").is_dir():
        return False
    return _PROJECT_NAME.search(pyproject.read_text(encoding="utf-8", errors="replace")) is not None


def _git_dir(start: Path) -> Path | None:
    """The ``.git`` of the ragnarok-ai checkout containing *start*, or ``None``.

    The walk stops at an installation folder (``site-packages``): a wheel
    installed in another repository's virtual environment must not inherit
    that repository's commit. The first ``.git`` met is accepted only when
    the folder holding it is a ragnarok-ai checkout; otherwise the walk ends.
    """
    for folder in (start, *start.parents):
        if folder.name in _INSTALL_FOLDERS:
            return None
        candidate = folder / ".git"
        if not candidate.exists():
            continue
        if not _is_ragnarok_checkout(folder):
            return None
        if candidate.is_dir():
            return candidate
        if candidate.is_file():  # worktree or submodule: a file pointing at the real directory
            text = candidate.read_text(encoding="utf-8", errors="replace").strip()
            if text.startswith("gitdir:"):
                target = Path(text.split(":", 1)[1].strip())
                return target if target.is_absolute() else (folder / target).resolve()
        return None
    return None


def _resolve_ref(git_dir: Path, ref: str) -> str | None:
    loose = git_dir / ref
    if loose.is_file():
        value = loose.read_text(encoding="utf-8", errors="replace").strip()
        return value if _SHA.match(value) else None
    # A worktree's gitdir holds HEAD; refs and packed-refs live in the common directory
    common = git_dir
    pointer = git_dir / "commondir"
    if pointer.is_file():
        common = (git_dir / pointer.read_text(encoding="utf-8").strip()).resolve()
    loose = common / ref
    if loose.is_file():
        value = loose.read_text(encoding="utf-8", errors="replace").strip()
        return value if _SHA.match(value) else None
    packed = common / "packed-refs"
    if packed.is_file():
        for line in packed.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) == 2 and parts[1] == ref and _SHA.match(parts[0]):
                return parts[0]
    return None


def commit_of(path: Path) -> str | None:
    """The HEAD commit of the git checkout containing *path*, ``None`` outside a checkout."""
    git_dir = _git_dir(path)
    if git_dir is None:
        return None
    head = git_dir / "HEAD"
    if not head.is_file():
        return None
    text = head.read_text(encoding="utf-8", errors="replace").strip()
    if _SHA.match(text):
        return text  # detached HEAD
    if text.startswith("ref:"):
        return _resolve_ref(git_dir, text.split(":", 1)[1].strip())
    return None


def library_commit() -> str | None:
    """The commit of the ragnarok-ai checkout the code runs from; ``None`` for an installed wheel."""
    try:
        return commit_of(_package_root())
    except OSError:  # unreadable repository metadata: provenance is best effort
        return None


__all__ = ["commit_of", "library_commit"]
