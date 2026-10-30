"""Importing ragnarok_ai must not emit warnings.

pydantic < 2.10 warns when a field starts with ``model_`` (its protected
namespace), which ``model_version`` does on two trace models. The import
check only fails on those pydantic versions (the CI ``install-minimum`` job
runs it on pydantic 2.0.1); the configuration check fails on any version if
the fix is removed.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from ragnarok_ai.core.artifacts import Trace
from ragnarok_ai.monitor.models import TraceEvent


def test_import_emits_no_user_warning() -> None:
    """A fresh interpreter imports the package with UserWarning as error."""
    code = "import ragnarok_ai, ragnarok_ai.core.artifacts, ragnarok_ai.monitor.models"
    proc = subprocess.run(
        [sys.executable, "-W", "error::UserWarning", "-c", code],
        capture_output=True,
        text=True,
    )

    assert proc.returncode == 0, proc.stderr


@pytest.mark.parametrize("model", [Trace, TraceEvent])
def test_models_with_model_version_disable_protected_namespaces(model: type) -> None:
    """Models declaring model_version opt out of pydantic's protected namespace."""
    assert "model_version" in model.model_fields
    assert model.model_config["protected_namespaces"] == ()
