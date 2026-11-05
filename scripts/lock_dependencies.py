"""Record the installed dependency closure after installing pyproject.toml's pins."""

import importlib.metadata as metadata
from pathlib import Path
import tomllib

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

root = Path(__file__).resolve().parents[1]
project = tomllib.loads((root / "pyproject.toml").read_text())["project"]
pending = project["dependencies"] + project["optional-dependencies"]["dev"]
resolved = {}
while pending:
    requirement = Requirement(pending.pop())
    if requirement.marker and not requirement.marker.evaluate():
        continue
    name = canonicalize_name(requirement.name)
    if name in resolved:
        continue
    distribution = metadata.distribution(name)
    if not requirement.specifier.contains(distribution.version):
        raise RuntimeError(f"Installed {name}=={distribution.version} violates {requirement}")
    # Public Torch pin works with PyPI CPU wheels and explicitly selected CUDA indexes.
    resolved[name] = distribution.version.split("+")[0]
    pending.extend(distribution.requires or [])
(root / "requirements-lock.txt").write_text(
    "# Python 3.11 dependency closure verified on Windows; regenerate with scripts/lock_dependencies.py.\n"
    + "\n".join(f"{name}=={version}" for name, version in sorted(resolved.items()))
    + "\n",
    encoding="utf-8",
)
