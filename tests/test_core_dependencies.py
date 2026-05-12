"""Architectural lint: ``hyperiax.core`` is L1 — no optional or heavy deps.

The whole point of layering is that ``import hyperiax.core`` works in any
JAX environment with just ``jax`` and ``numpy`` installed. If anything
in ``core/*.py`` ever ``import``s ete3, trimesh, matplotlib, scipy,
tqdm, or flax, this test catches it.

Covers T-16.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest


# Heavy / optional dependencies that must not appear in core.
FORBIDDEN_IMPORTS = (
    "matplotlib",
    "scipy",
    "tqdm",
    "ete3",
    "ete4",
    "flax",
    "trimesh",
    "plotly",
)


def _core_files() -> list[Path]:
    here = Path(__file__).resolve().parent.parent
    core_dir = here / "hyperiax" / "core"
    assert core_dir.is_dir(), f"expected core/ at {core_dir}"
    return sorted(core_dir.rglob("*.py"))


def _imports_in(path: Path) -> list[tuple[int, str, bool]]:
    """Return ``[(line_no, top_level_module, is_relative)]`` for every import."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    out: list[tuple[int, str, bool]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".", 1)[0]
                out.append((node.lineno, top, False))
        elif isinstance(node, ast.ImportFrom):
            is_relative = node.level > 0
            top = (node.module or "").split(".", 1)[0]
            out.append((node.lineno, top, is_relative))
    return out


@pytest.mark.parametrize("forbidden", FORBIDDEN_IMPORTS)
def test_core_does_not_import(forbidden: str) -> None:
    """No file under ``hyperiax/core`` may import the forbidden package."""
    offenders: list[str] = []
    for path in _core_files():
        for line_no, top, is_relative in _imports_in(path):
            if is_relative:
                continue
            if top == forbidden:
                offenders.append(f"{path.relative_to(path.parents[2])}:{line_no}: {top}")
    assert not offenders, (
        f"hyperiax.core must not import {forbidden!r}; found:\n" + "\n".join(offenders)
    )


def test_core_only_imports_allowed_top_level_packages() -> None:
    """Whitelist: every absolute import in ``core/*.py`` resolves to
    ``jax``, ``numpy``, ``hyperiax``, or a stdlib module.
    """
    import importlib.util
    import sys

    allowed_external = {"jax", "numpy", "hyperiax"}
    base = Path(sys.base_prefix).resolve()
    offenders: list[str] = []
    for path in _core_files():
        for line_no, top, is_relative in _imports_in(path):
            if is_relative or not top:
                continue
            if top in allowed_external:
                continue
            spec = importlib.util.find_spec(top)
            if spec is None:
                offenders.append(f"{path.name}:{line_no}: unresolved import {top!r}")
                continue
            origin = getattr(spec, "origin", None)
            if origin in (None, "built-in", "frozen"):
                continue                                # stdlib
            try:
                origin_path = Path(origin).resolve()
                if base in origin_path.parents:
                    continue                            # stdlib (lives under sys.base_prefix)
            except (OSError, ValueError):
                pass
            offenders.append(
                f"{path.name}:{line_no}: imports non-stdlib non-allowed {top!r}"
            )
    assert not offenders, (
        f"core/ may only import jax, numpy, hyperiax, or stdlib; found:\n"
        + "\n".join(offenders)
    )
