"""Architectural lint: enforce the L1/L2 layering between subpackages.

- L1 (``hyperiax.core``): no optional / heavy deps, no L2 imports.
- L2 (``hyperiax.io``, ``hyperiax.prebuilt``): may import L1; must not
  import each other; optional external deps must be lazy (function-local
  import) so ``import hyperiax`` works with just JAX installed.

Covers T-16 plus the cross-package layering rules.
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
    "diffrax",
    "equinox",
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
    assert not offenders, f"hyperiax.core must not import {forbidden!r}; found:\n" + "\n".join(
        offenders
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
                continue  # stdlib
            try:
                origin_path = Path(origin).resolve()
                if base in origin_path.parents:
                    continue  # stdlib (lives under sys.base_prefix)
            except (OSError, ValueError):
                pass
            offenders.append(f"{path.name}:{line_no}: imports non-stdlib non-allowed {top!r}")
    assert not offenders, (
        "core/ may only import jax, numpy, hyperiax, or stdlib; found:\n" + "\n".join(offenders)
    )


# ── L2 layering: io / prebuilt sibling isolation ───────────────────
def _subpkg_files(subpkg: str) -> list[Path]:
    here = Path(__file__).resolve().parent.parent
    pkg_dir = here / "hyperiax" / subpkg
    return sorted(pkg_dir.rglob("*.py")) if pkg_dir.is_dir() else []


def test_io_does_not_import_prebuilt() -> None:
    """``hyperiax.io`` must not depend on ``hyperiax.prebuilt`` (sibling L2)."""
    offenders: list[str] = []
    for path in _subpkg_files("io"):
        for line_no, top, is_relative in _imports_in(path):
            # Absolute import of hyperiax.prebuilt:
            if not is_relative and top == "hyperiax":
                # peek at full module name in the AST
                src = path.read_text(encoding="utf-8")
                tree = ast.parse(src, filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.lineno == line_no:
                        if (node.module or "").startswith("hyperiax.prebuilt"):
                            offenders.append(f"{path.name}:{line_no}: {node.module}")
    assert not offenders, "io/ must not import from hyperiax.prebuilt:\n" + "\n".join(offenders)


def test_prebuilt_does_not_import_io() -> None:
    """``hyperiax.prebuilt`` must not depend on ``hyperiax.io`` (sibling L2)."""
    offenders: list[str] = []
    for path in _subpkg_files("prebuilt"):
        for line_no, top, is_relative in _imports_in(path):
            if not is_relative and top == "hyperiax":
                src = path.read_text(encoding="utf-8")
                tree = ast.parse(src, filename=str(path))
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.lineno == line_no:
                        if (node.module or "").startswith("hyperiax.io"):
                            offenders.append(f"{path.name}:{line_no}: {node.module}")
    assert not offenders, "prebuilt/ must not import from hyperiax.io:\n" + "\n".join(offenders)


# Optional deps that must stay function-local (lazy) in L2 modules.
LAZY_REQUIRED = ("ete3", "diffrax", "jax_tqdm")


def _bare_top_level_imports(tree: ast.Module) -> list[ast.AST]:
    """Module-level import statements *not* wrapped in try/except."""
    return [n for n in tree.body if isinstance(n, (ast.Import, ast.ImportFrom))]


def _imported_top_names(node: ast.AST) -> list[str]:
    if isinstance(node, ast.Import):
        return [alias.name.split(".", 1)[0] for alias in node.names]
    if isinstance(node, ast.ImportFrom):
        return [(node.module or "").split(".", 1)[0]]
    return []


@pytest.mark.parametrize("forbidden", LAZY_REQUIRED)
def test_l2_does_not_import_optional_deps_at_module_top(forbidden: str) -> None:
    """Optional external deps must not be bare module-level imports — that
    would break ``import hyperiax`` for users without the relevant extra.

    Acceptable forms: (a) function-local import; (b) module-level import
    inside a ``try / except ImportError`` block (the ``prebuilt/mcmc.py``
    pattern, where ``run_chain`` degrades gracefully without ``jax_tqdm``).
    """
    paths = _subpkg_files("io") + _subpkg_files("prebuilt")
    offenders: list[str] = []
    for path in paths:
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        for node in _bare_top_level_imports(tree):
            if forbidden in _imported_top_names(node):
                offenders.append(f"{path.relative_to(path.parents[2])}:{node.lineno}")
    assert not offenders, (
        f"L2 modules must lazy-import {forbidden!r} (function-local or "
        f"try/except-guarded). Found unguarded top-level imports at:\n" + "\n".join(offenders)
    )
