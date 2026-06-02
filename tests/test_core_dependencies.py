"""Architectural lint: enforce the L1/L2 layering between subpackages.

- L1 (``hyperiax.core``): no optional / heavy deps. Newick I/O lives here
  too (pure-Python parser in ``core/builders.py``), so ``ete3`` must never
  appear.
- L2 (``hyperiax.prebuilt``): may import L1; optional external deps must be
  lazy (function-local import) so ``import hyperiax`` works with just JAX
  installed.

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


def _l1_files() -> list[Path]:
    """All L1 source files: ``hyperiax.core`` and ``hyperiax.utils``."""
    here = Path(__file__).resolve().parent.parent
    paths: list[Path] = []
    for pkg in ("core", "utils"):
        pkg_dir = here / "hyperiax" / pkg
        assert pkg_dir.is_dir(), f"expected {pkg}/ at {pkg_dir}"
        paths.extend(pkg_dir.rglob("*.py"))
    return sorted(paths)


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
def test_l1_does_not_import(forbidden: str) -> None:
    """No L1 file (core/ or utils/) may import the forbidden package."""
    offenders: list[str] = []
    for path in _l1_files():
        for line_no, top, is_relative in _imports_in(path):
            if is_relative:
                continue
            if top == forbidden:
                offenders.append(f"{path.relative_to(path.parents[2])}:{line_no}: {top}")
    assert not offenders, f"L1 must not import {forbidden!r}; found:\n" + "\n".join(offenders)


def test_l1_only_imports_allowed_top_level_packages() -> None:
    """Whitelist: every absolute import in L1 (core/ + utils/) resolves to
    ``jax``, ``numpy``, ``hyperiax``, or a stdlib module.
    """
    import importlib.util
    import sys

    allowed_external = {"jax", "numpy", "hyperiax"}
    base = Path(sys.base_prefix).resolve()
    offenders: list[str] = []
    for path in _l1_files():
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
        "L1 (core/ + utils/) may only import jax, numpy, hyperiax, or stdlib; found:\n"
        + "\n".join(offenders)
    )


def test_utils_does_not_import_core_or_prebuilt() -> None:
    """``hyperiax.utils`` is a standalone L1 numerics layer: it must not
    depend on the tree primitives in ``hyperiax.core`` nor on the L2
    ``hyperiax.prebuilt`` package.
    """
    offenders: list[str] = []
    for path in _subpkg_files("utils"):
        src = path.read_text(encoding="utf-8")
        tree = ast.parse(src, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                # Relative import escaping utils (level>=2 climbs to hyperiax.*).
                if node.level >= 2 or module.startswith(("hyperiax.core", "hyperiax.prebuilt")):
                    offenders.append(f"{path.name}:{node.lineno}: {'.' * node.level}{module}")
    assert not offenders, (
        "utils/ must not import from hyperiax.core or hyperiax.prebuilt:\n" + "\n".join(offenders)
    )


# ── L2 layering: prebuilt isolation ────────────────────────────────
def _subpkg_files(subpkg: str) -> list[Path]:
    here = Path(__file__).resolve().parent.parent
    pkg_dir = here / "hyperiax" / subpkg
    return sorted(pkg_dir.rglob("*.py")) if pkg_dir.is_dir() else []


# Optional deps that must stay function-local (lazy) in L2 modules.
LAZY_REQUIRED = ("jax_tqdm",)


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
    inside a ``try / except ImportError`` block.
    """
    paths = _subpkg_files("prebuilt")
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
