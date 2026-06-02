"""Sphinx configuration for the hyperiax docs.

Builds with ``make -C docs html`` (locally) or ``sphinx-build -W docs/source
docs/build/html``. Notebook outputs are rendered as-saved — install the
``[notebook]`` extra and re-run the .ipynb locally to refresh them.
"""

from __future__ import annotations

import importlib.metadata as _md

# ── project info ────────────────────────────────────────────────────
project = "hyperiax"
author = "The hyperiax authors"
copyright = "2026, hyperiax authors"

try:
    release = _md.version("hyperiax") or "0.0.0"
except _md.PackageNotFoundError:
    release = "0.0.0"
version = ".".join(release.split(".")[:2])

# ── extensions ──────────────────────────────────────────────────────
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",  # Google / NumPy docstring styles
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "sphinx_autodoc_typehints",  # render type hints inline
    "sphinx_copybutton",  # copy-to-clipboard on code blocks
    "myst_nb",  # MyST markdown + notebook rendering
]

autosummary_generate = True
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "inherited-members": False,
    "show-inheritance": True,
    "undoc-members": False,
}
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_rtype = False

# ── myst-nb: render saved outputs, do not execute ──────────────────
nb_execution_mode = "off"
myst_enable_extensions = [
    "amsmath",
    "dollarmath",
    "deflist",
    "colon_fence",
]

# ── intersphinx ────────────────────────────────────────────────────
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "jax": ("https://jax.readthedocs.io/en/latest/", None),
}

# ── theme ──────────────────────────────────────────────────────────
html_theme = "furo"
html_title = f"hyperiax {version}"
html_static_path = ["_static"]
html_theme_options = {
    "source_repository": "https://github.com/computationalevolutionarymorphometry/hyperiax",
    "source_branch": "v3",
    "source_directory": "docs/source/",
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/computationalevolutionarymorphometry/hyperiax",
            "html": "",
            "class": "fa-brands fa-github",
        },
    ],
}

# ── misc ───────────────────────────────────────────────────────────
exclude_patterns = ["_build", "**.ipynb_checkpoints"]
templates_path = ["_templates"]
suppress_warnings = ["mystnb.unknown_mime_type"]
