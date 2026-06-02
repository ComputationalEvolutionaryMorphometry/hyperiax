# Minimal sphinx Makefile.
#
# Build with:   make html
# Clean with:   make clean
# Strict mode:  make html SPHINXOPTS=-W           (warnings as errors)
# Open result:  make open                          (macOS)

SPHINXOPTS    ?=
SPHINXBUILD   ?= uv run sphinx-build
SOURCEDIR     = source
BUILDDIR      = build

.PHONY: help html clean linkcheck open _unhide-pth

# uv on macOS sets the UF_HIDDEN flag on `_`-prefixed files in site-packages
# (including the editable install's `__editable__.<pkg>....pth`). CPython's
# site.py silently skips hidden .pth files, so the editable finder never
# registers and autosummary fails with `ModuleNotFoundError: No module named
# 'hyperiax'`. Clear the flag before invoking sphinx.
_unhide-pth:
	@if [ "$$(uname)" = "Darwin" ]; then \
		SITE="$$(uv run python -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)"; \
		[ -n "$$SITE" ] && chflags nohidden "$$SITE"/*.pth 2>/dev/null || true; \
	fi

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS)

html: _unhide-pth
	@$(SPHINXBUILD) -b html "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS)

clean:
	rm -rf "$(BUILDDIR)" "$(SOURCEDIR)/api/generated"

linkcheck: _unhide-pth
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS)

open:
	@open "$(BUILDDIR)/html/index.html"
