# CLAUDE.md

Guidance for Claude Code (and any agent) working in this repo. Treat
this as ground truth before guessing.

## What this is

`hyperiax` is a **pure-JAX library for tree message passing**. Users
build a `Topology` (rooted tree structure), wrap it in an immutable
`Tree` (a JAX pytree carrying typed per-node arrays), and run sweeps —
`Tree -> Tree` functions written with `@hx.up` / `@hx.down`
decorators. Everything composes with `@jax.jit` and `jax.lax.scan`.

The current branch is `v3`, a ground-up rewrite of the legacy
`main`-branch 0.x. Do **not** look to `main` or to deleted
`examples/*.py` files for guidance — that code is dead. The v3 design
is captured by `pyproject.toml`, the docs under `docs/source/`, and
the tutorial notebooks `docs/source/notebooks/01..07`.

## Architecture (L1 / L2 layering — enforced by tests)

```
hyperiax/
├── core/            # L1 — Topology, Tree, Schema, views, sweep, dispatch, builders (incl. Newick), errors
├── utils/           # L1 — pure-JAX ODE / SDE solvers (no optional deps)
└── prebuilt/        # L2 — bffg, phylo_mean
```

Rules, enforced by `tests/test_core_dependencies.py`:

- **`core/` must not import** anything outside `{jax, numpy, hyperiax, stdlib}`. No matplotlib, scipy, ete3, diffrax, jax_tqdm, etc. The Newick parser is a small pure-Python implementation in `core/builders.py`.
- **`prebuilt/` modules must not import each other.**
- **Optional deps must be lazy** (function-local `import`, or top-level inside `try/except ImportError`). `import hyperiax` must work on a bare `jax + numpy` install.

Optional extras / groups and what they unlock:

| Kind | Name | Adds | Used by |
|---|---|---|---|
| extra | `notebook` | matplotlib, jupyter, joblib, optax, numpyro | `docs/source/notebooks/*.ipynb` (incl. NumPyro-driven BFFG-MCMC notebooks) |
| group | `dev` | pytest, ruff, pre-commit, sphinx + theme | test, lint, docs build |

The core library has no optional runtime deps — `jax + numpy` is enough to
import everything in `hyperiax/`, `hyperiax/core/`, `hyperiax/utils/`,
and `hyperiax/prebuilt/`.

Install patterns:
- contributor: `uv sync --group dev` (test + lint + docs)
- contributor + everything: `uv sync --all-extras --group dev`
- end user: `uv pip install 'hyperiax'`

## Hard invariants (don't regress these)

1. **`Tree` is immutable.** Mutators (`set`, `set_at`, `update`, `drop`) all return new `Tree` instances. Do **not** reintroduce `tree.data = {...}`. The v2 in-place pattern was the source of leaked-tracer bugs that motivated v3 — reverting it breaks `@jax.jit`-wrapped train steps in user code.
2. **`Topology` derived fields are `numpy` arrays, not `jax` arrays.** They ride through `jax.jit` as static aux_data. Promoting them to `jnp` would make `Topology` unhashable and break the JIT cache.
3. **`ChildrenAxis._num_segments` must be a Python `int`** (static), not a tracer — otherwise `jax.ops.segment_*` output shape becomes polymorphic and breaks JIT.
4. **Sweeps declare `reads` / `writes` explicitly.** No reflection / no inferring from function signature.
5. **Schema field names are validated against `_RESERVED_FIELD_NAMES`** in `core/tree.py` to keep attribute access (`tree.value`) unambiguous.
6. **One up-sweep dispatch path; `children.X` is always a `ChildrenAxis` proxy.** Any topology — equal- or unequal-degree — routes through `_up_dispatch` (segment-based). The proxy exposes pure reductions (`.sum/.mean/.max/.min/.prod(axis=0)` → `jax.ops.segment_*`) and a segment-preserving per-child vmap (`children.map(fn)` — `fn` receives a per-child `Node` view and returns a `dict[str, Array]`). Direct array ops (`children.X[0]`, `children.X * 2`, `children.X.shape`) are deliberately rejected: the dense vmap-over-parents path has been removed. Any non-reduction per-child work goes through `children.map`. `writes_children` outputs come back as `ChildrenAxis` and scatter to the contiguous child block.

## Common commands

```bash
make sync                  # uv sync --all-extras --group dev
make test                  # uv run pytest

# Sphinx (docs deps are in the dev group)
make -C docs html          # build HTML to docs/build/html
make -C docs clean         # nuke build + autosummary stubs
make -C docs open          # open in browser (macOS)
make -C docs html SPHINXOPTS=-W   # strict mode (warnings as errors)

# Lint / format (ruff config in pyproject)
uv run ruff check .
uv run ruff format .
uv run pre-commit run --all-files
```

CI runs `uv sync --all-extras --group dev && uv run pytest` on push/PR
to `main` and `v3` (`.github/workflows/pytest.yaml`). Releases trigger
PyPI publish (`.github/workflows/pypi_push.yaml`). **No `build_doc`
workflow yet** — sphinx is currently local-only.

## Tests

164 tests across 12 files in `tests/`, all passing on a clean checkout.
Run subsets with `uv run pytest tests/test_<thing>.py -x -q`. Key suites
worth knowing:

- `test_core_dependencies.py` — **architectural lint**. Enforces L1/L2 layering and the lazy-optional-import rule.
- `test_topology.py` / `test_tree.py` / `test_schema.py` / `test_sweep_*` — core API.
- `test_sweep_up.py` — up-sweep dispatch on both equal- and unequal-degree trees in one file (the single segment-based path makes them indistinguishable to user code; tests cover reductions, `children.map`, `writes_children`, ChildrenAxis proxy guards, jit cache + outer-jit on both topologies).
- `test_prebuilt_phylo_mean.py` — the only prebuilt with dedicated tests; the BFFG sweeps are covered by the core dispatch tests (a stable BFFG-specific test suite is pending the new public surface).
- `test_regression.py` — leaked-tracer regression, jit + scan composition.

## Docs

```
docs/
├── Makefile          # sphinx commands
├── source/
│   ├── conf.py       # furo theme, myst-nb (execution=off), autosummary
│   ├── index.rst     # landing page + toctrees
│   ├── notebooks/    # tutorials 01–07 (.ipynb)
│   └── api/index.rst # autosummary entry points per module
└── build/            # gitignored
```

**Notebook policy:** outputs are rendered as-saved (`nb_execution_mode = "off"`).
If you change a notebook, re-run it locally and commit the new outputs.
CI does **not** execute notebooks during the docs build (deliberate —
notebooks 05/07 run multi-chain MCMC and take 10+ minutes).

When adding a new prebuilt or public symbol, list it in
`docs/source/api/index.rst`. The matching stub page is generated by
`sphinx-autosummary` on the next `make -C docs html`.

## Coding conventions

- **Docstring style:** Google / Napoleon (`Args:` / `Returns:` blocks). Sphinx is configured for it.
- **No migration narration in docstrings.** Statements like "ported verbatim from the legacy `examples/...`" or "Stage N adds Y" are stale — they were stripped in commit `a6cebd1`. Don't add them back.
- **No emojis in code or docstrings** unless explicitly requested.
- **No backwards-compatibility shims.** v3 is on its own branch and there is no compatibility surface to preserve.
- **Comments explain *why*, not *what*.** The codebase favors short comments over verbose narration. Default to no comment.

## Things that recently changed (don't re-introduce)

Cleanups from earlier passes (still in effect):

- `StructureMismatch` exception removed (raised nowhere). Use `SchemaMismatch` for shape/dtype mismatches and `MissingField` for unknown field names.
- `Parent` view class is **not** re-exported at top level (`hyperiax.Parent` doesn't exist). It's still in `hyperiax.core.views` for the dispatcher's internal use, but users don't see it.
- `Schema.empty()` removed — use `Schema.from_dict({})`.
- `MHState.__repr__` doesn't call `float()` on a jax array — that broke under jit/vmap. Default dataclass repr now.

BFFG rewrite (`prebuilt/bffg.py` — `gaussian_*` → `discrete_*`, `sde_*` → `continuous_*`):

- Public names are `discrete_schema` / `continuous_schema`, `init_discrete_tree` / `init_continuous_tree`, `discrete_bf_sweep` / `discrete_forward_sweep` / `discrete_fg_sweep`, `continuous_bf_sweep` / `continuous_forward_sweep` / `continuous_fg_sweep`, plus `propagate_linearization`. The old `gaussian_*` / `sde_*` names are gone — don't bring them back.
- `_discrete_backward_filtering` applies the full linear-Gaussian auxiliary `(Φx + β, Q)` per Theorem 14 §6.1: `prec_msg = Φᵀ (I + HQ)⁻¹ H Φ`, `ptnl_msg = Φᵀ (I + HQ)⁻¹ (F − Hβ)`. The earlier version silently dropped `Φ` and `β`.
- `_discrete_forward_guiding` weight matches Theorem 14.3 (eval point `H⁻¹F`; numerator uses true `μ(x)` + `Q_true(x) + H⁻¹`; denominator uses auxiliary `Φx + β` + `Q_aux + H⁻¹`). `discrete_fg_sweep` therefore takes the proxy `(prxy_scale_fn, prxy_shift_fn, prxy_covar_fn)` as well as the true `(mean_fn, covar_fn)`.
- `continuous_bf_sweep` caches the per-edge `(H, F)` trajectory on each child via `writes_children=("precs","ptnls")`; the fused vertex message lives in new schema fields `prec_v` / `ptnl_v` and becomes the terminal condition for the parent's own edge one level up. `_continuous_bf_ode` uses `ã = σ̃ σ̃ᵀ` (the diffusion squared, not used raw); `_continuous_bf_anlt` is the closed-form path when `prxy_scale_fn = prxy_shift_fn = None`. `prxy_diffusion_fn(u, params)` returns σ̃(u) (time argument).
- `_continuous_forward_guiding` weight is the Theorem 23 / Remark 24 integrand `(b − b̃)·r − ½ tr((a − ã)H) + ½ r'(a − ã)r`; a `None`-guard treats the `B = β = 0` auxiliary as driftless.

**Iterative linearisation (Algorithm 3 §7.1).** Both schemas carry a per-node `anchor: (d,)` field; `continuous_schema` additionally carries `anchor_pa: (d,)` (the parent-end anchor, stored on each child so `children.map(...)` can read both endpoints without segment-aware gather). All `prxy_*_fn` callables take `anchor` as a positional argument before `params` (continuous: also after `t`). Linear-Gaussian models like 05/06 ignore it; nonlinear models like 07's shape SDE evaluate the auxiliary's `σ̃` at it.

`continuous_bf_sweep` uses two anchors per edge — `anchor_pa` at `t=0` and `anchor` at `t=T` — with `ã(t)` **linearly interpolated** between `σ̃(anchor_pa)σ̃ᵀ` and `σ̃(anchor)σ̃ᵀ`. The closed-form anlt path integrates this analytically; the ODE path passes `anchor_at(t) = (1-t/T)·anchor_pa + (t/T)·anchor` into the user's `prxy_*_fn`. The matching `continuous_fg_sweep` rebuilds the same interpolation for the bridge step's log-weight increment, so `a − ã` stays consistent across BF and FG.

`discrete_refine_anchor()` / `continuous_refine_anchor()` are `@down` sweeps that overwrite `anchor` with the BFFG posterior mean `prec⁻¹ ptnl` (continuous: `prec_v⁻¹ ptnl_v`) and `anchor_pa` with the parent's just-refined anchor. The user iterates `bf_sweep → refine_anchor` 3–10 times before running `fg_sweep` — 07 demonstrates this. `init_discrete_tree` / `init_continuous_tree` accept an optional `anchor_init` kwarg (default = `root_val`) and additionally pin the LEAF anchors to `leaf_obs` so the first BF iteration already has correct leaf-end anchors.

When the user pins the root at a non-zero value, the BFFG-implied marginal log-likelihood is **`log h(x_root) = log_norm + ptnl_v · x_root - ½ x_root' prec_v x_root`** at the root — not `log_norm` alone (`log_norm` is the canonical constant `c`, not `log h` at a non-zero pinned point). For 05/06 with root pinned at 0 this collapses to `log_norm`; for 07's shape SDE pinned at `ROOT_SHAPE` you MUST add the linear and quadratic terms — see `bffg_guided_forward` in 07.

Dispatch unification:

- **One up-sweep dispatch path.** `_up_dispatch_equal` is gone; `up_dispatch` always calls the single segment-based `_up_dispatch`. Equal-degree trees are just the special case where every segment has the same size. Performance is comparable or better on non-tiny trees.
- `Topology.gather_child_idx` and `Topology.level_non_leaf_indices` have been removed (they only fed the deleted equal-degree path). Don't reintroduce.
- `Children.map(fn)` is the new primitive for per-child non-reduction work. Existing pure-reduction idioms (`children.X.sum(0)` etc.) are unchanged.
- `prebuilt/phylo_mean.py` migrated to `children.map` and now runs on **unequal-degree** trees too (the old "raises on ragged" test flipped to a correctness test).

`hyperiax/prebuilt/__init__.py` does not re-export BFFG names — the new public surface is in flux. Import via `from hyperiax.prebuilt.bffg import discrete_bf_sweep` (etc.) for now.

## Scope and future directions

- **Rooted directed trees only — DAGs are a known future extension.** `Topology.from_parents` takes a 1-D `parents` array (one parent per node) with BFS ordering (`parents[i] < i`); `_down_dispatch` exploits this single-parent *function* to gather each node's parent into a rectangular `(n_at_level, *trailing)` block, so `Parent` stays a plain `_FieldsView` and no segment ops are needed in the down direction. Extending to general DAGs (multi-parent nodes) would mean:
  - `Parent` becomes a segment-aware proxy, mirroring today's `Children`;
  - `_down_dispatch` grows the same flat-`(M_total, *trailing)` + `segment_*` machinery the up-sweep now uses;
  - the BFS layout and `pbuckets` / `pbuckets_ref` derivation in `_build_topology` need to handle multi-parent fan-in (`parents[i] < i` no longer holds verbatim — needs a proper topological sort).
  Out of scope for now; the current API is tree-specific by design.

## Reference: BFFG paper

Most of the math in `prebuilt/bffg.py` comes from:

> van der Meulen, F. H. & Sommer, S. (2025). *Backward Filtering
> Forward Guiding.* JMLR 26(281), 1–51.
> https://arxiv.org/abs/2505.18239

Key sections:
- §3 — canonical-form propagation (backward information filter).
- §6.1 + Theorem 14 — discrete nonlinear-Gaussian kernels with linear auxiliary `(Φx + β, Q)`. Matches `discrete_bf_sweep` (pullback + fusion) and `discrete_fg_sweep` (guided proposal + log-weight `w(x)`).
- §7.1 + Theorem 23 + Remark 24 — continuous (SDE) edges with linear auxiliary `dX̃ = (B(u)X̃ + β(u))du + σ̃(u)dW`. Matches `continuous_bf_sweep` (eq 29 backward ODE for `(H, F)`; closed form when `B = β = 0`) and `continuous_fg_sweep` (guided SDE eq 31; weight eq 32 = ∫ `(L − L̃)g / g` via the Remark 24 integrand).

## Style for working in this repo

- **`uv run` everything.** Don't `python -m pytest` or `pip install`.
- **Branch is `v3`.** PRs target `v3`, not `main`, until v3 stabilizes.
- **Commit messages: imperative, concise.** Match the style of recent `git log` — e.g. `dispatch: collapse equal/unequal up paths into one` rather than `Fixed the dispatcher`.
- **Don't `--no-verify` commit.** Pre-commit hooks run ruff (lint + format) and a few stock checks; install once with `uv run pre-commit install`.
- **`uv.lock` is committed.** If you change `pyproject.toml`, run `uv sync` so the lockfile updates; commit both.
- **Versioning is via git tag.** `setuptools-scm` derives `__version__` from the closest `vX.Y.Z` tag; the v3 line starts at tag `v3.0.0`.
