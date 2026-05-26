# neurovrai implementation-loop — activity log

Append 1–3 lines per loop iteration (newest at top). See `main-plan.md`.

---

## 2026-05-26 — Loop scaffold established
- Brought package to dev parity with neurofaune: added `[dev]` extras, build-system,
  ruff/black/mypy/pytest config (with `regression` marker), Makefile gate, pre-commit, CI.
- Added `tests/` tree: a baseline unit test + the frozen-contract regression harness.
- gitignore exception so regression fixtures/golden are tracked despite data-file ignores.
- Open: reconcile version (pyproject 0.2.0 vs __init__ 2.0.0-alpha) before first tag.
- `$PKG` hash: <fill on commit>
