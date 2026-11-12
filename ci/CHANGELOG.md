# zi-driver CI changelog

## ci-rc1 (2026-11-12)

First release candidate of the zi-driver CI on sensebat/qibolab.

- Grid of code x laboneq version: R1 ElStabilini/qibolab@zi_drivers, R2
  sensebat/qibolab@orkun/26.10 + R1 merged per run; columns pinned (R1 `uv.lock`) and latest
  (PyPI).
- Targets: R1 x pinned and R2 x latest must be green; the other two cells are information.
  The page and the job summary open with `Targets: n of m green`; information cells are
  drawn secondary.
- Gate job: the run fails exactly when a target cell of the run is not green; red
  information cells stay red but do not fail the run (`continue-on-error`).
- Extra row X: a push to orkun/proto-* or the `extra_ref` dispatch input runs that code in
  both columns without a merge, with the job summary and the `test-report` artifact, never
  Pages.
- `repository_dispatch` type `zi-check`: the schedule's changes check on demand. The
  `*/30` schedule stays; GitHub disables scheduled workflows in forks by default
  (ci/README.md).
- Job summary: headline, grid, failed tests and the feature table.
- Driver branches (orkun/26.10, orkun/proto-*) use the ci scripts of orkun/ci.
- Skipped runs show one placeholder cell job "Cells: skipped, nothing changed" instead of
  "matrix.name".
- Conformance suite: contract and effect tests per feature, planted-bug runs, CP1-CP5
  reference files (tests/instruments/zi/conformance).
- Docs: ci/README.md.
