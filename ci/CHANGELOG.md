# zi-driver CI changelog

## ci-rc2 (draft)

On sensebat/qibolab@orkun/ci-std; not on orkun/ci yet.

- Test results: job `checks` publishes one check per cell, `ZI tests <cell> (<role>)`, with
  every failed test as an annotation (EnricoMi/publish-unit-test-result-action v2.26.0,
  pinned by commit). A target check fails on a failed test; an information check is set to
  neutral. Only this job has `checks: write`.
- Commit status: information cells no longer make it red. `continue-on-error` moved from
  the cell job to its test step, so a red information cell has a red step and a green job.
- Page and job summary: the failed-test lists are gone (they are in the checks); each grid
  cell links to its check, the page to all checks of the commit. ci/report.py 852 -> 818
  lines (the fault-injection explanation is new).
- Schedule: `17,47 * * * *` instead of `*/30 * * * *`, off the start of the hour (GitHub
  docs: delays and dropped runs at high load). zi-check and the skip-if-unchanged key stay.
- New laboneq releases: `.github/dependabot.yml` (uv, laboneq only, pull requests against
  orkun/26.10). Inert until it is on orkun/ci and version updates are enabled for the fork;
  ci/README.md lists the steps. `pull_request` to orkun/26.10 runs the R2 cells of the pull
  request's head (acts once orkun/26.10 has this workflow file). The PyPI poll stays.
- Naming: "planted bugs" / "mutants" are now "fault injection" / "injected faults" on the
  page, in the job summary, ci/README.md, TEETH.md and the suite's docstrings and messages.
  `@kills` test ids are `[fault-<name>]` (were `[mutant-<name>]`); ci/report.py reads both.
  `mutants.py`, `MUTANTS`, `@kills` and `--mutant` keep their names.

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
- Cells stop at "Check the cell plan" when the plan has no repo or commit (ci scripts on
  orkun/ci older than the workflow file), instead of testing the run's own commit under the
  row's label.
- Docs: ci/README.md.
