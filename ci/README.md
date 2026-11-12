# zi-driver CI

Offline tests of the Zurich Instruments driver of qibolab: LabOne Q emulation, no hardware.
Workflow: `.github/workflows/zi-driver.yml`. Page: https://sensebat.github.io/qibolab/
(`index.html` + `state.json`).

## What runs

A grid of code (rows) and laboneq version (columns). Each cell installs the row's `uv.lock`,
sets laboneq to the column's version (zhinst-* follow, everything else as locked) and runs
`tests/instruments/zi` of the row plus the conformance suite
`tests/instruments/zi/conformance` of sensebat/qibolab@orkun/ci.

|                                                          | pinned (laboneq in R1 `uv.lock`) | latest (newest laboneq on PyPI) |
|----------------------------------------------------------|----------------------------------|---------------------------------|
| **R1** ElStabilini/qibolab@zi_drivers                    | **target**                       | information                     |
| **R2** sensebat/qibolab@orkun/26.10 + R1 merged          | information                      | **target**                      |
| **X** extra row (orkun/proto-*, `extra_ref`), no merge   | information                      | information                     |

- **Target** cells must be green: zi_drivers works on the laboneq it pins, orkun/26.10 works
  on the newest laboneq. No branch has to do both.
- **Information** cells show what a branch would hit on the other version. Red there is
  shown (the job is red) but does not fail the run.
- The **gate** job ("Targets green") fails the run exactly when a target cell of the run is
  not green. The run's conclusion is the answer to "are the targets green".

## Reading the page

- Headline: `Targets: 2 of 2 green`, or which target is not.
- Grid: target cells filled green or red; information cells white with a green or red edge.
  Each cell: passed · not implemented yet · failed, and the installed laboneq and zhinst-core.
- What ran: driver tests, contract tests, effect tests, planted-bug runs per cell.
- Effect checks: one row per feature (`conformance/features.py`), contract and effect result
  per cell. Hover a chip for counts.
- CP1-CP5 reference files: what the driver hands to LabOne Q, compared with
  `conformance/references/<laboneq major.minor>/cpN.json`.
- Failed tests: per cell, first line of each message.

## A new commit on ElStabilini/qibolab@zi_drivers

1. **changes** job: reads the heads of R1, R2 and orkun/ci and the newest laboneq; the key
   (`r1=… r2=… ci=… laboneq=…`) is compared with the key in the published `state.json`.
   Scheduled and zi-check runs stop here when the key is unchanged.
2. **cell** jobs: R1 tests the new commit. R2 merges it into orkun/26.10 for this run only
   (nothing is pushed).
3. **report** job: `index.html`, `state.json`, the job summary, artifact `test-report`.
4. **gate** job: target cells green or not.
5. **pages** job: publishes the report (runs on orkun/ci with rows R1 and R2 only).

**R2 merge conflict:** both R2 cells show "merge conflict" with the conflicting files; no
tests run there, the R2 x latest target is red and the run fails. Resolve by merging
ElStabilini/qibolab@zi_drivers into sensebat/qibolab@orkun/26.10 and pushing; that push
runs the R2 cells again.

## Triggers

| Trigger | What it does |
|---|---|
| push to orkun/ci | full run, publishes Pages |
| push to orkun/26.10 | R2 cells of that commit, no Pages |
| push to orkun/proto-* | R1, R2 and the branch as row X, no Pages |
| workflow_dispatch | full run; `only_if_changed` skips like the schedule; `extra_ref` (+ `extra_repo`) adds row X and turns Pages off |
| repository_dispatch `zi-check` | same changes check as the schedule, on orkun/ci |
| schedule `*/30` | changes check every 30 min; does not fire in this fork (below) |

Check for news by hand (runs only when something changed, publishes Pages):

    gh api repos/sensebat/qibolab/dispatches -f event_type=zi-check

**Schedule.** sensebat/qibolab is a fork of ElStabilini/qibolab; the zi-driver workflow is
active, the default branch is orkun/ci, and no scheduled run has ever started. GitHub:
"When a public repository is forked, scheduled workflows are disabled by default."
([Disabling and enabling a workflow](https://docs.github.com/en/actions/how-tos/manage-workflow-runs/disable-and-enable-workflows)).
A schedule also runs only from the workflow file on the default branch and "can be delayed
during periods of high loads"
([Events that trigger workflows: schedule](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)).
The `zi-check` dispatch is the fallback; anything that can call the GitHub API (a cron job,
another workflow with a token) can send it.

**Proto branches.** A push runs the workflow file of the pushed commit, so an orkun/proto-*
branch needs `.github/workflows/zi-driver.yml` from orkun/ci (a branch made from orkun/26.10
has it once orkun/26.10 has merged orkun/ci). The ci scripts and the conformance suite always
come from orkun/ci. Without the file, use the dispatch:

    gh workflow run zi-driver.yml -R sensebat/qibolab --ref orkun/ci -f extra_ref=orkun/proto-NAME

## One cell locally

From a qibolab checkout (the driver) and an orkun/ci checkout (the suite):

    ROW=r1 COL=c2 R1_REPO=ElStabilini/qibolab R1_REF=zi_drivers \
      bash SUITE/ci/run_cell.sh DRIVER SUITE out/cell-r1-c2
    python3 SUITE/ci/report.py out out/report    # out/report/index.html

`ROW=r2` merges R1 into the checkout first (set `R2_REPO`, `R2_REF`); `ROW=x` runs the
checkout as it is (`EXTRA_REPO`, `EXTRA_REF` are labels). `LABONEQ_PINNED` and
`LABONEQ_LATEST` are read from R1's `uv.lock` and PyPI when unset. Exit code = pytest's.

## Not implemented yet

"Not implemented yet" are strict expected failures: a refusal with the exact
`NotImplementedError` message (`NOT_IMPLEMENTED` in `conformance/test_conformance.py`,
`REFUSED` in `conformance/test_checkpoints.py`) or a known LabOne Q rejection (`BROKEN`).
When the driver starts doing such a feature, the test passes, pytest reports
`XPASS(strict)` and the cell turns red. That is the signal to update the suite in
sensebat/qibolab@orkun/ci: drop the entry from the list and add effect tests for the
feature to its `effects` in `conformance/features.py`. A refusal with a changed message also
turns the cell red.

## Old reports

Every run keeps its report: the run's job summary (headline, grid, failed tests, feature
table) and the `test-report` artifact (`index.html`, `state.json`; kept 90 days). The page
shows only the newest orkun/ci run. Runs: https://github.com/sensebat/qibolab/actions/workflows/zi-driver.yml
