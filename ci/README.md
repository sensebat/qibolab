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
  shown (page, job summary, a neutral check) but fails neither the run nor the commit
  status: the test step is red, the cell job green (step-level `continue-on-error`).
- The **gate** job ("Targets green") fails the run exactly when a target cell of the run is
  not green. The run's conclusion is the answer to "are the targets green".

## Test results (checks)

Job `checks` publishes one check per cell on the run's commit, named
`ZI tests R1 x pinned (target)`, `ZI tests R1 x latest (information)`, ...: counts, every
failed test as an annotation with message and traceback, and a job summary. Action:
[EnricoMi/publish-unit-test-result-action](https://github.com/EnricoMi/publish-unit-test-result-action)
v2.26.0, pinned by commit; its Docker image is pinned by digest in its `action.yml`.

| Conclusion of the check | target cell | information cell |
|---|---|---|
| all tests pass (strict xfails hold) | success | success |
| a test fails or errors, or a strict xfail passes | failure (red) | set to **neutral** (grey) |
| no `junit.xml` (merge conflict, install failure) | neutral, "inconclusive"; the cell job and the gate are red | neutral |

The action sets the check to failure on failed tests (`fail_on: test failures`); for an
information cell the next step sets it to neutral with the
[checks API](https://docs.github.com/en/rest/checks/runs#update-a-check-run), so the commit
status shows red only for targets. Up to ci-rc1 the information cell jobs failed
(job-level `continue-on-error`), which keeps the run green but made the commit status red
(`statusCheckRollup` FAILURE on 3f1e1920 with all runs green).

The publishing job has `checks: write`; the cell jobs, which run the tested code, keep the
read-only token.

Why this action and not [dorny/test-reporter](https://github.com/dorny/test-reporter):

| | EnricoMi/publish-unit-test-result-action | dorny/test-reporter |
|---|---|---|
| maintenance | v2.26.0, 2026-11-06 | v3.0.0, 2026-03-21; last commit on main 2026-10-24 |
| pytest JUnit XML | supported | `python-xunit`, "Experimental" ([README](https://github.com/dorny/test-reporter#supported-formats)) |
| check per invocation | `check_name` | `name`; v3 writes a job summary instead of a check unless `use-actions-summary: false` |
| annotations | every failed test | `max-annotations`, at most 50 |
| failed tests without a red check | `fail_on: nothing` (check success) | `fail-on-error: false` (check success) |
| neutral check | only without results | no |
| permissions | `checks: write`; `pull-requests: write` only for PR comments ([README](https://github.com/EnricoMi/publish-unit-test-result-action#permissions)) | `checks: write`, `actions: read` for artifacts ([README](https://github.com/dorny/test-reporter#example)) |
| runtime | Docker image (digest-pinned); `/linux` variant without Docker | Node 24 |

Neither sets neutral for failed tests, hence the extra step. Both create check runs with
the workflow's `GITHUB_TOKEN`; on `pull_request` from a fork that token is read-only and
both need a second workflow on `workflow_run`
([EnricoMi](https://github.com/EnricoMi/publish-unit-test-result-action#support-fork-repositories-and-dependabot-branches),
[dorny](https://github.com/dorny/test-reporter#recommended-setup-for-public-repositories)).
This workflow has no fork pull requests. Dependabot pull requests also get a read-only
token, but the `permissions` key raises it
([Dependabot on GitHub Actions](https://docs.github.com/en/code-security/reference/supply-chain-security/troubleshoot-dependabot/dependabot-on-actions));
job `checks` sets `checks: write`. A check run belongs to a commit, not to a run: "it may not
be associated with the workflow that ran this action"
([EnricoMi, `check_run`](https://github.com/EnricoMi/publish-unit-test-result-action#configuration));
the page links each cell's check directly.

## Reading the page

- Headline: `Targets: 2 of 2 green`, or which target is not.
- Grid: target cells filled green or red; information cells white with a green or red edge.
  Each cell: passed · not implemented yet · failed, the installed laboneq and zhinst-core, and
  a link to the cell's check (test results).
- What ran: driver tests, contract tests, effect tests, fault-injection runs per cell.
- Effect checks: one row per feature (`conformance/features.py`), contract and effect result
  per cell. Hover a chip for counts.
- CP1-CP5 reference files: what the driver hands to LabOne Q, compared with
  `conformance/references/<laboneq major.minor>/cpN.json`.
- Failed tests: in the checks (above), no longer on the page.

**Fault injection**: hand-written driver bugs with a meaning (swept amplitudes ignored, nested
sweeps swapped, delays dropped), each patched into the driver for one test run
(`conformance/mutants.py`, matrix in `conformance/TEETH.md`); unlike mutation-testing tools
such as [mutmut](https://mutmut.readthedocs.io/), which make automatic mechanical code edits.
An effect test with `@kills(...)` runs once on the driver (must pass) and once per named
fault (must fail); the test ids are `[driver]` and `[fault-<name>]` (`[mutant-<name>]` up to
ci-rc1). The code keeps the names `mutants.py`, `MUTANTS` and `@kills`.

## A new commit on ElStabilini/qibolab@zi_drivers

1. **changes** job: reads the heads of R1, R2 and orkun/ci and the newest laboneq; the key
   (`r1=… r2=… ci=… laboneq=…`) is compared with the key in the published `state.json`.
   Scheduled and zi-check runs stop here when the key is unchanged.
2. **cell** jobs: R1 tests the new commit. R2 merges it into orkun/26.10 for this run only
   (nothing is pushed).
3. **checks** jobs: one check per cell (Test results above).
4. **report** job: `index.html`, `state.json`, the job summary, artifact `test-report`.
5. **gate** job: target cells green or not.
6. **pages** job: publishes the report (runs on orkun/ci with rows R1 and R2 only).

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
| pull_request to orkun/26.10 | R2 cells of the pull request's head, no Pages (needs this workflow file on orkun/26.10) |
| workflow_dispatch | full run; `only_if_changed` skips like the schedule; `extra_ref` (+ `extra_repo`) adds row X and turns Pages off |
| repository_dispatch `zi-check` | same changes check as the schedule, on orkun/ci |
| schedule `17,47 * * * *` | changes check twice an hour; fires rarely (below) |

Check for news by hand (runs only when something changed, publishes Pages):

    gh api repos/sensebat/qibolab/dispatches -f event_type=zi-check

**Schedule.** With `*/30 * * * *` (ci-rc1) two scheduled runs started in about 30 slots
(2026-11-12T23:59Z, 2026-11-13T06:26Z). GitHub: "The `schedule` event can be delayed during
periods of high loads of GitHub Actions workflow runs. High load times include the start of
every hour. If the load is sufficiently high enough, some queued jobs may be dropped. To
decrease the chance of delay, schedule your workflow to run at a different time of the
hour." Scheduled workflows run only on the default branch, and "In a public repository,
scheduled workflows are automatically disabled when no repository activity has occurred in
60 days." ([Events that trigger workflows: schedule](https://docs.github.com/en/actions/reference/workflows-and-actions/events-that-trigger-workflows#schedule)).
Hence minutes 17 and 47. The new cron takes effect once this file is on orkun/ci. The
`zi-check` dispatch stays the fallback; anything that can call the GitHub API (a cron job,
another workflow with a token) can send it.

**Proto branches.** A push runs the workflow file of the pushed commit, so an orkun/proto-*
branch needs `.github/workflows/zi-driver.yml` from orkun/ci (a branch made from orkun/26.10
has it once orkun/26.10 has merged orkun/ci). The ci scripts and the conformance suite always
come from orkun/ci, so orkun/ci must be at ci-rc1 or later: with an older `ci/changes.sh`
there is no row X and every cell stops at "Check the cell plan". Without the file, use the
dispatch:

    gh workflow run zi-driver.yml -R sensebat/qibolab --ref orkun/ci -f extra_ref=orkun/proto-NAME

## New laboneq releases

Today: the changes job reads the newest laboneq from PyPI on every scheduled and zi-check
run; a new version changes the key and runs all cells, R2 x latest (target) and R1 x latest
(information) on it. This stays.

Planned addition: a pull request against sensebat/qibolab@orkun/26.10 that raises laboneq
in `pyproject.toml` and `uv.lock`; the workflow runs the R2 cells on its head
(`pull_request` to orkun/26.10), and merging it moves orkun/26.10 to the new laboneq.
Config: `.github/dependabot.yml` (laboneq only, `target-branch: orkun/26.10`,
`versioning-strategy: increase`, daily).

| | Dependabot | Renovate |
|---|---|---|
| install | built into GitHub | Mend Renovate GitHub app, or self-hosted with [renovatebot/github-action](https://github.com/renovatebot/github-action) |
| uv / `uv.lock` | `package-ecosystem: uv`, version updates supported ([supported ecosystems](https://docs.github.com/en/code-security/reference/supply-chain-security/supported-ecosystems-and-repositories)) | `pep621` manager, updates `uv.lock` ([pep621](https://docs.renovatebot.com/modules/manager/pep621/)) |
| other branch | `target-branch`: "Only manifest files on the target branch are checked for version updates. All pull requests for version updates are opened targeting the specified branch." ([options](https://docs.github.com/en/code-security/reference/supply-chain-security/dependabot-options-reference#target-branch-)) | `baseBranchPatterns` ([options](https://docs.renovatebot.com/configuration-options/#basebranchpatterns)) |
| token | none to set up | the action needs a PAT or app token as a secret; with `GITHUB_TOKEN` "workflows run on RenovateBot's PRs will need to be manually approved" and "Allow GitHub Actions to create and approve pull requests" must be on ([action README](https://github.com/renovatebot/github-action#token)) |
| tests on the pull request | runs `pull_request` workflows (read-only token unless `permissions` raises it) | with `GITHUB_TOKEN`: events it triggers "will not create a new workflow run", pull requests wait for approval ([trigger a workflow](https://docs.github.com/en/actions/how-tos/write-workflows/choose-when-workflows-run/trigger-a-workflow#triggering-a-workflow-from-a-workflow)) |

Dependabot, because it needs no app, secret or token, and its pull requests run the
workflow. Renovate would need an app install or a secret, both repository settings.

What Dependabot needs before it does anything (none of it is done on orkun/ci-std):

1. `.github/dependabot.yml` on the default branch (orkun/ci): "You must store this file in
   the `.github` directory of your repository in the default branch"
   ([about dependabot.yml](https://docs.github.com/en/code-security/concepts/supply-chain-security/about-the-dependabot-yml-file)).
   On orkun/ci-std it is inert.
2. Version updates enabled for the fork by hand: "Version updates are not automatically
   enabled on forks when a `dependabot.yml` configuration file is present. [...] On a fork,
   you also need to explicitly enable Dependabot." Settings, Advanced Security, Dependabot
   version updates: Enable
   ([configure version updates](https://docs.github.com/en/code-security/how-tos/secure-your-supply-chain/secure-your-dependencies/configure-version-updates#enabling-version-updates-on-forks)).
3. This `zi-driver.yml` on orkun/26.10 (merge orkun/ci into orkun/26.10): a `pull_request`
   run uses the workflow file of the pull request's merge commit, so the trigger and the
   pull-request handling in `ci/changes.sh` act only once orkun/26.10 has them.

Not verified until the first pull request: how `versioning-strategy: increase` rewrites
`laboneq>=26.10,<26.11` in the `zi` dependency group. The PyPI poll in `ci/changes.sh`
stays: it also drives R1 x latest, the page, and runs between Dependabot's daily checks.

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

Every run keeps its report: the run's job summary (headline, grid, feature table, and one
summary per cell from job `checks`), the checks on the commit and the `test-report` artifact (`index.html`, `state.json`; kept 90 days). The page
shows only the newest orkun/ci run. Runs: https://github.com/sensebat/qibolab/actions/workflows/zi-driver.yml
