"""Build the zi-driver report (index.html) and state.json from the cells.

Usage: python3 ci/report.py CELLS_DIR OUT_DIR [--summary MARKDOWN_FILE] [--when TEXT]
                            [--checks CHECKS_DIR]

CELLS_DIR holds one folder per cell (cell-r1-c1 ... cell-r2-c2, cell-x-c1, cell-x-c2 for an
extra row) with meta.json and, when the tests ran, junit.xml (written by ci/run_cell.sh).
OUT_DIR gets index.html and state.json. Run facts come from the environment (set by the
workflow; the cells' meta.json fills gaps):
  KEY, RUN_URL, CHECKS_URL, MATRIX, R1_SHA, R2_SHA, CI_SHA, LABONEQ_PINNED, LABONEQ_LATEST,
  R1_REPO, R1_REF, R2_REPO, R2_REF, SUITE_REPO, SUITE_REF, EXTRA_REPO, EXTRA_REF, EXTRA_SHA
Target cells (R1 x pinned, R2 x latest) must be green; state.json["targets"] holds their
results for ci/gate.py. --summary appends the headline, the grid and the feature table as
Markdown (for $GITHUB_STEP_SUMMARY). --checks holds check-<cell>/url.txt, the test-result
check of each cell (EnricoMi/publish-unit-test-result-action, zi-driver workflow job
"checks"); the grid links there. Failed tests with message and traceback are listed in those
checks, not here. Standard library only.
"""

import argparse
import html
import json
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path

BASE_ROWS = ["r1", "r2"]
EXTRA_ROW = "x"
# Must be green: zi_drivers on the laboneq it pins, orkun/26.10 on the newest laboneq.
TARGETS = ["r1-c1", "r2-c2"]
COLS = ["c1", "c2"]
OUTCOMES = ["passed", "xfailed", "skipped", "failed", "error"]
CONFORMANCE = "conformance"


# ---------------------------------------------------------------- reading


def outcome(case: ET.Element) -> tuple[str, str]:
    for tag in ("failure", "error"):
        node = case.find(tag)
        if node is not None:
            return ("failed" if tag == "failure" else "error"), node.get("message", "")
    node = case.find("skipped")
    if node is not None:
        return ("xfailed" if node.get("type") == "pytest.xfail" else "skipped"), node.get("message", "")
    return "passed", ""


def one_line(message: str, limit: int = 220) -> str:
    line = next((s.strip() for s in message.splitlines() if s.strip()), "")
    return line if len(line) <= limit else line[: limit - 1] + "…"


def read_junit(path: Path) -> list[dict]:
    cases = []
    for case in ET.parse(path).getroot().iter("testcase"):
        classname = case.get("classname", "")
        result, message = outcome(case)
        cases.append(
            {
                "name": case.get("name", ""),
                "file": classname.rsplit(".", 1)[-1],
                "conformance": f".{CONFORMANCE}." in f".{classname}.",
                "outcome": result,
                "message": one_line(message),
                "props": {p.get("name"): p.get("value") for p in case.iter("property")},
            }
        )
    return cases


def read_cell(folder: Path) -> dict | None:
    meta_file = folder / "meta.json"
    if not meta_file.exists():
        return None
    meta = json.loads(meta_file.read_text())
    junit = folder / "junit.xml"
    return {"meta": meta, "cases": read_junit(junit) if junit.exists() else None}


def counts(cases: list[dict]) -> dict[str, int]:
    n = {o: 0 for o in OUTCOMES}
    for case in cases:
        n[case["outcome"]] += 1
    return n


# ---------------------------------------------------------------- run facts


class Run:
    def __init__(self, cells: dict[str, dict], when: str, checks: dict[str, str] | None = None):
        e = os.environ
        metas = [c["meta"] for c in cells.values()]

        def first(field: str, env: str, default: str = "") -> str:
            if e.get(env):
                return e[env]
            return next((m[field] for m in metas if m.get(field)), default)

        self.cells = cells
        self.when = when
        self.checks = checks or {}
        self.run_url = e.get("RUN_URL", "")
        self.checks_url = e.get("CHECKS_URL", "")
        self.key = e.get("KEY", "")
        self.r1_repo = e.get("R1_REPO", "ElStabilini/qibolab")
        self.r1_ref = e.get("R1_REF", "zi_drivers")
        self.r2_repo = e.get("R2_REPO", "sensebat/qibolab")
        self.r2_ref = e.get("R2_REF", "orkun/26.10")
        self.suite_repo = e.get("SUITE_REPO", "sensebat/qibolab")
        self.suite_ref = e.get("SUITE_REF", "orkun/ci")
        self.pinned = first("laboneq_pinned", "LABONEQ_PINNED")
        self.latest = first("laboneq_latest", "LABONEQ_LATEST")
        self.ci_sha = first("suite_sha", "CI_SHA")
        r1_from_cells = next((m["sha"] for m in metas if m.get("row") == "r1"), "") or next(
            (m["merged_sha"] for m in metas if m.get("merged_sha")), ""
        )
        self.r1_sha = e.get("R1_SHA") or r1_from_cells
        self.r2_sha = e.get("R2_SHA") or next((m["sha"] for m in metas if m.get("row") == "r2"), "")
        extra = next((m for m in metas if m.get("row") == EXTRA_ROW), {})
        self.extra_repo = e.get("EXTRA_REPO") or extra.get("repo", "")
        self.extra_ref = e.get("EXTRA_REF") or extra.get("ref", "")
        self.extra_sha = e.get("EXTRA_SHA") or extra.get("sha", "")
        if e.get("MATRIX"):
            self.planned = [c["cell"] for c in json.loads(e["MATRIX"])["include"]]
        else:
            self.planned = list(cells)
        has_extra = bool(self.extra_ref) or any(c.startswith(EXTRA_ROW + "-") for c in self.planned)
        self.rows = BASE_ROWS + ([EXTRA_ROW] if has_extra else [])
        self.cell_ids = [f"{r}-{c}" for r in self.rows for c in COLS]

    def targets(self) -> dict:
        """Results of the target cells this run planned; ok when all are green."""
        results = {cid: cell_state(self, cid)["result"] for cid in TARGETS if cid in self.planned}
        green = sum(r == "green" for r in results.values())
        headline = f"Targets: {green} of {len(results)} green"
        red = [f"{label(cid)} {r}" for cid, r in results.items() if r != "green"]
        if red:
            headline += " (" + ", ".join(red) + ")"
        return {"cells": results, "green": green, "total": len(results), "ok": green == len(results),
                "headline": headline}

    def merge_note(self) -> str:
        return next(
            (c["meta"].get("merge_note", "") for c in self.cells.values() if c["meta"].get("merge_note")), ""
        )


def short(sha: str) -> str:
    return (sha or "")[:8]


def esc(value) -> str:
    return html.escape(str(value))


def minor(version: str) -> str:
    return ".".join(version.split(".")[:2]) if version else "?"


def commit_link(repo: str, sha: str) -> str:
    if not sha:
        return "(?)"
    return f'(<a href="https://github.com/{esc(repo)}/commit/{esc(sha)}">{esc(short(sha))}</a>)'


def label(cell_id: str) -> str:
    """r2-c2 -> R2 x latest."""
    row, col = cell_id.split("-")
    return f"{row.upper()} x {'pinned' if col == 'c1' else 'latest'}"


def role(cell_id: str) -> str:
    return "target" if cell_id in TARGETS else "information"


def row_text(run: Run, row: str) -> str:
    r1 = f"{run.r1_repo}@{run.r1_ref} ({short(run.r1_sha)})"
    if row == "r1":
        return r1
    if row == EXTRA_ROW:
        return f"{run.extra_repo}@{run.extra_ref} ({short(run.extra_sha)}), extra row, no merge"
    return f"{run.r2_repo}@{run.r2_ref} ({short(run.r2_sha)}) + {r1} merged"


def row_html(run: Run, row: str) -> str:
    r1 = f"<span class=mono>{esc(run.r1_repo)}@{esc(run.r1_ref)}</span> {commit_link(run.r1_repo, run.r1_sha)}"
    if row == "r1":
        return r1
    if row == EXTRA_ROW:
        return (
            f"<span class=mono>{esc(run.extra_repo)}@{esc(run.extra_ref)}</span> "
            f"{commit_link(run.extra_repo, run.extra_sha)}<div class=small>extra row, no merge</div>"
        )
    note = run.merge_note()
    return (
        f"<span class=mono>{esc(run.r2_repo)}@{esc(run.r2_ref)}</span> {commit_link(run.r2_repo, run.r2_sha)}"
        f" + {r1} merged" + (f"<div class=small>{esc(note)}</div>" if note else "")
    )


COL_TITLE = {"c1": "laboneq {minor} (pinned)", "c2": "laboneq latest"}


def col_title(run: Run, col: str) -> str:
    return COL_TITLE[col].format(minor=minor(run.pinned))


def col_version(run: Run, col: str) -> str:
    return run.pinned if col == "c1" else run.latest


# ---------------------------------------------------------------- cell verdicts


def cell_state(run: Run, cell_id: str) -> dict:
    """result: green | red | merge conflict | no result | not run; text for the grid."""
    cell = run.cells.get(cell_id)
    if cell is None:
        if cell_id in run.planned:
            return {"result": "no result", "text": "no result (job stopped before writing results)"}
        return {"result": "not run", "text": "not run in this run"}
    meta, cases = cell["meta"], cell["cases"]
    if meta.get("stage") == "merge conflict":
        return {"result": "merge conflict", "text": "merge conflict", "conflicts": meta.get("conflicts", [])}
    if cases is None:
        return {"result": "red", "text": meta.get("stage") or "no test results"}
    n = counts(cases)
    failed = n["failed"] + n["error"]
    text = f"{n['passed']} passed · {n['xfailed']} not implemented yet · {failed} failed"
    if n["skipped"]:
        text += f" · {n['skipped']} skipped"
    ok = failed == 0 and str(meta.get("pytest_exit", "0")) in ("0", "")
    if not failed and not ok:
        text += f" · pytest exit {meta['pytest_exit']}"
    return {
        "result": "green" if ok else "red",
        "text": text,
        "passed": n["passed"],
        "not_implemented_yet": n["xfailed"],
        "failed": failed,
        "skipped": n["skipped"],
    }


CSS_CLASS = {"green": "good", "red": "bad", "merge conflict": "bad", "no result": "bad", "not run": "none"}


# ---------------------------------------------------------------- page


def grid(run: Run) -> str:
    head = "".join(
        f"<th scope=col>{esc(col_title(run, c))}<div class='small mono'>{esc(col_version(run, c) or '-')}</div></th>"
        for c in COLS
    )
    body = []
    for row in run.rows:
        tds = []
        for col in COLS:
            cell_id = f"{row}-{col}"
            state = cell_state(run, cell_id)
            meta = run.cells.get(cell_id, {}).get("meta", {})
            detail = ""
            if meta.get("laboneq"):
                detail = f"laboneq {meta['laboneq']} · zhinst-core {meta.get('zhinst_core') or '-'}"
            files = ""
            if state.get("conflicts"):
                files = "<div class='small mono'>" + "<br>".join(esc(f) for f in state["conflicts"]) + "</div>"
            css = f"{CSS_CLASS[state['result']]} {'target' if cell_id in TARGETS else 'info'}"
            check = run.checks.get(cell_id, "")
            text = f"<a href='{esc(check)}'><b>{esc(state['text'])}</b></a>" if check else f"<b>{esc(state['text'])}</b>"
            link = f"<div class=small><a href='{esc(check)}'>test results</a></div>" if check else ""
            tds.append(
                f"<td class='{css}' data-label='{esc(col_title(run, col))}'>"
                f"<div class=role>{role(cell_id)}</div>"
                f"{text}{files}"
                + (f"<div class='small mono'>{esc(detail)}</div>" if detail else "")
                + link
                + "</td>"
            )
        body.append(f"<tr><th scope=row><b>{row.upper()}</b> {row_html(run, row)}</th>{''.join(tds)}</tr>")
    return f"<table class=grid><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def cell_head(run: Run, cell_id: str) -> str:
    row, col = cell_id.split("-")
    which = "pinned" if col == "c1" else "latest"
    mark = " · target" if cell_id in TARGETS else ""
    return (
        f"<th scope=col class='cellhead{' target' if mark else ''}'>{row.upper()} · {which}"
        f"<div class='small mono'>{esc(col_version(run, col) or '-')}{mark}</div></th>"
    )


def ran(run: Run, cell_id: str) -> list[dict] | None:
    """The cell's test cases, or ``None`` when its tests did not run."""
    cell = run.cells.get(cell_id)
    return None if cell is None else cell["cases"]


def is_mutant_case(case: dict) -> bool:
    """A case of an effect test run under an injected driver fault (``[...fault-<name>]``;
    ``mutant-<name>`` in results from ci-rc1)."""
    return re.search(r"(?:^|-)(?:fault|mutant)-", case["name"].partition("[")[2]) is not None


def kind(case: dict) -> str:
    """driver test | contract | effect | fault injection."""
    if not case["conformance"]:
        return "driver test"
    if is_mutant_case(case):
        return "fault injection"
    return case["props"].get("kind", "contract")


def base_name(case: dict) -> str:
    return case["name"].split("[", 1)[0]


# ---------------------------------------------------------------- chips


def tally(cases: list[dict]) -> tuple[str, str, str]:
    """(css, short text, long text) of a group of cases run on the driver."""
    n = counts(cases)
    total, failed = len(cases), n["failed"] + n["error"]
    long = (
        f"{n['passed']} passed, {n['xfailed']} not implemented yet, {failed} failed"
        + (f", {n['skipped']} skipped" if n["skipped"] else "")
    )
    if failed:
        return "bad", f"✗ {failed} of {total}", long
    if n["passed"] == total:
        return "ok", "✓", long
    if n["xfailed"] == total:
        return "warn", "not yet", long
    if n["passed"]:
        return "warn", f"✓ {n['passed']} · not yet {n['xfailed']}", long
    return "none", "skipped", long


def caught(mutants: list[dict]) -> tuple[int, int]:
    """Fault-injection runs that turned the test red (outcome passed), of all."""
    return sum(c["outcome"] == "passed" for c in mutants), len(mutants)


def chip(css: str, label: str, text: str, title: str) -> str:
    return f"<span class='chip {css}' title='{esc(title)}'><b>{esc(label)}</b> {esc(text)}</span>"


def contract_chip(cases: list[dict]) -> str:
    if not cases:
        return chip("none", "contract", "-", "no contract test in this cell")
    css, text, long = tally(cases)
    return chip(css, "contract", text, f"contract tests: {long}")


def effect_chip(cases: list[dict], mutants: list[dict], listed: list[str] | None) -> str:
    if not cases:
        if listed == []:
            return chip("none", "effect", "none yet", "no effect test yet")
        return chip("none", "effect", "absent", "no effect test ran in this cell")
    css, text, long = tally(cases)
    title = f"effect tests: {long}"
    if mutants:
        hit, total = caught(mutants)
        title += f"; injected faults: {hit} of {total} runs red"
        if hit < total and css != "bad":
            css, text = "bad", f"✗ {total - hit} fault{'s' if total - hit > 1 else ''} missed"
    return chip(css, "effect", text, title)


# ---------------------------------------------------------------- what ran


def counts_line(cases: list[dict]) -> str:
    if not cases:
        return "<span class=muted>none</span>"
    n = counts(cases)
    failed = n["failed"] + n["error"]
    parts = [f"{n['passed']} passed"]
    if n["xfailed"]:
        parts.append(f"{n['xfailed']} not yet")
    if n["skipped"]:
        parts.append(f"{n['skipped']} skipped")
    if failed:
        parts.append(f"<b class=o-failed>{failed} failed</b>")
    return " · ".join(parts)


def what_ran(run: Run) -> str:
    groups = [
        ("driver test", "Driver tests", "tests/instruments/zi/*.py at the row's commit"),
        ("contract", "Contract tests", "conformance suite: keys, shapes, refusals"),
        ("effect", "Effect tests", "conformance suite: what LabOne Q gets, incl. CP1-CP5 references"),
        ("fault injection", "Fault-injection runs", "each effect test under the faults it names; must turn red"),
    ]
    body = []
    for key, title, what in groups:
        tds = []
        for cid in run.cell_ids:
            cases = ran(run, cid)
            if cases is None:
                tds.append("<td class=muted>-</td>")
                continue
            of_kind = [c for c in cases if kind(c) == key]
            if key == "fault injection":
                hit, total = caught(of_kind)
                css = "o-passed" if hit == total else "o-failed"
                tds.append(f"<td class={css}>{hit} of {total} red</td>" if total else "<td class=muted>none</td>")
            else:
                tds.append(f"<td>{counts_line(of_kind)}</td>")
        body.append(f"<tr><th scope=row>{esc(title)}<div class='small muted'>{esc(what)}</div></th>{''.join(tds)}</tr>")
    head = "".join(cell_head(run, cid) for cid in run.cell_ids)
    return (
        "<h2>What ran</h2><div class=scroll><table class=counts>"
        f"<thead><tr><th></th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"
    )


# ---------------------------------------------------------------- effect checks


def checkpoint_rank(cp: str) -> tuple[int, str]:
    return (0, cp) if cp.startswith("CP") else (1, cp)


def id_rank(feature_id: str) -> tuple[int, str]:
    digits = "".join(ch for ch in feature_id if ch.isdigit())
    return (int(digits) if digits else 0, feature_id)


def collect_features(run: Run) -> tuple[dict[str, dict], dict[str, dict[str, list]]]:
    """Feature infos by id, and per cell the conformance cases of each feature."""
    features: dict[str, dict] = {}
    per_cell: dict[str, dict[str, list]] = {cid: {} for cid in run.cell_ids}
    for cid in run.cell_ids:
        for case in ran(run, cid) or []:
            props = case["props"]
            if not case["conformance"] or not props.get("feature"):
                continue
            fid = props["feature"]
            info = features.setdefault(
                fid, {"id": fid, "title": props.get("title", ""), "cp": props.get("checkpoint", ""), "effects": None}
            )
            if info["effects"] is None and "effects" in props:
                info["effects"] = props["effects"].split()
            per_cell[cid].setdefault(fid, []).append(case)
    return features, per_cell


def effect_names(listed: list[str] | None) -> str:
    if not listed:
        return ""
    names = "".join(f"<li>{esc(e)}</li>" for e in listed)
    n = len(listed)
    return f"<details><summary>{n} effect test{'s' if n > 1 else ''}</summary><ul class=mono>{names}</ul></details>"


def effect_checks(run: Run) -> str:
    features, per_cell = collect_features(run)
    if not features:
        return "<h2>Effect checks</h2><p class=muted>No conformance results.</p>"
    order = sorted(features, key=lambda f: (checkpoint_rank(features[f]["cp"]), id_rank(f)))
    green = {cid: 0 for cid in run.cell_ids}
    body, last_cp = [], None
    for fid in order:
        info = features[fid]
        tds = []
        for cid in run.cell_ids:
            if ran(run, cid) is None:
                tds.append("<td class=muted>-</td>")
                continue
            cases = per_cell[cid].get(fid, [])
            contract = [c for c in cases if kind(c) == "contract"]
            effect = [c for c in cases if kind(c) == "effect"]
            mutants = [c for c in cases if kind(c) == "fault injection"]
            e_chip = effect_chip(effect, mutants, info["effects"])
            if "chip ok" in e_chip:
                green[cid] += 1
            tds.append(f"<td><div class=chips>{contract_chip(contract)}{e_chip}</div></td>")
        group = " class=group" if last_cp is not None and info["cp"] != last_cp else ""
        last_cp = info["cp"]
        body.append(
            f"<tr{group}><td class='mono id'>{esc(fid)}</td>"
            f"<td>{esc(info['title'])}{effect_names(info['effects'])}</td>"
            f"<td class=mono>{esc(info['cp'] or '-')}</td>{''.join(tds)}</tr>"
        )
    with_effects = sum(1 for f in features.values() if f["effects"])
    foot = "".join(
        f"<td>{green[cid]} of {with_effects}</td>" if ran(run, cid) is not None else "<td class=muted>-</td>"
        for cid in run.cell_ids
    )
    head = "".join(cell_head(run, cid) for cid in run.cell_ids)
    intro = (
        f"<p class=muted>One row per qibolab feature that qibocal needs, ordered by the first checkpoint "
        f"that needs it ({len(features)} features, {with_effects} with effect tests). "
        "<b>contract</b>: the qibolab interface keeps its promises (keys, shapes, refusals). "
        "<b>effect</b>: what the qibolab call makes LabOne Q do, read from the experiment the driver "
        "hands to LabOne Q (amplitudes, frequencies, sweep values, waits, results), and the effect "
        "tests turn red under each injected driver fault they name.</p>"
        "<p class=muted><b>Fault injection</b>: hand-written driver bugs with a meaning (swept "
        "amplitudes ignored, nested sweeps swapped, delays dropped), each patched into the driver "
        "for one test run (tests/instruments/zi/conformance/mutants.py, matrix in TEETH.md); "
        "unlike mutation-testing tools such as mutmut, which make automatic mechanical code "
        "edits.</p>"
        "<p class='legend small'>"
        + chip("ok", "", "✓", "")
        + " all pass "
        + chip("warn", "", "not yet", "")
        + " strict expected failure: refused or not implemented yet "
        + chip("bad", "", "✗", "")
        + " a test failed or an injected fault went unnoticed "
        + chip("none", "", "none yet", "")
        + " no such test. Hover a chip for counts.</p>"
    )
    return (
        f"<h2 id=effects>Effect checks</h2>{intro}<div class=scroll><table class=effects>"
        f"<thead><tr><th class=id>Id</th><th>Feature</th><th>Checkpoint</th>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody>"
        f"<tfoot><tr><td></td><td colspan=2>Features with effect checks green</td>{foot}</tr></tfoot>"
        "</table></div>"
    )


# ---------------------------------------------------------------- CP1-CP5 references


def reference_chip(cases: list[dict]) -> str:
    driver = [c for c in cases if not is_mutant_case(c)]
    mutants = [c for c in cases if is_mutant_case(c)]
    if not driver:
        return chip("none", "reference", "-", "no reference test in this cell")
    outcomes = {c["outcome"] for c in driver}
    message = next((c["message"] for c in driver if c["message"]), "")
    hit, total = caught(mutants)
    bugs = f"; injected faults: {hit} of {total} runs red" if total else ""
    if outcomes & {"failed", "error"}:
        return chip("bad", "reference", "changed", (message or "differs from the reference file") + bugs)
    if outcomes == {"passed"}:
        if hit < total:
            return chip("bad", "reference", f"equal, {total - hit} missed", f"equal to the file{bugs}")
        return chip("ok", "reference", "equal", f"equal to the file{bugs}")
    return chip("none", "reference", "no file", message or "no reference file for this laboneq version")


def references(run: Run) -> str:
    cps: dict[str, dict] = {}
    for cid in run.cell_ids:
        for case in ran(run, cid) or []:
            cp = case["props"].get("checkpoint", "")
            if not case["conformance"] or case["props"].get("feature") or not cp.startswith("CP"):
                continue
            entry = cps.setdefault(cp, {"titles": [], "cells": {c: {"run": [], "ref": []} for c in run.cell_ids}})
            if case["file"] == "test_references":
                entry["cells"][cid]["ref"].append(case)
            else:
                title = base_name(case).removeprefix("test_").replace("_", " ")
                if title not in entry["titles"]:
                    entry["titles"].append(title)
                entry["cells"][cid]["run"].append(case)
    if not cps:
        return "<h2>CP1-CP5 reference files</h2><p class=muted>No checkpoint results.</p>"
    body = []
    for cp in sorted(cps):
        entry = cps[cp]
        tds = []
        for cid in run.cell_ids:
            if ran(run, cid) is None:
                tds.append("<td class=muted>-</td>")
                continue
            cell = entry["cells"][cid]
            run_chip = contract_chip(cell["run"]).replace("<b>contract</b>", "<b>checkpoint</b>", 1)
            tds.append(f"<td><div class=chips>{run_chip}{reference_chip(cell['ref'])}</div></td>")
        body.append(
            f"<tr><td class='mono id'>{esc(cp)}</td><td>{esc(', '.join(entry['titles']) or '-')}</td>{''.join(tds)}</tr>"
        )
    head = "".join(cell_head(run, cid) for cid in run.cell_ids)
    intro = (
        "<p class=muted>Each checkpoint runs the exact qibocal calls on the invented rack. "
        "<b>checkpoint</b>: the calls succeed and the results have the shapes qibocal reads. "
        "<b>reference</b>: everything the driver hands to LabOne Q, written in qibolab terms "
        "(per channel: frequency, LO, offset, range, played amplitudes and phases, waits; sweep values "
        "and what they drive; results), compared with "
        "tests/instruments/zi/conformance/references/&lt;laboneq major.minor&gt;/cpN.json. "
        "<b>equal</b> = no difference; <b>changed</b> = what LabOne Q gets differs (hover for where); "
        "<b>no file</b> = no reference for this laboneq version.</p>"
    )
    return (
        f"<h2 id=references>CP1-CP5 reference files</h2>{intro}<div class=scroll><table class=effects>"
        f"<thead><tr><th class=id>Id</th><th>Checkpoint tests</th>{head}</tr></thead>"
        f"<tbody>{''.join(body)}</tbody></table></div>"
    )


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#ddddd8;--card:#fff;
--good:#e3f2e6;--good-fg:#1e5b2c;--bad:#fbe4e2;--bad-fg:#8a1f17;--warn:#f8efd2;--warn-fg:#6b5300;--none:#f1f1ee}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#171716;--fg:#e8e8e3;
--muted:#9a9a93;--line:#34342f;--card:#1f1f1d;--good:#173322;--good-fg:#9fd9ad;--bad:#3a1a17;
--bad-fg:#f2a49c;--warn:#332b12;--warn-fg:#e8cf7a;--none:#262624}}
:root[data-theme="dark"]{--bg:#171716;--fg:#e8e8e3;--muted:#9a9a93;--line:#34342f;--card:#1f1f1d;
--good:#173322;--good-fg:#9fd9ad;--bad:#3a1a17;--bad-fg:#f2a49c;--warn:#332b12;--warn-fg:#e8cf7a;--none:#262624}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,sans-serif;margin:0;padding:24px 16px}
main{max-width:1040px;margin:0 auto}p,h3{overflow-wrap:anywhere}h1{font-size:22px;margin:0 0 4px}
h2{font-size:18px;margin:32px 0 6px;padding-top:12px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:18px 0 4px}a{color:inherit}.muted{color:var(--muted)}
nav{font-size:13.5px;margin:6px 0 14px;color:var(--muted)}nav a{margin-right:12px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;overflow-wrap:anywhere}
.small{font-size:12.5px;margin-top:2px}
table{border-collapse:collapse;background:var(--card);width:100%}
th,td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}
.grid th[scope=row]{font-weight:400;width:34%}.grid td a{text-decoration:none}
.good{background:var(--good);color:var(--good-fg)}.bad{background:var(--bad);color:var(--bad-fg)}
.none{background:var(--none);color:var(--muted)}
.grid td.info{background:var(--card);color:var(--fg);border-left:5px solid var(--line)}
.grid td.info.good{border-left-color:var(--good-fg)}.grid td.info.bad{border-left-color:var(--bad-fg)}
.grid td.info b{font-weight:500}.grid td.info .role{color:var(--muted)}
.role{font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;font-weight:600;margin-bottom:2px}
.headline{font-size:16px;font-weight:600;padding:8px 12px;border-radius:6px;margin:10px 0 8px}
th.cellhead.target{font-weight:700}
.scroll{overflow-x:auto;margin:8px 0;-webkit-overflow-scrolling:touch}
.counts{min-width:640px}.counts th,.counts td{font-size:13.5px;padding:6px 8px}.counts th[scope=row]{font-weight:600}
.effects{min-width:760px}.effects td,.effects th{padding:5px 8px;font-size:13.5px}
.effects td.id,.effects th.id{white-space:nowrap;overflow-wrap:normal;position:sticky;left:0;background:var(--card)}
.effects tr.group td{border-top:2px solid var(--muted)}
.effects tfoot td{font-weight:600}
th.cellhead{white-space:nowrap;font-size:13px}
.chips{display:flex;flex-direction:column;align-items:flex-start;gap:3px}
.chip{display:inline-block;font-size:12px;line-height:1.4;padding:1px 7px;border-radius:9px;white-space:nowrap}
.chip b{font-weight:600}.chip.ok{background:var(--good);color:var(--good-fg)}
.chip.bad{background:var(--bad);color:var(--bad-fg)}.chip.warn{background:var(--warn);color:var(--warn-fg)}
.chip.none{background:var(--none);color:var(--muted)}
.legend .chip{margin:0 2px}
.o-failed{color:var(--bad-fg);font-weight:600}.o-passed{color:var(--good-fg)}
.note{margin:8px 0 0;color:var(--muted);font-size:13.5px}
details summary{cursor:pointer;color:var(--muted);font-size:12.5px}details ul{margin:4px 0;padding-left:18px;font-size:12px}
@media (max-width:640px){
 .grid thead{display:none}.grid,.grid tbody,.grid tr,.grid th,.grid td{display:block;width:100%!important}
 .grid tr{margin-bottom:12px}.grid td::before{content:attr(data-label);display:block;font-size:12px;opacity:.8}
}
"""


def headline(run: Run) -> str:
    t = run.targets()
    css = "good" if t["ok"] else "bad"
    which = " and ".join(label(c) for c in TARGETS)
    return (
        f"<p class='headline {css}'>{esc(t['headline'])}</p>"
        f"<p class=note>Targets: {esc(which)} must be green "
        f"({esc(run.r1_ref)} on the laboneq it pins, {esc(run.r2_ref)} on the newest laboneq). "
        "The other cells are information: what the code would hit on the other laboneq version.</p>"
    )


def failed_note(run: Run) -> str:
    """Where the failed tests are: one test-result check per cell, linked from the grid."""
    commit = f" (<a href='{esc(run.checks_url)}'>all checks of the commit</a>)" if run.checks_url else ""
    return (
        "Failed tests, with message and traceback: the check <i>ZI tests &lt;cell&gt;</i> of each "
        f"cell, linked as <i>test results</i> in the grid{commit}."
    )


def page(run: Run) -> str:
    link = f'<a href="{esc(run.run_url)}">{esc(run.run_url.rsplit("/", 1)[-1])}</a>' if run.run_url else "local"
    note = (
        f"Generated {esc(run.when or '-')}, run {link}. pinned = laboneq version in "
        f"{esc(run.r1_repo)}@{esc(run.r1_ref)} uv.lock; latest = newest laboneq on PyPI."
    )
    suite = (
        f"Each cell: the row's tests/instruments/zi plus the conformance suite "
        f"{esc(run.suite_repo)}@{esc(run.suite_ref)} {commit_link(run.suite_repo, run.ci_sha)}, "
        "LabOne Q emulation, no hardware. Row's uv.lock installed, then laboneq set to the column's "
        "version (zhinst-* follow). Not implemented yet = strict expected failure; it turns red if it passes."
    )
    checks = f"<a href='{esc(run.checks_url)}'>Test-result checks</a>" if run.checks_url else ""
    nav = (
        "<nav><a href='#effects'>Effect checks</a><a href='#references'>CP1-CP5 reference files</a>"
        f"{checks}</nav>"
    )
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>ZI driver tests</title><style>{CSS}</style></head><body><main>"
        f"<h1>ZI driver tests</h1>{nav}{headline(run)}"
        f"<div class=scroll>{grid(run)}</div><p class=note>{note}</p><p class=note>{suite}</p>"
        f"<p class=note>{failed_note(run)}</p>"
        f"{what_ran(run)}{effect_checks(run)}{references(run)}"
        "</main></body></html>"
    )


def state_json(run: Run) -> dict:
    cells = {}
    for cid in run.cell_ids:
        state = cell_state(run, cid)
        meta = run.cells.get(cid, {}).get("meta", {})
        state.pop("text", None)
        state["role"] = role(cid)
        state.update({k: meta[k] for k in ("laboneq", "zhinst_core", "sha", "merged_sha") if meta.get(k)})
        cells[cid] = state
    return {
        "key": run.key,
        "generated": run.when,
        "run": run.run_url,
        "targets": run.targets(),
        "rows": {r: row_text(run, r) for r in run.rows},
        "columns": {"c1": f"laboneq {run.pinned} (pinned)", "c2": f"laboneq {run.latest} (latest)"},
        "suite": f"{run.suite_repo}@{run.suite_ref} ({short(run.ci_sha)})",
        "cells": cells,
    }


def md(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ")


def summary_grid(run: Run) -> list[str]:
    lines = [
        f"| | {col_title(run, 'c1')}: {run.pinned} | {col_title(run, 'c2')}: {run.latest} |",
        "|---|---|---|",
    ]
    for row in run.rows:
        cells = []
        for col in COLS:
            cid = f"{row}-{col}"
            state = cell_state(run, cid)
            text = state["text"]
            if state.get("conflicts"):
                text += ": " + ", ".join(f"`{f}`" for f in state["conflicts"])
            mark = "**target**" if cid in TARGETS else "_information_"
            cells.append(f"{mark} {state['result']}: {text}")
        lines.append(f"| **{row.upper()}** `{row_text(run, row)}` | {' | '.join(cells)} |")
    return lines


def summary_features(run: Run) -> list[str]:
    features, per_cell = collect_features(run)
    if not features:
        return []
    cids = [cid for cid in run.cell_ids if ran(run, cid) is not None]
    lines = [
        "#### Features (contract / effect)",
        "",
        "| Id | Feature | CP | " + " | ".join(label(c) for c in cids) + " |",
        "|---|---|---|" + "---|" * len(cids),
    ]
    order = sorted(features, key=lambda f: (checkpoint_rank(features[f]["cp"]), id_rank(f)))
    for fid in order:
        info, tds = features[fid], []
        for cid in cids:
            cases = per_cell[cid].get(fid, [])
            parts = []
            for k in ("contract", "effect"):
                group = [c for c in cases if kind(c) == k]
                parts.append(tally(group)[1] if group else "-")
            mutants = [c for c in cases if kind(c) == "fault injection"]
            hit, total = caught(mutants)
            if hit < total:
                parts[1] += f" ({total - hit} fault{'s' if total - hit > 1 else ''} missed)"
            tds.append(" / ".join(parts))
        lines.append(f"| {fid} | {md(info['title'])} | {info['cp'] or '-'} | " + " | ".join(tds) + " |")
    lines.append("")
    return lines


def summary_markdown(run: Run) -> str:
    t = run.targets()
    lines = ["### ZI driver tests", "", f"**{t['headline']}**", ""] + summary_grid(run) + [""]
    lines += ["Failed tests, with message and traceback: the check `ZI tests <cell>` of each cell "
              "(its job summary is in this run).", ""]
    lines += summary_features(run)
    lines += ["Report: artifact `test-report` (index.html); on orkun/ci also https://sensebat.github.io/qibolab/.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cells_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--when", default="")
    parser.add_argument("--checks", type=Path, help="folder with check-<cell>/url.txt")
    args = parser.parse_args()

    cells = {}
    folders = [f for f in args.cells_dir.iterdir() if f.is_dir()] + [args.cells_dir]
    for folder in folders:
        cell = read_cell(folder)
        if cell is not None:
            cells[cell["meta"]["cell"]] = cell
    checks = {}
    if args.checks and args.checks.is_dir():
        for url in args.checks.glob("check-*/url.txt"):
            if url.read_text().strip():
                checks[url.parent.name.removeprefix("check-")] = url.read_text().strip()
    run = Run(cells, args.when, checks)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "index.html").write_text(page(run))
    (args.out_dir / "state.json").write_text(json.dumps(state_json(run), indent=2) + "\n")
    print(run.targets()["headline"])
    for cid in run.cell_ids:
        print(f"{label(cid)} ({role(cid)}): {cell_state(run, cid)['text']}")
    if args.summary:
        with args.summary.open("a") as f:
            f.write(summary_markdown(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
