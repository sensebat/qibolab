"""Build the zi-driver report (index.html) and state.json from the 2x2 cells.

Usage: python3 ci/report.py CELLS_DIR OUT_DIR [--summary MARKDOWN_FILE] [--when TEXT]

CELLS_DIR holds one folder per cell (cell-r1-c1 ... cell-r2-c2) with meta.json and, when the
tests ran, junit.xml (written by ci/run_cell.sh). OUT_DIR gets index.html and state.json.
Run facts come from the environment (set by the workflow; the cells' meta.json fills gaps):
  KEY, RUN_URL, MATRIX, R1_SHA, R2_SHA, CI_SHA, LABONEQ_PINNED, LABONEQ_LATEST,
  R1_REPO, R1_REF, R2_REPO, R2_REF, SUITE_REPO, SUITE_REF
--summary appends the 2x2 as a Markdown table (for $GITHUB_STEP_SUMMARY). Standard library only.
"""

import argparse
import html
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

ROWS = ["r1", "r2"]
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
    def __init__(self, cells: dict[str, dict], when: str):
        e = os.environ
        metas = [c["meta"] for c in cells.values()]

        def first(field: str, env: str, default: str = "") -> str:
            if e.get(env):
                return e[env]
            return next((m[field] for m in metas if m.get(field)), default)

        self.cells = cells
        self.when = when
        self.run_url = e.get("RUN_URL", "")
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
        if e.get("MATRIX"):
            self.planned = [c["cell"] for c in json.loads(e["MATRIX"])["include"]]
        else:
            self.planned = list(cells)

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


def row_text(run: Run, row: str) -> str:
    r1 = f"{run.r1_repo}@{run.r1_ref} ({short(run.r1_sha)})"
    if row == "r1":
        return r1
    return f"{run.r2_repo}@{run.r2_ref} ({short(run.r2_sha)}) + {r1} merged"


def row_html(run: Run, row: str) -> str:
    r1 = f"<span class=mono>{esc(run.r1_repo)}@{esc(run.r1_ref)}</span> {commit_link(run.r1_repo, run.r1_sha)}"
    if row == "r1":
        return r1
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
    for row in ROWS:
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
            tds.append(
                f"<td class={CSS_CLASS[state['result']]} data-label='{esc(col_title(run, col))}'>"
                f"<a href='#{cell_id}'><b>{esc(state['text'])}</b></a>{files}"
                + (f"<div class='small mono'>{esc(detail)}</div>" if detail else "")
                + "</td>"
            )
        body.append(f"<tr><th scope=row><b>{row.upper()}</b> {row_html(run, row)}</th>{''.join(tds)}</tr>")
    return f"<table class=grid><thead><tr><th></th>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def failed_section(run: Run) -> str:
    parts = ["<h2>Failed tests</h2>"]
    for row in ROWS:
        for col in COLS:
            cell_id = f"{row}-{col}"
            state = cell_state(run, cell_id)
            title = f"{row.upper()} × {col_title(run, col)}"
            head = f"<h3 id={cell_id}>{esc(title)} <span class='pill {CSS_CLASS[state['result']]}'>{esc(state['result'])}</span></h3>"
            cell = run.cells.get(cell_id)
            if state["result"] == "merge conflict":
                files = "".join(f"<li class=mono>{esc(f)}</li>" for f in state.get("conflicts", []))
                parts.append(f"{head}<p>{esc(row_text(run, row))} does not merge. Files:</p><ul>{files}</ul>")
            elif cell is None or cell["cases"] is None:
                parts.append(f"{head}<p class=muted>{esc(state['text'])}.</p>")
            else:
                bad = [c for c in cell["cases"] if c["outcome"] in ("failed", "error")]
                if not bad:
                    parts.append(f"{head}<p class=muted>None.</p>")
                    continue
                items = "".join(
                    f"<li><span class=mono>{esc(c['file'])}::{esc(c['name'])}</span>"
                    + (" <span class=o-failed>error</span>" if c["outcome"] == "error" else "")
                    + f"<div class=msg>{esc(c['message'])}</div></li>"
                    for c in bad
                )
                parts.append(f"{head}<ul class=fails>{items}</ul>")
    return "".join(parts)


def feature_key(case: dict) -> tuple[str, dict] | None:
    props = case["props"]
    if props.get("feature"):
        info = {"id": props["feature"], "title": props.get("title", ""), "cp": props.get("checkpoint", "")}
        info["effects"] = props["effects"].split() if "effects" in props else None
        return props["feature"], info
    if props.get("checkpoint"):
        base = case["name"].split("[", 1)[0]
        cp = props["checkpoint"]
        effects = [f"{case['file']}.py::{base}"] if props.get("kind") == "effect" else []
        return f"{cp} {base}", {"id": cp, "title": f"checkpoint test {base}", "cp": cp, "effects": effects}
    return None


def is_mutant_case(case: dict) -> bool:
    """A case of an effect test run under a planted driver bug (``[...mutant-<name>]``)."""
    return "mutant-" in case["name"].partition("[")[2]


def effects_cell(effects: list[str] | None) -> str:
    if effects is None:
        return "<td class=muted>-</td>"
    if not effects:
        return "<td class=o-xfailed>none yet</td>"
    names = "<br>".join(esc(e) for e in effects)
    return f"<td><details><summary>{len(effects)}</summary><div class=mono>{names}</div></details></td>"


def mutants_line(run: Run) -> str:
    """How many planted-bug cases each cell ran, and how many turned the test red."""
    parts = []
    for row in ROWS:
        for col in COLS:
            cell = run.cells.get(f"{row}-{col}")
            if cell is None or cell["cases"] is None:
                continue
            cases = [c for c in cell["cases"] if c["conformance"] and is_mutant_case(c)]
            if not cases:
                continue
            caught = sum(c["outcome"] == "passed" for c in cases)
            parts.append(f"{row.upper()} {'pinned' if col == 'c1' else 'latest'}: {caught} of {len(cases)} caught")
    if not parts:
        return ""
    return (
        "<p class=note>Planted driver bugs (tests/instruments/zi/conformance/mutants.py, matrix in TEETH.md): "
        "each effect test also runs under the bugs it names and passes only if the bug turns it red. "
        + esc(" · ".join(parts))
        + ".</p>"
    )


def feature_status(cases: list[dict]) -> tuple[str, str]:
    n = counts(cases)
    total = len(cases)
    if n["failed"] + n["error"]:
        return "o-failed", f"failed {n['failed'] + n['error']}/{total}"
    if n["passed"] == total:
        return "o-passed", "passed"
    if n["xfailed"] == total:
        return "o-xfailed", "not yet"
    if n["passed"]:
        return "o-xfailed", f"partly {n['passed']}/{total}"
    return "o-skipped", "skipped"


def conformance_table(run: Run) -> str:
    cell_ids = [f"{r}-{c}" for r in ROWS for c in COLS]
    features: dict[str, dict] = {}
    per_cell: dict[str, dict[str, list]] = {cid: {} for cid in cell_ids}
    for cid in cell_ids:
        cell = run.cells.get(cid)
        for case in (cell or {}).get("cases") or []:
            if not case["conformance"]:
                continue
            found = feature_key(case)
            if found is None:
                continue
            key, info = found
            known = features.setdefault(key, info)
            if known.get("effects") is None and info.get("effects") is not None:
                known["effects"] = info["effects"]
            per_cell[cid].setdefault(key, []).append(case)
    if not features:
        return "<p class=muted>No conformance results.</p>"
    head = "".join(f"<th>{cid[:2].upper()}<br>{'pinned' if cid.endswith('c1') else 'latest'}</th>" for cid in cell_ids)
    body = []
    for key in sorted(features, key=lambda k: (not features[k]["id"].startswith("F"), features[k]["id"], k)):
        info = features[key]
        tds = []
        for cid in cell_ids:
            if cid not in run.cells or run.cells[cid]["cases"] is None:
                tds.append("<td class=muted>-</td>")
            elif key not in per_cell[cid]:
                tds.append("<td class=muted>absent</td>")
            else:
                css, text = feature_status(per_cell[cid][key])
                tds.append(f"<td class={css}>{esc(text)}</td>")
        cp = f" <span class=muted>{esc(info['cp'])}</span>" if info["cp"] and info["cp"] != info["id"] else ""
        body.append(
            f"<tr><td class='mono id'>{esc(info['id'])}{cp}</td><td>{esc(info['title'])}</td>"
            f"{effects_cell(info.get('effects'))}{''.join(tds)}</tr>"
        )
    return (
        f"<div class=scroll><table class=features><tr><th>Id</th><th>Feature</th><th>Effect tests</th>{head}</tr>"
        f"{''.join(body)}</table></div>"
    )


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#ddddd8;--card:#fff;
--good:#e3f2e6;--good-fg:#1e5b2c;--bad:#fbe4e2;--bad-fg:#8a1f17;--warn-fg:#6b5300;--none:#f1f1ee}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#171716;--fg:#e8e8e3;
--muted:#9a9a93;--line:#34342f;--card:#1f1f1d;--good:#173322;--good-fg:#9fd9ad;--bad:#3a1a17;
--bad-fg:#f2a49c;--warn-fg:#e8cf7a;--none:#262624}}
:root[data-theme="dark"]{--bg:#171716;--fg:#e8e8e3;--muted:#9a9a93;--line:#34342f;--card:#1f1f1d;
--good:#173322;--good-fg:#9fd9ad;--bad:#3a1a17;--bad-fg:#f2a49c;--warn-fg:#e8cf7a;--none:#262624}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,sans-serif;margin:0;padding:24px 16px}
main{max-width:1000px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}
h2{font-size:18px;margin:32px 0 6px;padding-top:12px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:18px 0 4px}a{color:inherit}.muted{color:var(--muted)}
.mono,.msg{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;overflow-wrap:anywhere}
.small{font-size:12.5px;color:inherit;opacity:.85;margin-top:4px}
.msg{color:var(--muted);margin:2px 0 8px}
table{border-collapse:collapse;background:var(--card);width:100%}
th,td{border:1px solid var(--line);padding:8px 10px;text-align:left;vertical-align:top}
.grid th[scope=row]{font-weight:400;width:34%}.grid td a{text-decoration:none}
.good{background:var(--good);color:var(--good-fg)}.bad{background:var(--bad);color:var(--bad-fg)}
.none{background:var(--none);color:var(--muted)}
.pill{font-size:12px;font-weight:600;padding:1px 8px;border-radius:10px;vertical-align:middle}
.scroll{overflow-x:auto;margin:8px 0}ul.fails{padding-left:20px}ul.fails li{overflow-wrap:anywhere}
.features{min-width:600px}.features td,.features th{padding:5px 8px;font-size:13.5px}.features td.id{white-space:nowrap;overflow-wrap:normal}
.o-failed{color:var(--bad-fg);font-weight:600}.o-xfailed,.o-skipped{color:var(--warn-fg)}.o-passed{color:var(--good-fg)}
.note{margin:8px 0 0;color:var(--muted);font-size:13.5px}
details summary{cursor:pointer}details .mono{font-size:12px;margin-top:4px}
@media (max-width:640px){
 .grid thead{display:none}.grid,.grid tbody,.grid tr,.grid th,.grid td{display:block;width:100%!important}
 .grid tr{margin-bottom:12px}.grid td::before{content:attr(data-label);display:block;font-size:12px;opacity:.8}
}
"""


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
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>ZI driver tests</title><style>{CSS}</style></head><body><main>"
        "<h1>ZI driver tests</h1>"
        f"{grid(run)}<p class=note>{note}</p><p class=note>{suite}</p>"
        f"{failed_section(run)}"
        "<h2>Conformance per feature</h2>"
        "<p class=muted>What qibocal needs from the controller, one line per feature (id, first checkpoint). "
        "Effect tests check what a qibolab call makes LabOne Q do, at the LabOne Q boundary; "
        "the other tests check the qibolab contract (keys, shapes, refusals).</p>"
        f"{mutants_line(run)}{conformance_table(run)}</main></body></html>"
    )


def state_json(run: Run) -> dict:
    cells = {}
    for row in ROWS:
        for col in COLS:
            cid = f"{row}-{col}"
            state = cell_state(run, cid)
            meta = run.cells.get(cid, {}).get("meta", {})
            state.pop("text", None)
            state.update({k: meta[k] for k in ("laboneq", "zhinst_core", "sha", "merged_sha") if meta.get(k)})
            cells[cid] = state
    return {
        "key": run.key,
        "generated": run.when,
        "run": run.run_url,
        "rows": {"r1": row_text(run, "r1"), "r2": row_text(run, "r2")},
        "columns": {"c1": f"laboneq {run.pinned} (pinned)", "c2": f"laboneq {run.latest} (latest)"},
        "suite": f"{run.suite_repo}@{run.suite_ref} ({short(run.ci_sha)})",
        "cells": cells,
    }


def summary_markdown(run: Run) -> str:
    lines = [
        "### ZI driver tests",
        "",
        f"| | {col_title(run, 'c1')}: {run.pinned} | {col_title(run, 'c2')}: {run.latest} |",
        "|---|---|---|",
    ]
    for row in ROWS:
        cells = []
        for col in COLS:
            state = cell_state(run, f"{row}-{col}")
            text = state["text"]
            if state.get("conflicts"):
                text += ": " + ", ".join(f"`{f}`" for f in state["conflicts"])
            cells.append(f"**{state['result']}**: {text}" if state["result"] in ("green", "red") else text)
        lines.append(f"| **{row.upper()}** `{row_text(run, row)}` | {' | '.join(cells)} |")
    lines += ["", "Report: artifact `test-report` (index.html); on orkun/ci also https://sensebat.github.io/qibolab/.", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cells_dir", type=Path)
    parser.add_argument("out_dir", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--when", default="")
    args = parser.parse_args()

    cells = {}
    folders = [f for f in args.cells_dir.iterdir() if f.is_dir()] + [args.cells_dir]
    for folder in folders:
        cell = read_cell(folder)
        if cell is not None:
            cells[cell["meta"]["cell"]] = cell
    run = Run(cells, args.when)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "index.html").write_text(page(run))
    (args.out_dir / "state.json").write_text(json.dumps(state_json(run), indent=2) + "\n")
    for row in ROWS:
        for col in COLS:
            print(f"{row.upper()} x {col_title(run, col)}: {cell_state(run, f'{row}-{col}')['text']}")
    if args.summary:
        with args.summary.open("a") as f:
            f.write(summary_markdown(run))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
