"""Build one self-contained HTML test report from the rows of the zi-driver workflow.

Usage: python3 ci/report.py ROWS_DIR HTML_FILE [--summary MARKDOWN_FILE]

ROWS_DIR holds one folder per row (pinned, 26.10, merge-trial), each with meta.json and,
when the tests ran, junit.xml. --summary appends one headline per row as a Markdown
table (for $GITHUB_STEP_SUMMARY). Standard library only. Adapted from the summary.py of
the earlier zi-driver-ci harness.
"""

import argparse
import html
import json
import xml.etree.ElementTree as ET
from pathlib import Path

ROW_ORDER = ["pinned", "26.10", "merge-trial"]
ROW_ABOUT = {
    "pinned": "Upstream driver line, installed from its own uv.lock. Must be green.",
    "26.10": "Our line on laboneq 26.10, installed from its own uv.lock.",
    "merge-trial": "Upstream driver merged into the 26.10 line. Informational only.",
}
OUTCOMES = ["passed", "xfailed", "skipped", "failed", "error"]
LABELS = {
    "passed": "passed",
    "xfailed": "not implemented yet",
    "skipped": "skipped",
    "failed": "failed",
    "error": "error",
}
SUITE_CONFORMANCE = "conformance"
SUITE_DRIVER = "driver tests"


def outcome(case: ET.Element) -> tuple[str, str]:
    for tag in ("failure", "error"):
        node = case.find(tag)
        if node is not None:
            return ("failed" if tag == "failure" else "error"), node.get("message", "")
    node = case.find("skipped")
    if node is not None:
        kind = "xfailed" if node.get("type") == "pytest.xfail" else "skipped"
        return kind, node.get("message", "")
    return "passed", ""


def one_line(message: str, limit: int = 220) -> str:
    line = next((s.strip() for s in message.splitlines() if s.strip()), "")
    return line if len(line) <= limit else line[: limit - 1] + "…"


def read_junit(path: Path) -> list[dict]:
    cases = []
    for case in ET.parse(path).getroot().iter("testcase"):
        classname = case.get("classname", "")
        props = {p.get("name"): p.get("value") for p in case.iter("property")}
        result, message = outcome(case)
        cases.append(
            {
                "id": f"{classname}::{case.get('name')}",
                "name": case.get("name", ""),
                "file": classname.rsplit(".", 1)[-1],
                "suite": SUITE_CONFORMANCE if ".conformance." in f".{classname}." else SUITE_DRIVER,
                "outcome": result,
                "message": one_line(message),
                "props": props,
            }
        )
    return cases


def read_row(folder: Path) -> dict:
    meta_file = folder / "meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    name = meta.get("row") or folder.name.removeprefix("row-")
    junit = folder / "junit.xml"
    cases = read_junit(junit) if junit.exists() else None
    return {"name": name, "meta": meta, "cases": cases}


def counts(cases: list[dict]) -> dict[str, int]:
    n = {o: 0 for o in OUTCOMES}
    for case in cases:
        n[case["outcome"]] += 1
    return n


def verdict(row: dict) -> tuple[str, str]:
    """(css class, short text) for the row as a whole."""
    meta, cases = row["meta"], row["cases"]
    stage = meta.get("stage", "")
    if cases is None:
        return "bad", stage or "no results"
    n = counts(cases)
    bad = n["failed"] + n["error"]
    if bad:
        return "bad", f"{bad} failing"
    if str(meta.get("pytest_exit", "0")) not in ("0", ""):
        return "bad", f"pytest exit code {meta['pytest_exit']}"
    return "good", "green"


def count_text(n: dict[str, int]) -> str:
    parts = [
        f"{n['passed']} passed",
        f"{n['xfailed']} not implemented yet",
        f"{n['failed']} failed",
        f"{n['error']} errors",
    ]
    if n["skipped"]:
        parts.append(f"{n['skipped']} skipped")
    return ", ".join(parts)


def esc(value) -> str:
    return html.escape(str(value))


def short(sha: str) -> str:
    return (sha or "")[:8]


def driver_text(meta: dict) -> str:
    return f"{meta.get('repo', '?')}@{meta.get('ref', '?')}@{short(meta.get('sha', ''))}"


def headline(row: dict) -> str:
    meta = row["meta"]
    _, text = verdict(row)
    result = count_text(counts(row["cases"])) if row["cases"] is not None else text
    return (
        f"{row['name']}: {text}. {result}. Driver {driver_text(meta)}, "
        f"laboneq {meta.get('laboneq') or '-'}, zhinst-core {meta.get('zhinst_core') or '-'}."
    )


def overview(rows: list[dict]) -> str:
    body = []
    for row in rows:
        meta = row["meta"]
        css, text = verdict(row)
        result = count_text(counts(row["cases"])) if row["cases"] is not None else "no test results"
        body.append(
            f"<tr><th><a href='#row-{esc(row['name'])}'>{esc(row['name'])}</a></th>"
            f"<td class={css}><b>{esc(text)}</b></td>"
            f"<td>{esc(result)}</td>"
            f"<td class=mono>{esc(driver_text(meta))}</td>"
            f"<td class=mono>{esc(meta.get('laboneq') or '-')}</td>"
            f"<td class=mono>{esc(meta.get('zhinst_core') or '-')}</td></tr>"
        )
    return (
        "<table><tr><th>Line</th><th>Result</th><th>Counts</th><th>Driver (repo@ref@sha)</th><th>laboneq</th>"
        f"<th>zhinst-core</th></tr>{''.join(body)}</table>"
    )


def row_section(row: dict) -> str:
    meta, cases = row["meta"], row["cases"]
    css, text = verdict(row)
    facts = [
        ("Driver", f"{meta.get('repo', '?')} @ {meta.get('ref', '?')} @ {short(meta.get('sha', ''))}"),
        ("laboneq", meta.get("laboneq") or "-"),
        ("zhinst-core", meta.get("zhinst_core") or "-"),
        ("Python", meta.get("python") or "-"),
        ("Conformance suite", f"orkun/ci @ {short(meta.get('suite_sha', ''))}"),
    ]
    if meta.get("merged_from"):
        facts.insert(1, ("Merged in", f"{meta['merged_from']} @ {short(meta.get('merged_sha', ''))}"))
    if meta.get("merge_note"):
        facts.insert(2, ("Merge", meta["merge_note"]))
    rows = "".join(f"<tr><th>{esc(k)}</th><td class=mono>{esc(v)}</td></tr>" for k, v in facts)
    parts = [
        f"<h2 id='row-{esc(row['name'])}'>{esc(row['name'])} "
        f"<span class='pill {css}'>{esc(text)}</span></h2>",
        f"<p class=muted>{esc(ROW_ABOUT.get(row['name'], ''))}</p>",
        f"<div class=scroll><table class=facts>{rows}</table></div>",
    ]
    if cases is None:
        stage = meta.get("stage") or "the job stopped before the tests"
        detail = ""
        if meta.get("conflicts"):
            files = "".join(f"<li class=mono>{esc(f)}</li>" for f in meta["conflicts"])
            detail = f"<p>Conflicting files:</p><ul>{files}</ul>"
        parts.append(f"<p class=bad-text><b>No test results:</b> {esc(stage)}.</p>{detail}")
        return "".join(parts)

    suites = []
    for suite in (SUITE_DRIVER, SUITE_CONFORMANCE):
        n = counts([c for c in cases if c["suite"] == suite])
        cells = "".join(f"<td>{n[o]}</td>" for o in ("passed", "xfailed", "failed", "error"))
        suites.append(f"<tr><th>{esc(suite)}</th>{cells}</tr>")
    n = counts(cases)
    total = "".join(f"<td><b>{n[o]}</b></td>" for o in ("passed", "xfailed", "failed", "error"))
    suites.append(f"<tr><th>total</th>{total}</tr>")
    parts.append(
        "<div class=scroll><table><tr><th></th><th>passed</th><th>not implemented yet</th>"
        f"<th>failed</th><th>errors</th></tr>{''.join(suites)}</table></div>"
    )
    if n["skipped"]:
        parts.append(f"<p class=muted>{n['skipped']} skipped.</p>")

    bad = [c for c in cases if c["outcome"] in ("failed", "error")]
    if bad:
        items = "".join(
            f"<li><span class=mono>{esc(c['file'])}::{esc(c['name'])}</span> "
            f"<span class=o-{c['outcome']}>{c['outcome']}</span>"
            f"<div class=msg>{esc(c['message'])}</div></li>"
            for c in bad
        )
        parts.append(f"<h3>Failed tests</h3><ul class=fails>{items}</ul>")
    else:
        parts.append("<p>No failed tests.</p>")
    return "".join(parts)


def feature_status(cases: list[dict]) -> tuple[str, str]:
    n = counts(cases)
    total = len(cases)
    if n["failed"] + n["error"]:
        return "o-failed", f"failed ({n['failed'] + n['error']}/{total})"
    if n["passed"] == total:
        return "o-passed", "passed"
    if n["xfailed"] == total:
        return "o-xfailed", "not implemented yet"
    if n["passed"]:
        return "o-xfailed", f"partly ({n['passed']}/{total} passed)"
    return "o-skipped", "skipped"


def conformance_table(rows: list[dict]) -> str:
    keys: dict[str, dict] = {}
    for row in rows:
        for case in row["cases"] or []:
            if case["suite"] != SUITE_CONFORMANCE:
                continue
            props = case["props"]
            if props.get("feature"):
                key = props["feature"]
                info = {"id": key, "title": props.get("title", ""), "checkpoint": props.get("checkpoint", "")}
            elif props.get("checkpoint"):
                base = case["name"].split("[", 1)[0]
                key = f"{props['checkpoint']} {base}"
                info = {"id": props["checkpoint"], "title": f"checkpoint test {base}", "checkpoint": props["checkpoint"]}
            else:
                continue
            keys.setdefault(key, info)
    if not keys:
        return "<p>No conformance results.</p>"

    def sort_key(key: str) -> tuple:
        info = keys[key]
        return (not info["id"].startswith("F"), info["id"], key)

    head = "".join(f"<th>{esc(r['name'])}</th>" for r in rows)
    body = []
    for key in sorted(keys, key=sort_key):
        info = keys[key]
        cells = []
        for row in rows:
            if row["cases"] is None:
                cells.append("<td class=muted>no result</td>")
                continue
            matching = []
            for case in row["cases"]:
                if case["suite"] != SUITE_CONFORMANCE:
                    continue
                props = case["props"]
                if props.get("feature") == key or (
                    not props.get("feature")
                    and props.get("checkpoint")
                    and f"{props['checkpoint']} {case['name'].split('[', 1)[0]}" == key
                ):
                    matching.append(case)
            if not matching:
                cells.append("<td class=muted>absent</td>")
                continue
            css, text = feature_status(matching)
            cells.append(f"<td class={css}>{esc(text)}</td>")
        body.append(
            f"<tr><td class=mono>{esc(info['id'])}</td><td>{esc(info['title'])}</td>"
            f"<td class=mono>{esc(info['checkpoint'])}</td>{''.join(cells)}</tr>"
        )
    return (
        f"<table><tr><th>Id</th><th>Title</th><th>Checkpoint</th>{head}</tr>"
        f"{''.join(body)}</table>"
    )


CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#ddddd8;--card:#ffffff;
--good:#e3f2e6;--good-fg:#1e5b2c;--bad:#fbe4e2;--bad-fg:#8a1f17;--warn:#fff2cc;--warn-fg:#6b5300}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#171716;--fg:#e8e8e3;
--muted:#9a9a93;--line:#34342f;--card:#1f1f1d;--good:#173322;--good-fg:#9fd9ad;--bad:#3a1a17;
--bad-fg:#f2a49c;--warn:#352d10;--warn-fg:#e8cf7a}}
:root[data-theme="dark"]{--bg:#171716;--fg:#e8e8e3;--muted:#9a9a93;--line:#34342f;--card:#1f1f1d;
--good:#173322;--good-fg:#9fd9ad;--bad:#3a1a17;--bad-fg:#f2a49c;--warn:#352d10;--warn-fg:#e8cf7a}
*{box-sizing:border-box}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,-apple-system,sans-serif;margin:0;padding:24px 16px}
main{max-width:1000px;margin:0 auto}h1{font-size:22px;margin:0 0 4px}
h2{font-size:18px;margin:32px 0 6px;padding-top:12px;border-top:1px solid var(--line)}
h3{font-size:15px;margin:16px 0 4px}.muted{color:var(--muted)}a{color:inherit}
.mono,.msg{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
.msg{color:var(--muted);overflow-wrap:anywhere;margin:2px 0 8px}
.scroll{overflow-x:auto;margin:8px 0}table{border-collapse:collapse;background:var(--card);width:100%}
th,td{border:1px solid var(--line);padding:6px 10px;text-align:left;vertical-align:top}
table.facts th{width:11em}.mono{overflow-wrap:anywhere}
.good{background:var(--good);color:var(--good-fg)}.bad{background:var(--bad);color:var(--bad-fg)}
.pill{font-size:13px;font-weight:600;padding:2px 8px;border-radius:10px;vertical-align:middle}
.bad-text{color:var(--bad-fg)}ul.fails{padding-left:20px}ul.fails li{overflow-wrap:anywhere}
.o-failed,.o-error{color:var(--bad-fg);font-weight:600}.o-xfailed,.o-skipped{color:var(--warn-fg)}
.o-passed{color:var(--good-fg)}
"""


def page(rows: list[dict], run_url: str, when: str) -> str:
    stamp = f"Generated {esc(when)}." if when else ""
    link = f' Workflow run: <a href="{esc(run_url)}">{esc(run_url)}</a>.' if run_url else ""
    top = f"<p class=mono>{stamp}{link}</p>" if stamp or link else ""
    sections = [
        "<h2>Overview</h2><div class=scroll>" + overview(rows) + "</div>",
        "<p class=muted>“Not implemented yet” means a strict expected failure: the driver "
        "refuses the feature, or LabOne Q rejects what the driver sends. It turns red if "
        "it starts passing.</p>",
        *(row_section(r) for r in rows),
        "<h2>Conformance per feature</h2>"
        "<p class=muted>What qibocal needs from the controller, one row per feature; "
        "checkpoint is the first checkpoint that needs it.</p>"
        "<div class=scroll>" + conformance_table(rows) + "</div>",
    ]
    return (
        "<!doctype html><html lang=en><head><meta charset=utf-8>"
        '<meta name=viewport content="width=device-width,initial-scale=1">'
        f"<title>ZI driver tests</title><style>{CSS}</style></head><body><main>"
        f"<h1>ZI driver tests</h1>{top}"
        "<p class=muted>The driver's own tests (tests/instruments/zi) and the conformance "
        "suite, run in LabOne Q emulation without hardware. Each line installs from its own "
        "uv.lock.</p>"
        f"{''.join(sections)}</main></body></html>"
    )


def summary_markdown(rows: list[dict], run_url: str) -> str:
    lines = ["### ZI driver tests", "", "| Line | Result | Counts | Driver | laboneq | zhinst-core |", "|---|---|---|---|---|---|"]
    for row in rows:
        meta = row["meta"]
        _, text = verdict(row)
        result = count_text(counts(row["cases"])) if row["cases"] is not None else "no test results"
        lines.append(
            f"| {row['name']} | {text} | {result} | `{driver_text(meta)}` | "
            f"{meta.get('laboneq') or '-'} | {meta.get('zhinst_core') or '-'} |"
        )
    lines += ["", "Full report: artifact `test-report`" + (f" of {run_url}" if run_url else "") + ".", ""]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("rows_dir", type=Path)
    parser.add_argument("html_file", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--run-url", default="")
    parser.add_argument("--when", default="")
    args = parser.parse_args()

    rows = [read_row(f) for f in args.rows_dir.iterdir() if f.is_dir()]
    if (args.rows_dir / "meta.json").exists():
        # download-artifact unpacks a single artifact without its own folder.
        rows.append(read_row(args.rows_dir))
    rows.sort(key=lambda r: (ROW_ORDER.index(r["name"]) if r["name"] in ROW_ORDER else 99, r["name"]))
    if not rows:
        print(f"no row folders in {args.rows_dir}")
        return 1
    args.html_file.parent.mkdir(parents=True, exist_ok=True)
    args.html_file.write_text(page(rows, args.run_url, args.when))
    print(f"wrote {args.html_file}")
    for row in rows:
        print(headline(row))
    if args.summary:
        with args.summary.open("a") as f:
            f.write(summary_markdown(rows, args.run_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
