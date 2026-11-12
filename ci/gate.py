"""Exit 0 exactly when every target cell of the run is green (the zi-driver gate job).

Usage: python3 ci/gate.py STATE_JSON

Reads "targets" from the state.json written by ci/report.py: the target cells this run
planned (R1 x pinned, R2 x latest) and their results. Information cells are not read.
"""

import json
import sys


def main(path: str) -> int:
    targets = json.load(open(path))["targets"]
    print(targets["headline"])
    for cell, result in targets["cells"].items():
        print(f"  {cell}: {result}")
    if not targets["ok"]:
        print(f"::error::{targets['headline']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
