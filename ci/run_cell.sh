#!/usr/bin/env bash
# Run one cell of the zi-driver 2x2 on a checked-out driver.
#
#   rows     r1  ElStabilini/qibolab@zi_drivers
#            r2  sensebat/qibolab@orkun/26.10 + ElStabilini/qibolab@zi_drivers merged (for
#                this run only, nothing is pushed)
#            x   extra row: any other commit (orkun/proto-* branches, dispatch input), no merge
#   columns  c1  laboneq pinned: the version in the uv.lock of ElStabilini/qibolab@zi_drivers
#            c2  laboneq latest: the newest laboneq on PyPI
#
# Steps: (r2) merge R1_SHA into the checkout; copy in the conformance suite; install from the
# checkout's uv.lock; replace laboneq by the column's version with uv pip (zhinst-* follow,
# every other package stays as locked); run tests/instruments/zi.
# Always writes OUT_DIR/meta.json; OUT_DIR/junit.xml when the tests ran. Exit code is the
# pytest exit code, or 1 when a step before the tests failed.
#
# Usage: ci/run_cell.sh DRIVER_DIR SUITE_DIR OUT_DIR
# Env:   ROW (r1|r2|x), COL (c1|c2)
#        R1_REPO, R1_REF, R1_SHA   ElStabilini/qibolab, zi_drivers, its commit
#        R2_REPO, R2_REF           sensebat/qibolab, orkun/26.10 (labels; r2 only)
#        EXTRA_REPO, EXTRA_REF     labels of the extra row (x only)
#        SUITE_REF                 orkun/ci (label)
#        LABONEQ_PINNED, LABONEQ_LATEST   optional; read from R1's uv.lock / PyPI when empty
#        R1_FETCH_URL              optional; where to fetch R1_SHA from (default GitHub)
set -uo pipefail
: "${ROW:?r1, r2 or x}" "${COL:?c1 or c2}"

driver=$(cd "$1" && pwd)
suite=$(cd "$2" && pwd)
mkdir -p "$3"
out=$(cd "$3" && pwd)

export ROW COL OUT="$out" STAGE="checkout" SHA MERGED_SHA="" MERGE_NOTE="" CONFLICTS=""
export LABONEQ_PINNED="${LABONEQ_PINNED:-}" LABONEQ_LATEST="${LABONEQ_LATEST:-}" TARGET=""
export LABONEQ="" ZHINST_CORE="" PYVERSION="" PYTEST_EXIT="" SUITE_SHA
SHA=$(git -C "$driver" rev-parse HEAD)
SUITE_SHA=$(git -C "$suite" rev-parse HEAD)

# shellcheck disable=SC2329  # called by the EXIT trap
write_meta() {
  python3 - <<'EOF'
import json, os
e = os.environ
row, col = e["ROW"], e["COL"]
meta = {
    "cell": f"{row}-{col}", "row": row, "col": col,
    "repo": e.get({"r1": "R1_REPO", "r2": "R2_REPO"}.get(row, "EXTRA_REPO"), ""),
    "ref": e.get({"r1": "R1_REF", "r2": "R2_REF"}.get(row, "EXTRA_REF"), ""),
    "sha": e["SHA"],
    "merged_repo": e.get("R1_REPO", "") if row == "r2" else "",
    "merged_ref": e.get("R1_REF", "") if row == "r2" else "",
    "merged_sha": e["MERGED_SHA"], "merge_note": e["MERGE_NOTE"],
    "conflicts": [f for f in e["CONFLICTS"].splitlines() if f],
    "laboneq_pinned": e["LABONEQ_PINNED"], "laboneq_latest": e["LABONEQ_LATEST"],
    "laboneq_target": e["TARGET"],
    "laboneq": e["LABONEQ"], "zhinst_core": e["ZHINST_CORE"], "python": e["PYVERSION"],
    "suite_ref": e.get("SUITE_REF", ""), "suite_sha": e["SUITE_SHA"],
    "stage": e["STAGE"], "pytest_exit": e["PYTEST_EXIT"],
}
with open(os.path.join(e["OUT"], "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
EOF
}
trap write_meta EXIT

lock_laboneq() {  # laboneq version in the uv.lock read from stdin
  python3 -c 'import sys, tomllib
print(next(p["version"] for p in tomllib.loads(sys.stdin.read())["package"] if p["name"] == "laboneq"))'
}

if [ "$ROW" = r2 ]; then
  from="${R1_REPO}@${R1_REF}"
  STAGE="fetch of ${from} failed"
  git -C "$driver" fetch --no-tags "${R1_FETCH_URL:-https://github.com/$R1_REPO}" "${R1_SHA:-$R1_REF}" || exit 1
  MERGED_SHA=$(git -C "$driver" rev-parse FETCH_HEAD)
  if git -C "$driver" merge-base --is-ancestor FETCH_HEAD HEAD; then
    MERGE_NOTE="already contained"
  elif git -C "$driver" -c user.name=zi-driver-ci -c user.email=ci@invalid \
      merge --no-edit FETCH_HEAD; then
    MERGE_NOTE="clean merge"
  else
    STAGE="merge conflict"
    CONFLICTS=$(git -C "$driver" diff --name-only --diff-filter=U)
    echo "merge conflict in:"; echo "$CONFLICTS"
    echo "::error::merge conflict: ${from} (${MERGED_SHA:0:8}) into ${R2_REPO}@${R2_REF} (${SHA:0:8})"
    git -C "$driver" merge --abort
    exit 1
  fi
  r1_lock() { git -C "$driver" show "$MERGED_SHA:uv.lock"; }
elif [ "$ROW" = r1 ]; then
  r1_lock() { git -C "$driver" show "$SHA:uv.lock"; }
else
  r1_lock() { curl -fsS "https://raw.githubusercontent.com/$R1_REPO/${R1_SHA:-$R1_REF}/uv.lock"; }
fi

STAGE="reading laboneq versions failed"
# pinned = the laboneq version in R1's uv.lock, for every row.
if [ -z "$LABONEQ_PINNED" ]; then
  LABONEQ_PINNED=$(r1_lock | lock_laboneq) || exit 1
fi
if [ -z "$LABONEQ_LATEST" ]; then
  LABONEQ_LATEST=$(curl -fsS https://pypi.org/pypi/laboneq/json \
    | python3 -c 'import json, sys; print(json.load(sys.stdin)["info"]["version"])') || exit 1
fi
if [ "$COL" = c1 ]; then TARGET=$LABONEQ_PINNED; else TARGET=$LABONEQ_LATEST; fi
echo "laboneq pinned $LABONEQ_PINNED, latest $LABONEQ_LATEST; this cell: $TARGET"

echo "conformance suite ${SUITE_REF:-} (${SUITE_SHA:0:8})"
mkdir -p "$driver/tests/instruments/zi/conformance"
cp -R "$suite/tests/instruments/zi/conformance/." "$driver/tests/instruments/zi/conformance/"

cd "$driver" || exit 1
STAGE="install from uv.lock failed"
uv sync --group zi --group tests --frozen || exit 1

installed=$(uv run --no-sync python -c 'import importlib.metadata as m; print(m.version("laboneq"))')
if [ "$installed" != "$TARGET" ]; then
  STAGE="install of laboneq $TARGET failed"
  echo "laboneq $installed from uv.lock -> $TARGET"
  uv pip install --python .venv "laboneq==$TARGET" || exit 1
fi

read -r LABONEQ ZHINST_CORE PYVERSION < <(uv run --no-sync python -c '
import importlib.metadata as m, platform
print(m.version("laboneq"), m.version("zhinst-core"), platform.python_version())')
echo "laboneq $LABONEQ, zhinst-core $ZHINST_CORE, python $PYVERSION"

STAGE="tests ran"
# -o addopts= drops qibolab's coverage options.
uv run --no-sync pytest tests/instruments/zi -q -o addopts= -p no:cacheprovider \
  --junitxml="$out/junit.xml"
PYTEST_EXIT=$?
exit "$PYTEST_EXIT"
