#!/usr/bin/env bash
# Run one line ("row") of the zi-driver workflow on a checked-out driver:
# optional merge, copy in the conformance suite, install from the driver's own uv.lock,
# run tests/instruments/zi. Always writes OUT_DIR/meta.json; junit.xml when the tests ran.
#
# Usage: ci/run_row.sh DRIVER_DIR SUITE_DIR OUT_DIR
# Env:   ROW, DRIVER_REPO, DRIVER_REF (labels for the report)
#        MERGE_URL, MERGE_REF  merge this branch into the driver first (merge-trial)
#        UV_PYTHON             Python for uv (3.13 in CI)
set -uo pipefail

driver=$(cd "$1" && pwd)
suite=$(cd "$2" && pwd)
mkdir -p "$3"
out=$(cd "$3" && pwd)

export STAGE="checkout" SHA MERGED_FROM="" MERGED_SHA="" MERGE_NOTE="" CONFLICTS=""
export LABONEQ="" ZHINST_CORE="" PYVERSION="" PYTEST_EXIT="" OUT="$out"
SHA=$(git -C "$driver" rev-parse HEAD)
export SUITE_SHA
SUITE_SHA=$(git -C "$suite" rev-parse HEAD)

write_meta() {
  python3 - <<'EOF'
import json, os
e = os.environ
meta = {
    "row": e.get("ROW", ""), "repo": e.get("DRIVER_REPO", ""), "ref": e.get("DRIVER_REF", ""),
    "sha": e["SHA"], "suite_sha": e["SUITE_SHA"], "stage": e["STAGE"],
    "merged_from": e["MERGED_FROM"], "merged_sha": e["MERGED_SHA"], "merge_note": e["MERGE_NOTE"],
    "conflicts": [f for f in e["CONFLICTS"].splitlines() if f],
    "laboneq": e["LABONEQ"], "zhinst_core": e["ZHINST_CORE"], "python": e["PYVERSION"],
    "pytest_exit": e["PYTEST_EXIT"],
}
with open(os.path.join(e["OUT"], "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
EOF
}
trap write_meta EXIT

if [ -n "${MERGE_URL:-}" ]; then
  MERGED_FROM="${MERGE_URL#https://github.com/}@${MERGE_REF}"
  STAGE="fetch of ${MERGED_FROM} failed"
  git -C "$driver" fetch --no-tags "$MERGE_URL" "$MERGE_REF" || exit 1
  MERGED_SHA=$(git -C "$driver" rev-parse FETCH_HEAD)
  if git -C "$driver" merge-base --is-ancestor FETCH_HEAD HEAD; then
    MERGE_NOTE="nothing to merge, already contained"
  elif git -C "$driver" -c user.name=zi-driver-ci -c user.email=ci@invalid \
      merge --no-edit FETCH_HEAD; then
    MERGE_NOTE="clean merge"
  else
    STAGE="merge conflict"
    CONFLICTS=$(git -C "$driver" diff --name-only --diff-filter=U)
    echo "merge conflict in:"; echo "$CONFLICTS"
    git -C "$driver" merge --abort
    exit 1
  fi
fi

echo "conformance suite from $(git -C "$suite" rev-parse --short HEAD)"
mkdir -p "$driver/tests/instruments/zi/conformance"
cp -R "$suite/tests/instruments/zi/conformance/." "$driver/tests/instruments/zi/conformance/"

cd "$driver" || exit 1
STAGE="install failed"
uv sync --group zi --group tests --frozen || exit 1

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
