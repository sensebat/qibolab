#!/usr/bin/env bash
# First job of the zi-driver workflow: resolve the commits and laboneq versions of this run,
# decide whether it tests anything, and list the cells.
#
# key = R1 commit + R2 commit + orkun/ci commit + newest laboneq on PyPI. The key of the last
# published report is in STATE_URL (state.json next to index.html). Scheduled runs (and
# dispatches with only_if_changed) stop when the key is unchanged; every other run tests.
# A push to orkun/26.10 tests only the two R2 cells.
#
# Env: EVENT, REF_NAME, SHA         github.event_name, github.ref_name, github.sha
#      ONLY_IF_CHANGED              "true" makes a dispatch behave like a scheduled run
#      R1_REPO R1_REF R2_REPO R2_REF SUITE_REPO SUITE_REF STATE_URL
# Output: key=value lines to $GITHUB_OUTPUT (stdout when unset), a note to $GITHUB_STEP_SUMMARY.
set -euo pipefail

out=${GITHUB_OUTPUT:-/dev/stdout}
summary=${GITHUB_STEP_SUMMARY:-/dev/stderr}

head_of() { git ls-remote --exit-code "https://github.com/$1" "refs/heads/$2" | cut -f1; }

r1_sha=$(head_of "$R1_REPO" "$R1_REF")
if [ "$EVENT" = push ] && [ "$REF_NAME" = "$R2_REF" ]; then
  r2_sha=$SHA
else
  r2_sha=$(head_of "$R2_REPO" "$R2_REF")
fi
if [ "$REF_NAME" = "$SUITE_REF" ]; then
  ci_sha=$SHA
else
  ci_sha=$(head_of "$SUITE_REPO" "$SUITE_REF")
fi

pinned=$(curl -fsS "https://raw.githubusercontent.com/$R1_REPO/$r1_sha/uv.lock" | python3 -c '
import sys, tomllib
print(next(p["version"] for p in tomllib.loads(sys.stdin.read())["package"] if p["name"] == "laboneq"))')
latest=$(curl -fsS https://pypi.org/pypi/laboneq/json \
  | python3 -c 'import json, sys; print(json.load(sys.stdin)["info"]["version"])')

key="r1=$r1_sha r2=$r2_sha ci=$ci_sha laboneq=$latest"
# The query string bypasses the Pages cache (max-age 600).
published=$(curl -fsS "$STATE_URL?t=$(date +%s)" 2>/dev/null \
  | python3 -c 'import json, sys; print(json.load(sys.stdin).get("key", ""))' 2>/dev/null || true)

run=true
if [ "$EVENT" = schedule ] || [ "${ONLY_IF_CHANGED:-false}" = true ]; then
  if [ "$key" = "$published" ]; then run=false; fi
fi

if [ "$EVENT" = push ] && [ "$REF_NAME" = "$R2_REF" ]; then rows="r2"; else rows="r1 r2"; fi
export ROWS=$rows R1_SHA=$r1_sha R2_SHA=$r2_sha PINNED=$pinned LATEST=$latest
matrix=$(python3 - <<'EOF'
import json, os
e = os.environ
r1 = f"{e['R1_REPO']}@{e['R1_REF']} ({e['R1_SHA'][:8]})"
r2 = f"{e['R2_REPO']}@{e['R2_REF']} ({e['R2_SHA'][:8]}) + {r1} merged"
code = {"r1": r1, "r2": r2}
env = {"c1": f"laboneq {e['PINNED']} (pinned)", "c2": f"laboneq {e['LATEST']} (latest)"}
cells = [
    {"cell": f"{r}-{c}", "row": r, "col": c, "name": f"{r.upper()} x {c.upper()}: {code[r]} on {env[c]}"}
    for r in e["ROWS"].split()
    for c in ("c1", "c2")
]
print(json.dumps({"include": cells}))
EOF
)

echo "key:           $key"
echo "published key: ${published:-none}"
echo "event $EVENT, only_if_changed ${ONLY_IF_CHANGED:-false} -> run=$run, rows: $rows"

{
  echo "run=$run"
  echo "key=$key"
  echo "r1_sha=$r1_sha"
  echo "r2_sha=$r2_sha"
  echo "ci_sha=$ci_sha"
  echo "pinned=$pinned"
  echo "latest=$latest"
  echo "matrix=$matrix"
} >> "$out"

{
  echo "### Commits and versions"
  echo ""
  echo "| | |"
  echo "|---|---|"
  echo "| R1 | \`$R1_REPO@$R1_REF\` ${r1_sha:0:8} |"
  echo "| R2 | \`$R2_REPO@$R2_REF\` ${r2_sha:0:8} + R1 merged |"
  echo "| conformance suite | \`$SUITE_REPO@$SUITE_REF\` ${ci_sha:0:8} |"
  echo "| laboneq pinned (R1 uv.lock) | $pinned |"
  echo "| laboneq latest (PyPI) | $latest |"
  echo ""
  echo "Key: \`$key\`"
  echo ""
  echo "Published key: \`${published:-none}\`"
  echo ""
  if [ "$run" = true ]; then
    echo "Decision: test rows $rows (event $EVENT)."
  else
    echo "Decision: unchanged since the published report, nothing to test (event $EVENT)."
  fi
} >> "$summary"
