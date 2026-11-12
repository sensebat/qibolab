#!/usr/bin/env bash
# First job of the zi-driver workflow: resolve the commits and laboneq versions of this run,
# decide whether it tests anything, and list the cells.
#
# key = R1 commit + R2 commit + orkun/ci commit + newest laboneq on PyPI. The key of the last
# published report is in STATE_URL (state.json next to index.html). Scheduled runs, zi-check
# repository dispatches and dispatches with only_if_changed stop when the key is unchanged;
# every other run tests.
#
# Rows: r1 and r2; a push to orkun/26.10 tests only r2. An extra row x (no merge) runs the
# branch itself on orkun/proto-* branches, or EXTRA_REF from a dispatch. Runs with an extra
# row always test and never publish Pages.
# Targets (must be green): r1-c1 (R1 x pinned) and r2-c2 (R2 x latest); the rest is information.
#
# Env: EVENT, REF_NAME, SHA, REPO   github.event_name, github.ref_name, github.sha, github.repository
#      ONLY_IF_CHANGED              "true" makes a dispatch behave like a scheduled run
#      EXTRA_REPO_IN, EXTRA_REF_IN  dispatch inputs: run this ref as the extra row
#      R1_REPO R1_REF R2_REPO R2_REF SUITE_REPO SUITE_REF STATE_URL PROTO_PREFIX
# Output: key=value lines to $GITHUB_OUTPUT (stdout when unset), a note to $GITHUB_STEP_SUMMARY.
set -euo pipefail

out=${GITHUB_OUTPUT:-/dev/stdout}
summary=${GITHUB_STEP_SUMMARY:-/dev/stderr}
PROTO_PREFIX=${PROTO_PREFIX:-orkun/proto-}

head_of() { git ls-remote --exit-code "https://github.com/$1" "refs/heads/$2" | cut -f1; }

# A branch, a tag or a full commit sha of a GitHub repository.
resolve() {
  if [[ $2 =~ ^[0-9a-f]{40}$ ]]; then echo "$2"; return; fi
  git ls-remote --exit-code "https://github.com/$1" "refs/heads/$2" "refs/tags/$2" | head -1 | cut -f1
}

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

extra_repo="" extra_ref="" extra_sha=""
if [ -n "${EXTRA_REF_IN:-}" ]; then
  extra_repo=${EXTRA_REPO_IN:-$REPO}
  extra_ref=$EXTRA_REF_IN
  extra_sha=$(resolve "$extra_repo" "$extra_ref") || {
    echo "::error::extra ref not found: $extra_repo@$extra_ref"
    exit 1
  }
elif [[ $REF_NAME == "$PROTO_PREFIX"* ]]; then
  extra_repo=$REPO extra_ref=$REF_NAME extra_sha=$SHA
fi

# ci scripts: from the run's own commit on ci branches (orkun/ci, orkun/rc, ...); from orkun/ci
# on driver branches (orkun/26.10, orkun/proto-*), whose copies may lag behind.
if [ "$REF_NAME" = "$R2_REF" ] || [[ $REF_NAME == "$PROTO_PREFIX"* ]]; then
  scripts_sha=$ci_sha
else
  scripts_sha=$SHA
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
if [ -z "$extra_sha" ]; then
  if [ "$EVENT" = schedule ] || [ "$EVENT" = repository_dispatch ] || [ "${ONLY_IF_CHANGED:-false}" = true ]; then
    if [ "$key" = "$published" ]; then run=false; fi
  fi
fi

if [ "$EVENT" = push ] && [ "$REF_NAME" = "$R2_REF" ]; then rows="r2"; else rows="r1 r2"; fi
if [ -n "$extra_sha" ]; then rows="$rows x"; fi

deploy=false
if [ "$REF_NAME" = "$SUITE_REF" ] && [ "$rows" = "r1 r2" ]; then deploy=true; fi

export RUN=$run ROWS=$rows R1_SHA=$r1_sha R2_SHA=$r2_sha PINNED=$pinned LATEST=$latest
export EXTRA_REPO=$extra_repo EXTRA_REF=$extra_ref EXTRA_SHA=$extra_sha
matrix=$(python3 - <<'EOF'
import json, os
e = os.environ
targets = {"r1-c1", "r2-c2"}
code = {
    "r1": (e["R1_REPO"], e["R1_REF"], e["R1_SHA"], ""),
    "r2": (e["R2_REPO"], e["R2_REF"], e["R2_SHA"], f" + {e['R1_REF']}"),
    "x": (e["EXTRA_REPO"], e["EXTRA_REF"], e["EXTRA_SHA"], ""),
}
env = {"c1": f"pinned {e['PINNED']}", "c2": f"latest {e['LATEST']}"}
cells = []
for r in e["ROWS"].split():
    repo, ref, sha, merged = code[r]
    for c in ("c1", "c2"):
        cell = f"{r}-{c}"
        role = "target" if cell in targets else "information"
        cells.append({
            "cell": cell, "row": r, "col": c, "target": cell in targets,
            "repo": repo, "sha": sha, "depth": 0 if r == "r2" else 1,
            "name": f"{r.upper()} x {env[c]} · {role} · {repo}@{ref} ({sha[:8]}){merged}",
        })
if e["RUN"] != "true":  # one placeholder, so the skipped cell job has a readable name
    cells = [{"cell": "none", "skip": True, "name": "Cells: skipped, nothing changed"}]
print(json.dumps({"include": cells}))
EOF
)

echo "key:           $key"
echo "published key: ${published:-none}"
echo "event $EVENT, only_if_changed ${ONLY_IF_CHANGED:-false} -> run=$run, rows: $rows, deploy=$deploy"

{
  echo "run=$run"
  echo "deploy=$deploy"
  echo "key=$key"
  echo "r1_sha=$r1_sha"
  echo "r2_sha=$r2_sha"
  echo "ci_sha=$ci_sha"
  echo "scripts_sha=$scripts_sha"
  echo "extra_repo=$extra_repo"
  echo "extra_ref=$extra_ref"
  echo "extra_sha=$extra_sha"
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
  if [ -n "$extra_sha" ]; then
    echo "| X (extra, no merge) | \`$extra_repo@$extra_ref\` ${extra_sha:0:8} |"
  fi
  echo "| conformance suite | \`$SUITE_REPO@$SUITE_REF\` ${ci_sha:0:8} |"
  echo "| ci scripts | ${scripts_sha:0:8} |"
  echo "| laboneq pinned (R1 uv.lock) | $pinned |"
  echo "| laboneq latest (PyPI) | $latest |"
  echo ""
  echo "Targets: R1 x pinned and R2 x latest. Other cells are information."
  echo ""
  echo "Key: \`$key\`"
  echo ""
  echo "Published key: \`${published:-none}\`"
  echo ""
  if [ "$run" = true ]; then
    echo "Decision: test rows $rows (event $EVENT). Pages: $deploy."
  else
    echo "Decision: unchanged since the published report, nothing to test (event $EVENT)."
  fi
} >> "$summary"
