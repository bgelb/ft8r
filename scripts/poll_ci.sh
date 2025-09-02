#!/usr/bin/env bash
set -euo pipefail

# Poll the latest GitHub Actions run for a branch until completion,
# then print the ft8r decode metrics from the PR comments.

BRANCH="${1:-feat/dedup-neighborhood}"
PR_NUM="${2:-59}"
INTERVAL="${3:-10}"

echo "[poll] Watching latest CI run on branch '$BRANCH' (PR #$PR_NUM)"

run_id() {
  gh run list --branch "$BRANCH" --limit 1 --json databaseId --jq '.[0].databaseId' 2>/dev/null || true
}

status_of() {
  local rid="$1"
  gh run view "$rid" --json status --jq .status 2>/dev/null || true
}

RID="$(run_id)"
if [[ -z "$RID" ]]; then
  echo "[poll] No runs found for branch '$BRANCH' yet." >&2
  exit 1
fi
echo "[poll] Run ID: $RID"

while true; do
  ST="$(status_of "$RID")"
  TS="$(date -u +%H:%M:%S)"
  echo "[poll $TS] status=$ST"
  if [[ "$ST" == "completed" ]]; then
    break
  fi
  sleep "$INTERVAL"
  # In case a newer run was triggered, always follow the latest for the branch
  NEW_RID="$(run_id)"
  if [[ -n "$NEW_RID" && "$NEW_RID" != "$RID" ]]; then
    echo "[poll] Detected newer run: $NEW_RID (was $RID)"
    RID="$NEW_RID"
  fi
done

echo "[poll] Completed run: $RID"
echo "[poll] Summary:"
gh run view "$RID" --json conclusion,url,displayTitle --jq '.displayTitle + " | conclusion=" + .conclusion + " | " + .url'

echo "[poll] Decoding metrics from PR comments (marked block)"
# Print the ft8r metrics comment block if present
gh pr view "$PR_NUM" --comments | awk '/<!-- ft8r-metrics -->/{flag=1} flag{print} /^--$/{flag=0}'

