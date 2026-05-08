#!/usr/bin/env bash
# Bump pyproject.toml version, commit, and push to the current branch.
# Invoked by auto's @auto-it/exec plugin from `.autorc` for both the
# `version` (stable) and `next` (prerelease) hooks.
#
# Auto's git-tag plugin handles tagging — it tags HEAD after the version
# hook runs. We push the branch ourselves so the bump commit lands on
# origin before git-tag pushes the tag (otherwise the tag is created at
# the runner's local HEAD but the branch on origin stays stale).
#
# Usage: auto_bump.sh <new_version>
set -euo pipefail

if [ "$#" -ne 1 ]; then
  echo "usage: auto_bump.sh <new_version>" >&2
  exit 2
fi

NEW_VERSION="$1"
BRANCH="$(git rev-parse --abbrev-ref HEAD)"

echo "==> auto_bump.sh: bumping pyproject.toml to ${NEW_VERSION} on ${BRANCH}"

python3 scripts/bump_version.py "${NEW_VERSION}"

git add pyproject.toml

# Inline -c flags so we don't mutate persistent git config on the runner.
# `[skip release]` in the message prevents the bump commit from
# re-triggering release.yml / release-rc.yml.
git \
  -c user.email=actions@github.com \
  -c "user.name=github-actions[bot]" \
  commit -m "Bump version to ${NEW_VERSION} [skip release]"

git push origin "HEAD:${BRANCH}"

echo "==> auto_bump.sh: pushed bump commit to origin/${BRANCH}"
