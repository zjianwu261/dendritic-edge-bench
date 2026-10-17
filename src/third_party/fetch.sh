#!/bin/bash
# Clone the third-party repositories at the commits pinned in src/third_party_VERSIONS.txt.
# Usage: bash src/third_party/fetch.sh [name ...]      (default: all; e.g. "dendritic_anns visualnav-transformer")
# Existing checkouts are left alone if they are already at the pinned commit.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
ROOT=$(dirname "$(dirname "$HERE")")
VERSIONS="$ROOT/src/third_party_VERSIONS.txt"
want=" $* "
grep -v '^#' "$VERSIONS" | while IFS='|' read -r path url commit lic use; do
  path=$(echo "$path" | xargs); url=$(echo "$url" | xargs); commit=$(echo "$commit" | xargs)
  [ -z "$path" ] && continue
  name=$(basename "$path")
  if [ $# -gt 0 ] && [[ "$want" != *" $name "* ]]; then continue; fi
  dest="$ROOT/$path"
  if [ -d "$dest/.git" ]; then
    have=$(git -C "$dest" rev-parse HEAD)
    if [ "$have" = "$commit" ]; then echo "$name: already at ${commit:0:7}"; continue; fi
    echo "$name: exists at ${have:0:7}, checking out ${commit:0:7}"
  else
    echo "$name: cloning $url"
    git clone --quiet "$url" "$dest"
  fi
  git -C "$dest" -c advice.detachedHead=false checkout --quiet "$commit"
  echo "$name: at $(git -C "$dest" rev-parse --short=7 HEAD)"
done
