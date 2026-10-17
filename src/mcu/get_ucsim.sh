#!/bin/bash
# Fetch ucsim (s51) from the Ubuntu sdcc-ucsim 4.2.0 package WITHOUT installing it system-wide:
# the .deb is only unpacked into $1 (default: ./ucsim). Then: export S51=$1/usr/bin/s51
set -e
DEST=${1:-ucsim}
TMP=$(mktemp -d)
(cd "$TMP" && apt-get download sdcc-ucsim)
dpkg -x "$TMP"/sdcc-ucsim_*.deb "$DEST"
rm -rf "$TMP"
"$DEST/usr/bin/s51" -h 2>&1 | head -1
