#!/usr/bin/env bash
# Apply local fixes from patches/ to the engine sources in a build stage (both Dockerfiles run it).
#
#   apply-patches.sh <patches dir> <source root> "<PATCHES>"
#
# PATCHES: empty or "none" (the default): nothing, the build is upstream's own code.
#          "all": every *.patch, in name order.
#          a comma list: just those, by file name with or without .patch, or by number ("0001,0002").
# The patch files name paths as vendor/...; the source root is where vendor/'s contents are (/src in the build stage).
# Writes the applied names, comma-separated, to <source root>/PATCHES (empty for none) for BUILD.json.
set -euo pipefail

dir=$1 root=$2 want=$(echo "${3:-}" | tr -d '[:space:]')
picked=()
case "$want" in
  ""|none) ;;
  all) for p in "$dir"/*.patch; do [ -e "$p" ] && picked+=("$p"); done ;;
  *)
    IFS=, read -ra names <<< "$want"
    for n in "${names[@]}"; do
      [ -n "$n" ] || continue
      match=("$dir/${n%.patch}.patch")
      [ -f "${match[0]}" ] || match=("$dir/$n"-*.patch)        # a number: 0001 -> 0001-*.patch
      [ -f "${match[0]}" ] || { echo "PATCHES: no patch named $n in patches/" >&2; exit 1; }
      [ ${#match[@]} -eq 1 ] || { echo "PATCHES: $n matches more than one patch" >&2; exit 1; }
      picked+=("${match[0]}")
    done ;;
esac

applied=()
for p in "${picked[@]}"; do
  patch -d "$root" -p2 --forward --batch --no-backup-if-mismatch < "$p"
  applied+=("$(basename "$p" .patch)")
  echo "patch applied: $(basename "$p")"
done
[ ${#applied[@]} -gt 0 ] || echo "no patches: upstream's code as released"
(IFS=,; echo "${applied[*]}") > "$root/PATCHES"
