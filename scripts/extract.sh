#!/usr/bin/env bash
# Take one upstream Strata release into vendor/: only the paths in manifest.txt.
#
#   ./scripts/extract.sh v0.2.10                         # from the cached mirror in .upstream/ (cloned on first use)
#   UPSTREAM_GIT=/srv/llm/strata/repo ./scripts/extract.sh v0.2.10    # from a clone you already have (no download)
#
# vendor/ is replaced as a whole, so `git diff vendor/` afterwards shows exactly what upstream changed in the
# parts this image uses. patches/*.patch (local fixes) are then applied on top and listed in vendor/UPSTREAM.
# Nothing is built here; docker build is next.
set -euo pipefail

here=$(cd "$(dirname "$0")/.." && pwd)     # the project root: manifest.txt, vendor/, .upstream/
cd "$here"

tag=${1:-}
[ -n "$tag" ] || { echo "usage: $0 <upstream tag, e.g. v0.2.10>" >&2; exit 2; }

UPSTREAM_URL=${UPSTREAM_URL:-https://github.com/Niko1221/Strata.git}
if [ -n "${UPSTREAM_GIT:-}" ]; then
  src=$UPSTREAM_GIT
else
  src=$here/.upstream
  if [ ! -d "$src" ]; then
    echo "[extract] cloning $UPSTREAM_URL into .upstream/ (once)"
    git clone --quiet --bare "$UPSTREAM_URL" "$src"
  fi
  echo "[extract] fetching tags from $UPSTREAM_URL"
  git -C "$src" fetch --quiet --tags --force "$UPSTREAM_URL" '+refs/heads/main:refs/heads/main'
fi

commit=$(git -C "$src" rev-parse --verify --quiet "$tag^{commit}") \
  || { echo "[extract] $tag is not a tag or commit in $src" >&2; exit 2; }

# manifest.txt -> pathspecs ("!x" -> ":(exclude)x")
specs=()
while IFS= read -r line || [ -n "$line" ]; do
  line=${line%%#*}; line=$(printf '%s' "$line" | sed 's/[[:space:]]*$//; s/^[[:space:]]*//')
  [ -n "$line" ] || continue
  case "$line" in
    '!'*) specs+=(":(exclude)${line#!}") ;;
    *)    specs+=("$line") ;;
  esac
done < manifest.txt

# every included (non-exclude) entry must exist upstream: a renamed folder should fail here, not in docker build
missing=0
for s in "${specs[@]}"; do
  case "$s" in ':(exclude)'*) continue ;; esac
  if [ -z "$(git -C "$src" ls-tree -r --name-only "$commit" -- "$s" | head -1)" ]; then
    echo "[extract] manifest entry not in $tag: $s" >&2; missing=1
  fi
done
[ "$missing" = 0 ] || { echo "[extract] fix manifest.txt (upstream moved or removed these)" >&2; exit 1; }

new=$here/vendor.new
rm -rf "$new"; mkdir -p "$new"
git -C "$src" archive --format=tar "$commit" -- "${specs[@]}" | tar -x -C "$new"

# the config keys the server reads (cfg.get("key") / cfg["key"] in serve/): launch.py warns about a CONFIG_<KEY> not among them
grep -rhoE 'cfg(\.get\(|\[)"[a-z_][a-z0-9_]*"' "$new"/serve/*.py | sed -E 's/.*"(.*)"/\1/' | LC_ALL=C sort -u > "$new/.config-keys"

engine_ver=$(sed -n 's/^[[:space:]]*project([^)]*VERSION[[:space:]]*\([0-9][0-9.]*\).*/\1/p' "$new/CMakeLists.txt" | head -1)
llama=$(awk 'NR==1{print $1}' "$new/third_party/ggml/VERSION.txt")
cat > "$new/UPSTREAM" <<EOF
tag=$tag
commit=$commit
date=$(git -C "$src" show -s --format=%cI "$commit")
engine_version=$engine_ver
llama_cpp_commit=$llama
EOF

rm -rf vendor
mv "$new" vendor

# local fixes on top of upstream (patches/*.patch, paths relative to the project root: vendor/...), in name order.
# One upstream already has (it applies in reverse) is skipped with a note - drop it from patches/. One that applies
# neither way stops here: upstream changed that code, so the patch needs a rebase.
applied=()
for p in patches/*.patch; do
  [ -e "$p" ] || continue
  if git apply --check "$p" 2>/dev/null; then
    git apply "$p"; applied+=("$(basename "$p")"); echo "[extract] patch applied: $p"
  elif git apply --reverse --check "$p" 2>/dev/null; then
    echo "[extract] patch already in $tag, skipped (remove it): $p"
  else
    echo "[extract] patch does not apply to $tag: $p (rebase it on the new vendor/)" >&2; exit 1
  fi
done
[ ${#applied[@]} -eq 0 ] || echo "patches=${applied[*]}" >> vendor/UPSTREAM

echo "[extract] vendor/ = $tag ($commit), engine $engine_ver, $(find vendor -type f | wc -l) files, $(du -sh vendor | cut -f1)"
echo "[extract] next: git diff --stat vendor/, then docker build"
