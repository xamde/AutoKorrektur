#!/usr/bin/env bash
# Downloads the large build inputs listed in scripts/assets.manifest (ML models and the
# instrumented-test fixtures) into the working tree. They are deliberately not stored in git; see
# the manifest's header for the rationale and the format.
#
#   scripts/fetch_assets.sh            download whatever is missing or stale (idempotent)
#   scripts/fetch_assets.sh --check    verify only, download nothing; non-zero exit if incomplete
#   scripts/fetch_assets.sh --force    re-download everything, even if it is already there
#
# Every download is verified against the SHA-256 in the manifest before it is installed, so a
# truncated transfer or a tampered mirror fails loudly instead of producing a broken build.
# Downloads are cached in .assets-cache/ (gitignored) so a re-run costs nothing.
#
# Needs: curl, sha256sum, tar with zstd support (tar --zstd, GNU tar >= 1.31).
set -euo pipefail

repo="$(cd "$(dirname "$0")/.." && pwd)"
manifest="$repo/scripts/assets.manifest"
cache="$repo/.assets-cache"

mode=fetch
case "${1:-}" in
  --check) mode=check ;;
  --force) mode=force ;;
  -h|--help) sed -n '2,15p' "$0" | sed 's/^# \?//'; exit 0 ;;
  "") ;;
  *) echo "fetch_assets.sh: unknown option '$1' (try --help)" >&2; exit 2 ;;
esac

[ -f "$manifest" ] || { echo "fetch_assets.sh: $manifest not found" >&2; exit 2; }
for tool in curl sha256sum tar; do
  command -v "$tool" >/dev/null || { echo "fetch_assets.sh: $tool is required but not installed" >&2; exit 2; }
done

sha_of() { sha256sum "$1" | cut -d' ' -f1; }

# Downloads $url into the cache and verifies it against $sha; echoes the cached path.
cached_download() {
  local sha=$1 url=$2 name
  name="$(basename "$url")"
  local out="$cache/$sha-$name"
  if [ -f "$out" ] && [ "$(sha_of "$out")" = "$sha" ]; then
    echo "$out"; return 0
  fi
  mkdir -p "$cache"
  echo "  downloading $name" >&2
  # -C - resumes a partial file; the hash check below is what actually decides acceptance.
  # A progress bar on a terminal, quiet in CI logs (which are not a TTY).
  local progress=(--progress-bar); [ -t 2 ] || progress=(--silent --show-error)
  curl -fL --retry 3 --retry-delay 2 -C - "${progress[@]}" -o "$out.part" "$url" >&2 || {
    rm -f "$out.part"
    echo "fetch_assets.sh: download failed: $url" >&2; return 1
  }
  local got; got="$(sha_of "$out.part")"
  if [ "$got" != "$sha" ]; then
    rm -f "$out.part"
    echo "fetch_assets.sh: SHA-256 mismatch for $name" >&2
    echo "  expected $sha" >&2
    echo "  got      $got" >&2
    echo "  Refusing to install it. If the asset was republished on purpose, update the hash in" >&2
    echo "  scripts/assets.manifest in the same commit." >&2
    return 1
  fi
  mv "$out.part" "$out"
  echo "$out"
}

missing=()
installed=0
ok=0

while read -r kind dest sha url extra || [ -n "${kind:-}" ]; do
  case "${kind:-}" in ''|'#'*) continue ;; esac
  [ -n "${url:-}" ] || { echo "fetch_assets.sh: malformed manifest line for '$dest'" >&2; exit 2; }
  target="$repo/$dest"

  case "$kind" in
    file)
      if [ "$mode" != force ] && [ -f "$target" ] && [ "$(sha_of "$target")" = "$sha" ]; then
        ok=$((ok + 1)); continue
      fi
      if [ "$mode" = check ]; then missing+=("$dest"); continue; fi
      src="$(cached_download "$sha" "$url")"
      mkdir -p "$(dirname "$target")"
      cp "$src" "$target"
      installed=$((installed + 1))
      ;;
    archive)
      # A stamp holding the archive's hash: re-hashing every extracted file on each build would
      # be slow, and the archive hash is what identifies the contents anyway.
      stamp="$target/.assets-stamp"
      if [ "$mode" != force ] && [ -f "$stamp" ] && [ "$(cat "$stamp")" = "$sha" ]; then
        ok=$((ok + 1)); continue
      fi
      if [ "$mode" = check ]; then missing+=("$dest/"); continue; fi
      src="$(cached_download "$sha" "$url")"
      parent="$(dirname "$target")"
      mkdir -p "$parent"
      # shellcheck disable=SC2086 # $extra holds optional tar flags from the manifest, intentionally split
      tar --zstd -xf "$src" -C "$parent" ${extra:-}
      echo "$sha" > "$stamp"
      installed=$((installed + 1))
      ;;
    *)
      echo "fetch_assets.sh: unknown kind '$kind' in manifest" >&2; exit 2 ;;
  esac
done < "$manifest"

if [ "$mode" = check ]; then
  if [ ${#missing[@]} -gt 0 ]; then
    echo "Missing or out-of-date build assets:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    echo >&2
    echo "Run: scripts/fetch_assets.sh" >&2
    exit 1
  fi
  echo "fetch_assets.sh: all $ok assets present and verified"
  exit 0
fi

echo "fetch_assets.sh: $installed installed, $ok already up to date"
