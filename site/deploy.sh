#!/usr/bin/env bash
# Builds the site (site/build.sh) into a temporary directory and rsyncs it to the server's
# docroot, which the infrastructure repo (~/files/work/server, role autokorrektur) owns: Caddy,
# the domain, the headers and the certificate live there, this script only delivers files.
# Same shape as Laberampel's site/deploy.sh.
#
# Environment (defaults from site/site.env, then the gitignored site/deploy.local.env):
#   VPS_HOST   user@<server>   (key via ~/.ssh/config) -- deploy.local.env only, the repo is public
#   VPS_WWW    /srv/autokorrektur/www
#   DRY_RUN=1  show what rsync would do, upload nothing
set -euo pipefail
cd "$(dirname "$0")"
. ./site.env
[ -f ./deploy.local.env ] && . ./deploy.local.env
VPS_HOST=${VPS_HOST:?set VPS_HOST=user@host in site/deploy.local.env (gitignored)}
VPS_WWW=${VPS_WWW:-/srv/autokorrektur/www}

out=$(mktemp -d)
trap 'rm -rf "$out"' EXIT
./build.sh "$out"
echo "deploy.sh: https://$SITE_DOMAIN <- $VPS_HOST:$VPS_WWW"
rsync -a --delete --info=stats1 ${DRY_RUN:+--dry-run} "$out/" "$VPS_HOST:$VPS_WWW/" |
  grep -E 'Number of (regular files transferred|deleted)|Total transferred'
