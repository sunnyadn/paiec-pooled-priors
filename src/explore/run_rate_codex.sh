#!/bin/bash
# run_rate_codex.sh DIR MODEL: rate every *.prompt in DIR with codex exec (subscription), 4 at a time
D=$(cd "$1" && pwd); export M=$2
one() { f=$1; o=${f%.prompt}.json; [ -s "$o" ] && return; (cd /tmp && codex exec -m "$M" -c model_reasoning_effort=low --skip-git-repo-check --sandbox read-only -o "$o" - < "$f" > "${f%.prompt}.err" 2>&1); }
export -f one
ls "$D"/*.prompt | xargs -P 4 -n 1 bash -c 'one "$0"'
ls "$D"/*.json | wc -l
