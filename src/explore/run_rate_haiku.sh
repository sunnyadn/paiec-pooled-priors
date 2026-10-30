#!/bin/bash
# run_rate.sh DIR: rate every *.prompt in DIR with claude sonnet (subscription), 6 at a time
D=$(cd "$1" && pwd)
one() { f=$1; o=${f%.prompt}.json; [ -s "$o" ] && return; (cd /tmp && claude -p --safe-mode --model haiku --output-format text < "$f" > "$o" 2> "${f%.prompt}.err"); }
export -f one
ls "$D"/*.prompt | xargs -P 6 -n 1 bash -c 'one "$0"'
ls "$D"/*.json | wc -l
