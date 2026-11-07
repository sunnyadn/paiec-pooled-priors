#!/bin/bash
# usage: tools/make_zip.sh MODEL_PY KEYFILE|default OUTNAME  -> dist/OUTNAME.zip (key inside; never publish)
set -e
M=$1; K=$2; N=$3; D=$(mktemp -d)
[ -f submission/anchors.json ] || grep -q "^ANCHORS = \[{" "$M" || { echo "submission/anchors.json missing: run src/make_anchors.py first"; exit 1; }
python3 tools/enable_prompt_cache.py "$M" "$D/model.py"; cp submission/labeling.py $D/; [ -f submission/anchors.json ] && cp submission/anchors.json $D/
if [ "$K" = default ]; then cp submission/llm_config.json $D/; else python3 -c "import json,sys; c=json.load(open('submission/llm_config.json')); c['key']=open(sys.argv[1]).read().strip(); json.dump(c,open(sys.argv[2],'w'))" "$K" $D/llm_config.json; fi
[ -n "$OPENAI_KEYFILE" ] && python3 -c "import json,sys; c=json.load(open(sys.argv[1])); c['openai_key']=open(sys.argv[2]).read().strip(); json.dump(c,open(sys.argv[1],'w'))" $D/llm_config.json "$OPENAI_KEYFILE"
(cd $D && zip -q out.zip model.py labeling.py llm_config.json $( [ -f $D/anchors.json ] && echo anchors.json )) && mv $D/out.zip dist/$N.zip && rm -rf $D
${PYTHON:-python3} ${CHECK:-../paiec_baseline/check_submission_zip.py} --static-only dist/$N.zip | tail -1 | cut -c1-60
