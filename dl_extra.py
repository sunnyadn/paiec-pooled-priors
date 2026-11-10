"""Tables of the 44 further benchmarks (Table 1 middle block, Figure 2) -> data/mdb_extra, at the revision pinned in dl.py.
Only the benchmarks listed under "keep" in data/llm/extra/semantics.json are needed. Gated: request access on Hugging Face first."""
import json, re
from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor
REV = re.search(r"REV='([0-9a-f]{40})'", open('dl.py').read()).group(1)
KEEP = set(json.load(open('data/llm/extra/semantics.json'))['keep'])
api = HfApi()   # one listing per benchmark: dataset_info() lists only the first 100,000 files of the repository, which stops before these
want = [e.path for b in sorted(KEEP) for e in api.list_repo_tree('aims-foundations/measurement-db', path_in_repo=f'{b}/formatted_tables', repo_type='dataset', revision=REV)
        if e.path.split('/')[-1] in ('responses.parquet', 'subjects.parquet', 'items.parquet')]
print(len(want), 'files for', len({f.split('/')[0] for f in want}), 'of', len(KEEP), 'benchmarks', flush=True)
def get(f):
    try: hf_hub_download('aims-foundations/measurement-db', f, repo_type='dataset', revision=REV, local_dir='data/mdb_extra'); return None
    except Exception as e: return f + ' ' + str(e)[:100]
with ThreadPoolExecutor(8) as ex:
    errs = [e for e in ex.map(get, want) if e]
print('errors', len(errs)); print('\n'.join(errs[:10]))
if errs: raise SystemExit(1)
