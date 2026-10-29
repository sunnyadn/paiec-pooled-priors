from huggingface_hub import HfApi, hf_hub_download
from concurrent.futures import ThreadPoolExecutor
info=HfApi().dataset_info('aims-foundations/measurement-db')
want=[s.rfilename for s in info.siblings if '/formatted_tables/' in s.rfilename and s.rfilename.split('/')[-1] in ('responses.parquet','subjects.parquet','benchmarks.parquet','items.parquet')]
print(len(want),'files',flush=True)
def get(f):
    try: hf_hub_download('aims-foundations/measurement-db',f,repo_type='dataset',local_dir='data/mdb'); return None
    except Exception as e: return f+' '+str(e)[:100]
with ThreadPoolExecutor(8) as ex:
    errs=[e for e in ex.map(get,want) if e]
print('errors',len(errs)); print('\n'.join(errs[:10]))
