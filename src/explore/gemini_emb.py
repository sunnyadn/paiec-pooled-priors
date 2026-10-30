import json, os, time, urllib.request, numpy as np
K = open(os.path.expanduser('~/.config/gemini/key-personal')).read().strip()
D = json.load(open('/tmp/emb_data.json'))
URL = 'https://generativelanguage.googleapis.com/v1beta/models/gemini-embedding-001:batchEmbedContents?key=' + K
def batch(texts):
    body = {'requests': [{'model': 'models/gemini-embedding-001', 'content': {'parts': [{'text': t[:2000]}]}, 'outputDimensionality': 768} for t in texts]}
    for att in range(6):
        try:
            req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={'content-type': 'application/json'})
            o = json.load(urllib.request.urlopen(req, timeout=120)); return [e['values'] for e in o['embeddings']]
        except urllib.error.HTTPError as e:
            msg = e.read()[:200]
            if e.code in (429, 500, 503): time.sleep(10 * (att + 1)); continue
            raise RuntimeError(f'{e.code} {msg}')
    raise RuntimeError('retries exhausted')
def emb(texts):
    out = []
    for i in range(0, len(texts), 100):
        out += batch(texts[i:i + 100])
        if i % 5000 == 0: print(i, flush=True)
    a = np.array(out, dtype=np.float32); return a / np.linalg.norm(a, axis=1, keepdims=True)
np.save('/tmp/gem_test.npy', emb(D['test_text'])); np.save('/tmp/gem_train.npy', emb(D['train_text'])); print('done')
