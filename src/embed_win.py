import json, sys, torch, numpy as np
from transformers import AutoTokenizer, AutoModel
name = sys.argv[1]; D = json.load(open('/home/sunny/paiec_emb/emb_data.json'))
tok = AutoTokenizer.from_pretrained(name, padding_side='left'); model = AutoModel.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()
TASK = 'Instruct: Represent this test item so that its difficulty for AI models can be estimated\nQuery: '
def emb(texts, bs=16):
    out = []
    for i in range(0, len(texts), bs):
        b = tok([TASK + t for t in texts[i:i + bs]], padding=True, truncation=True, max_length=512, return_tensors='pt').to('cuda')
        with torch.no_grad():
            h = model(**b).last_hidden_state[:, -1]
        out.append(torch.nn.functional.normalize(h.float(), dim=-1).cpu().numpy())
        if i % 2000 == 0: print(i, flush=True)
    return np.concatenate(out)
np.save('/home/sunny/paiec_emb/train.npy', emb(D['train_text'])); np.save('/home/sunny/paiec_emb/test.npy', emb(D['test_text'])); print('done')
