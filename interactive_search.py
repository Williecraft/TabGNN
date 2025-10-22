import os,sys,torch
import torch.nn.functional as F
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer=None
from train_model import DiffusionRetrievalGNN

def get_embedder():
    force = os.environ.get('FORCE_SEMANTIC','0') == '1'
    model_name = os.environ.get('EMBED_MODEL','distiluse-base-multilingual-cased-v2')
    device = os.environ.get('EMBED_DEVICE','cpu')
    class E:
        def __init__(self):
            self.ok=False
            self.m=None
            if SentenceTransformer:
                try:
                    self.m=SentenceTransformer(model_name)
                    try:
                        self.m.to(device)
                    except Exception:
                        pass
                    self.ok=True
                except Exception:
                    self.m=None
            if force and not self.ok:
                print('FORCE_SEMANTIC=1 但無法載入語義嵌入模型，請安裝 sentence-transformers 或提供可用模型名稱')
                sys.exit(1)
        def encode(self,t):
            if self.ok:
                return self.m.encode(t)
            import re,hashlib,numpy as np
            v=np.zeros(512,dtype=np.float32)
            for w in re.findall(r"\w+",str(t).lower()):
                h=int(hashlib.md5(w.encode()).hexdigest(),16)%512; v[h]+=1.0
            n=np.linalg.norm(v); return (v/n if n>0 else v)
    return E()

def load_graph(base):
    p=os.path.join(base,'graph_cache.pt')
    print('載入圖結構...')
    try:
        # PyTorch 2.6 default weights_only=True breaks HeteroData unpickling
        g=torch.load(p,map_location='cpu',weights_only=False)
        return g
    except Exception as e:
        print('載入圖失敗:',e); sys.exit(1)

def load_model(base):
    ck=os.path.join(base,'retrieval_model.pt')
    m=DiffusionRetrievalGNN(h=int(os.environ.get('HIDDEN_DIM','128')),a=0.2,k=10)
    if not os.path.exists(ck):
        print('找不到模型，請先執行 train_model.py'); sys.exit(1)
    try:
        sd=torch.load(ck,map_location='cpu'); m.load_state_dict(sd,strict=False)
    except Exception as e:
        print('載入模型失敗:',e); sys.exit(1)
    m.eval(); return m

def fmt(d,idx,score):
    tid=getattr(d['table'],'table_idx_to_id',{}).get(idx,str(idx))
    sh=getattr(d['table'],'sheet_names',[]) or []
    sheet=sh[idx] if idx<len(sh) else None
    return f"#{idx} | id={tid} | sheet={sheet} | score={float(score):.4f}"

def main():
    base=os.path.dirname(os.path.abspath(__file__))
    print('載入圖結構...')
    data=load_graph(base)
    print(f'完成（共 {data["table"].x.size(0)} 張表格）')
    print('載入模型...')
    model=load_model(base)
    emb=get_embedder()
    top=int(os.environ.get('TOPK','5'))
    envq=os.environ.get('QUERY')
    if envq:
        q=torch.tensor(emb.encode(envq),dtype=torch.float)
        with torch.no_grad(): s=model(data,q)
        k=torch.topk(s,k=min(top,s.numel()),largest=True)
        print(f'查詢: {envq}')
        for r,(i,sc) in enumerate(zip(k.indices.tolist(),k.values.tolist()),1):
            print(f'{r}. '+fmt(data,i,sc))
        return
    print('========================================')
    print('表格檢索系統 v1.0')
    print("輸入查詢或 'quit' 退出")
    print('========================================')
    while True:
        q=input('> ').strip()
        if not q: continue
        if q.lower() in ('quit','exit','q'): break
        qv=torch.tensor(emb.encode(q),dtype=torch.float)
        with torch.no_grad(): s=model(data,qv)
        k=torch.topk(s,k=min(top,s.numel()),largest=True)
        print('檢索結果（Top {}）:'.format(top))
        for r,(i,sc) in enumerate(zip(k.indices.tolist(),k.values.tolist()),1):
            print(f'{r}. '+fmt(data,i,sc))

if __name__=='__main__':
    main()