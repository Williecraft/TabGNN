import os,io,json,random
import torch,torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv,SAGEConv,APPNP
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer=None

def get_embedder():
    class E:
        def __init__(self):
            import os
            self.force = os.environ.get('FORCE_SEMANTIC','0') == '1'
            self.model_name = os.environ.get('EMBED_MODEL','distiluse-base-multilingual-cased-v2')
            self.batch = int(os.environ.get('EMBED_BATCH','64'))
            self.device = os.environ.get('EMBED_DEVICE','cpu')
            self.ok = False
            self.m = None
            try:
                from sentence_transformers import SentenceTransformer as ST
            except Exception:
                ST = None
            if ST is not None:
                try:
                    self.m = ST(self.model_name, device=self.device)
                    self.ok = True
                except Exception:
                    self.m = None
            if self.force and not self.ok:
                raise RuntimeError('Semantic embedder unavailable but FORCE_SEMANTIC=1')
        def encode(self, t):
            return self.encode_batch([t])[0]
        def encode_batch(self, texts):
            if self.ok and self.m is not None:
                import numpy as np
                vs = []
                for i in range(0, len(texts), self.batch):
                    chunk = texts[i:i+self.batch]
                    v = self.m.encode(chunk, batch_size=self.batch, convert_to_numpy=True, show_progress_bar=False)
                    vs.append(v)
                return np.vstack(vs).astype('float32')
            # fallback: 512-d hashed embedding
            import re, hashlib, numpy as np
            arr = np.zeros((len(texts), 512), dtype=np.float32)
            for i, t in enumerate(texts):
                s = ' '.join(map(str, t)) if isinstance(t, list) else str(t)
                v = np.zeros(512, dtype=np.float32)
                for w in re.findall(r"\w+", s.lower()):
                    h = int(hashlib.md5(w.encode()).hexdigest(), 16) % 512
                    v[h] += 1.0
                n = np.linalg.norm(v)
                arr[i] = (v/n if n>0 else v)
            return arr
    return E()

def load_jsonl(p):
    with io.open(p,'r',encoding='utf-8') as f:
        for l in f:
            l=l.strip();
            if l: yield json.loads(l)

class DiffusionRetrievalGNN(nn.Module):
    def __init__(self,h=128,a=0.2,k=10):
        super().__init__()
        self.c1=HeteroConv({('table','has_column','column'):SAGEConv((-1,-1),h),('column','rev_has_column','table'):SAGEConv((-1,-1),h),('column','col_sim','column'):SAGEConv((-1,-1),h),('column','name_sim','column'):SAGEConv((-1,-1),h),('column','dist_sim','column'):SAGEConv((-1,-1),h),('table','sim','table'):SAGEConv((-1,-1),h)},aggr='sum')
        self.c2=HeteroConv({('table','has_column','column'):SAGEConv((-1,-1),h),('column','rev_has_column','table'):SAGEConv((-1,-1),h),('column','col_sim','column'):SAGEConv((-1,-1),h),('column','name_sim','column'):SAGEConv((-1,-1),h),('column','dist_sim','column'):SAGEConv((-1,-1),h),('table','sim','table'):SAGEConv((-1,-1),h)},aggr='sum')
        self.qp=nn.Linear(512,h); self.app=APPNP(K=k,alpha=a); self.lin=nn.Linear(h,1)
    def forward(self,data,q):
        xd=self.c1(data.x_dict,data.edge_index_dict)
        for k in xd: xd[k]=F.relu(xd[k])
        xd=self.c2(xd,data.edge_index_dict)
        xt=xd['table']; qv=self.qp(q)
        if data['table'].x.size(0)>0:
            xr=F.normalize(data['table'].x,p=2,dim=1); qr=F.normalize(q,p=2,dim=0)
            w=torch.clamp(torch.mv(xr,qr),min=0.0)
            z0=w.unsqueeze(1)*qv.unsqueeze(0)
        else: z0=torch.zeros_like(xt)
        ei=data['table','sim','table'].edge_index; ew=data['table','sim','table'].get('edge_weight',None)
        z=self.app(xt+z0,ei,ew) if ew is not None else self.app(xt+z0,ei)
        return self.lin(z).squeeze(-1)

def contrastive_loss(scores,pos,negs):
    lg=torch.cat([scores[pos].unsqueeze(0),scores[negs]]); lab=torch.zeros(1,dtype=torch.long)
    return F.cross_entropy(lg.view(1,-1),lab)

def evaluate(model,data,q_embs,pos_idx,k=5):
    model.eval(); rec=[]; rr=[]
    with torch.no_grad():
        for q,p in zip(q_embs,pos_idx):
            s=model(data,q); top=torch.topk(s,k=min(k,s.numel())).indices.tolist()
            rec.append(1.0 if p in top else 0.0)
            r=torch.argsort(torch.argsort(s,descending=True))[p].item()+1; rr.append(1.0/r)
    return sum(rec)/len(rec),sum(rr)/len(rr)

def main():
    base=os.path.dirname(os.path.abspath(__file__))
    dev_env=os.environ.get('EMBED_DEVICE','cuda' if torch.cuda.is_available() else 'cpu')
    dev=torch.device('cuda' if (dev_env=='cuda' and torch.cuda.is_available()) else 'cpu')
    gpath=os.path.join(base,'graph_cache.pt')
    qpath=os.path.join(base,'data','train','feta','generate_query.jsonl')
    try:
        data=torch.load(gpath,map_location='cpu',weights_only=False)
    except Exception as e:
        print('載入圖失敗:',e); return
    data=data.to(dev)
    emb=get_embedder()
    items=[]
    for o in load_jsonl(qpath):
        tid=o.get('table_id'); qs=[q for q in o.get('queries',[]) if str(q).strip()]
        if tid in getattr(data['table'],'table_id_to_idx',{}):
            pi=data['table'].table_id_to_idx[tid]
            for q in qs: items.append((q,pi))
    if not items:
        print('查詢資料為空'); return
    # limit number of queries from env
    NQ=int(os.environ.get('MAX_QUERIES','0'))
    if NQ>0:
        items=items[:NQ]
    qtxt=[q for q,_ in items]
    print(f'開始批次嵌入查詢向量，共 {len(qtxt)} 筆')
    # batch semantic embeddings
    qmat=emb.encode_batch(qtxt)
    qvec=[torch.tensor(v,dtype=torch.float).to(dev) for v in qmat]
    pidx=[p for _,p in items]
    print(f'載入查詢: {len(qvec)}，裝置: {dev}')
    H=int(os.environ.get('HIDDEN_DIM','128'))
    E=int(os.environ.get('EPOCHS','50'))
    LR=float(os.environ.get('LR','1e-3'))
    K=int(os.environ.get('NEG_SAMPLES','8'))
    model=DiffusionRetrievalGNN(h=H,a=0.2,k=10).to(dev)
    opt=torch.optim.Adam(model.parameters(),lr=LR)
    logp=os.path.join(base,'training_log.txt'); best=-1.0; wait=0; P=int(os.environ.get('PATIENCE','10'))
    ck=os.path.join(base,'retrieval_model.pt')
    if os.path.exists(ck):
        try:
            sd=torch.load(ck,map_location='cpu'); model.load_state_dict(sd,strict=False)
            print('從checkpoint恢復')
        except Exception: pass
    for ep in range(1,E+1):
        model.train(); tot=0.0
        for q,pos in zip(qvec,pidx):
            opt.zero_grad(); s=model(data,q)
            if ep<=10:
                neg=[]
                while len(neg)<K:
                    r=random.randrange(data['table'].x.size(0))
                    if r!=pos and r not in neg: neg.append(r)
            else:
                with torch.no_grad():
                    sc=model(data,q)
                    top=torch.topk(sc,k=min(K+1,sc.numel())).indices.tolist()
                    neg=[i for i in top if i!=pos][:K]
            loss=contrastive_loss(s,pos,neg); loss.backward(); opt.step(); tot+=loss.item()
        if ep%5==0:
            rec,mrr=evaluate(model,data,qvec,pidx,k=int(os.environ.get('TOPK','5')))
            print(f'Epoch {ep}/{E} | Loss {tot:.4f} | Recall@5 {rec:.4f} | MRR {mrr:.4f}')
            with open(logp,'a',encoding='utf-8') as f: f.write(f'{ep}\t{tot:.4f}\t{rec:.4f}\t{mrr:.4f}\n')
            if mrr>best: best=mrr; wait=0; torch.save(model.state_dict(),ck)
            else:
                wait+=1
                if wait>=P:
                    print('Early stopping'); break
        else:
            print(f'Epoch {ep}/{E} | Loss {tot:.4f}')
    if not os.path.exists(ck): torch.save(model.state_dict(),ck)
    print('訓練完成，模型已保存')

if __name__=='__main__':
    main()