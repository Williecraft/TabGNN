import os,io,csv,json,re,hashlib,math
import torch,torch.nn.functional as F
from torch_geometric.data import HeteroData
import numpy as np
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer=None

# ---------------- Utils ----------------

def load_jsonl(p):
    with io.open(p,'r',encoding='utf-8') as f:
        for l in f:
            l=l.strip()
            if l: yield json.loads(l)


def parse_csv_line(s):
    return next(csv.reader([s]))


class Embedder:
    def __init__(self):
        self.force = bool(int(os.environ.get('FORCE_SEMANTIC','0')))
        self.device = os.environ.get('EMBED_DEVICE','cuda' if torch.cuda.is_available() else 'cpu')
        self.bs = int(os.environ.get('EMBED_BATCH','64'))
        self.ok=False
        if SentenceTransformer:
            try:
                model_name = os.environ.get('EMBED_MODEL','distiluse-base-multilingual-cased-v2')
                self.m=SentenceTransformer(model_name, device=self.device)
                self.ok=True
                print(f'使用語意模型: {model_name} | device={self.device} | batch={self.bs}')
            except Exception as e:
                self.m=None
                if self.force:
                    raise RuntimeError(f'無法載入語意模型：{e}')
    def encode(self,t):
        if self.ok:
            if isinstance(t,(list,tuple)):
                return self.m.encode(list(t), batch_size=self.bs, convert_to_numpy=True, show_progress_bar=False)
            return self.m.encode(t, batch_size=self.bs, convert_to_numpy=True, show_progress_bar=False)
        if self.force:
            raise RuntimeError('未載入語意模型（FORCE_SEMANTIC=1）')
        # hash fallback 512-d
        v=np.zeros(512,dtype=np.float32)
        if isinstance(t,list): t=' '.join(map(str,t))
        for w in re.findall(r"\w+",str(t).lower()):
            h=int(hashlib.md5(w.encode()).hexdigest(),16)%512
            v[h]+=1.0
        n=np.linalg.norm(v)
        return (v/n if n>0 else v)


# ---------------- Core build ----------------

def build_graph(table_jsonl_path,topk_t=10,topk_c=3,topk_name=5,topk_dist=5,sim_chunk=None,log_every=None):
    """
    構建異質圖（支援分批相似度計算）。
    - 分批計算 table 的 top-K 相似邊，以避免 O(N^2) 內存爆炸
    - 欄位內容相似度：保留表內 all-pairs；全局 top-K 省略以提升效率（大規模數據）
    """
    if sim_chunk is None:
        sim_chunk=int(os.environ.get('SIM_CHUNK','512'))
    if log_every is None:
        log_every=int(os.environ.get('LOG_EVERY_CHUNK','1'))
    emb=Embedder()

    # 讀取所有表
    tables=[]
    mt=int(os.environ.get('MAX_TABLES','0'))
    for i,o in enumerate(load_jsonl(table_jsonl_path),1):
        tables.append(o)
        if i%100==0: print(f"已處理 {i} 張表格...")
        if mt>0 and i>=mt:
            print(f"已達 MAX_TABLES 限制: {mt}")
            break
    tids=[t.get('id',str(i)) for i,t in enumerate(tables)]
    t2i={tid:i for i,tid in enumerate(tids)}
    sheets=[str(t.get('sheet_name','')) for t in tables]

    # 生成表與欄位特徵（批次嵌入）
    t_feats=[]; c_feats=[]; name_embs=[]; c_tbl_idx=[]; c_hist=[]; c_is_num=[]
    t2cols={}; key2i={}

    # 收集待嵌入文本
    t_texts=[]
    col_entries=[]  # {key, ti, c_txt, n_txt}
    for ti,t in enumerate(tables):
        sheet=sheets[ti]
        hdr=t.get('header',[])
        hdr=parse_csv_line(hdr[0]) if hdr else []
        inst=[parse_csv_line(r) for r in t.get('instances',[])]
        parts=[sheet]+hdr
        for r in inst[:10]: parts+=r
        t_texts.append(', '.join(map(lambda x:str(x).strip(),parts)))
        cs=[]
        for j,cn in enumerate(hdr):
            vals=[]
            for r in inst[:10]:
                if j<len(r): vals.append(str(r[j]))
            # 數值分布統計
            def isf(s):
                try: float(s); return True
                except: return False
            nums=[float(v) for v in vals if isf(v)]
            is_num=1.0 if nums else 0.0
            c_is_num.append(is_num==1.0)
            if nums:
                arr=np.array(nums,dtype=np.float32)
                b=16; mn=float(arr.min()); mx=float(arr.max())
                cnt=np.zeros(b,dtype=np.float32)
                if mx==mn: cnt[0]=float(arr.size)
                else:
                    rng=mx-mn
                    for x in arr:
                        k=int((float(x)-mn)/rng*b); k=min(max(k,0),b-1); cnt[k]+=1
                if cnt.sum()>0: cnt=cnt/cnt.sum()
                c_hist.append(torch.tensor(cnt,dtype=torch.float))
            else: c_hist.append(None)
            key=f"{tids[ti]}::{j}::{cn}"
            cs.append(key)
            col_entries.append({'key':key,'ti':ti,'c_txt':', '.join([str(cn)]+vals),'n_txt':str(cn)})
        t2cols[tids[ti]]=cs

    # 批次語意嵌入
    print('開始批次嵌入特徵...')
    t_vecs=emb.encode(t_texts) if t_texts else []
    c_vecs=emb.encode([e['c_txt'] for e in col_entries]) if col_entries else []
    n_vecs=emb.encode([e['n_txt'] for e in col_entries]) if col_entries else []

    # 建立索引與特徵張量
    for idx,e in enumerate(col_entries):
        key2i[e['key']]=idx
        c_tbl_idx.append(e['ti'])
        c_feats.append(torch.tensor(c_vecs[idx],dtype=torch.float))
        name_embs.append(torch.tensor(n_vecs[idx],dtype=torch.float))
    t_feats=[torch.tensor(v,dtype=torch.float) for v in (t_vecs if t_vecs is not None else [])]

    d=HeteroData()
    d['table'].x=torch.stack(t_feats) if t_feats else torch.empty((0,512))
    d['column'].x=torch.stack(c_feats) if c_feats else torch.empty((0,512))
    d['table'].table_ids=tids; d['table'].sheet_names=sheets
    d['table'].table_id_to_idx=t2i; d['table'].table_idx_to_id={i:tid for tid,i in t2i.items()}

    # has_column / rev_has_column
    ts,td=[],[]
    for tid,cks in t2cols.items():
        for ck in cks:
            ts.append(t2i[tid]); td.append(key2i[ck])
    if ts:
        d['table','has_column','column'].edge_index=torch.tensor([ts,td],dtype=torch.long)
        d['column','rev_has_column','table'].edge_index=torch.tensor([td,ts],dtype=torch.long)

    # Normalize
    with torch.no_grad():
        xt=F.normalize(d['table'].x,p=2,dim=1) if d['table'].x.size(0)>0 else d['table'].x
        xc=F.normalize(d['column'].x,p=2,dim=1) if d['column'].x.size(0)>0 else d['column'].x

    # -------- 分批計算 table sim top-K --------
    print('開始分批計算表格相似度邊...')
    N=xt.size(0)
    sim_src,sim_dst,sim_w=[],[],[]
    if N>0:
        # 以 chunk 行塊與整體做乘法，提取每行 top-K
        for s in range(0,N,sim_chunk):
            e=min(s+sim_chunk,N)
            block=xt[s:e]  # [B,512]
            # [B,N]
            sims=torch.matmul(block,xt.T)
            # 排除自身
            idx=torch.arange(s,e)
            sims[torch.arange(e-s),idx]=float('-inf')
            top=torch.topk(sims,k=min(topk_t,N),largest=True)
            top_idx=top.indices.cpu().tolist()
            top_val=top.values.cpu().tolist()
            for row,i0 in enumerate(range(s,e)):
                for j,w in zip(top_idx[row],top_val[row]):
                    sim_src.append(i0); sim_dst.append(j); sim_w.append(max(0.0,float(w)))
            if (s//sim_chunk)%10==0:
                print(f"表格相似度進度: {e}/{N}")
    if sim_src:
        d['table','sim','table'].edge_index=torch.tensor([sim_src,sim_dst],dtype=torch.long)
        d['table','sim','table'].edge_weight=torch.tensor(sim_w,dtype=torch.float)
        print(f"表格相似邊數: {len(sim_src)}")

    # -------- 欄位內容相似度：僅表內 all-pairs + 近鄰近似（可選） --------
    print('生成欄位內容相似度（表內 all-pairs）...')
    cs,cd=[],[]
    for tid,cks in t2cols.items():
        idx=[key2i[k] for k in cks]
        for a in idx:
            for b in idx:
                if a!=b:
                    cs.append(a); cd.append(b)
    if cs:
        d['column','col_sim','column'].edge_index=torch.tensor([cs,cd],dtype=torch.long)

    # -------- 欄位名稱相似度（跨表） --------
    print('計算欄位名稱相似度（跨表）...')
    ns,nd=[],[]
    if name_embs:
        nm=torch.stack([F.normalize(e,p=2,dim=0) for e in name_embs])
        M=nm.size(0)
        # 分批行塊以避免 O(M^2) 內存
        for s in range(0,M,sim_chunk):
            e=min(s+sim_chunk,M)
            block=nm[s:e]  # [B,512]
            sims=torch.matmul(block,nm.T)  # [B,M]
            idx=torch.arange(s,e)
            sims[torch.arange(e-s),idx]=float('-inf')
            top=torch.topk(sims,k=min(topk_name,M),largest=True)
            top_idx=top.indices.cpu().tolist()
            for row,i0 in enumerate(range(s,e)):
                for j in top_idx[row]:
                    if c_tbl_idx[i0]!=c_tbl_idx[j]:
                        ns.append(i0); nd.append(j)
            if ((s//sim_chunk) % log_every)==0 or e==M:
                print(f"欄位名稱相似度進度: {e}/{M} (chunk {s//sim_chunk + 1}/{math.ceil(M/sim_chunk)})")
    if ns:
        d['column','name_sim','column'].edge_index=torch.tensor([ns,nd],dtype=torch.long)
        print(f"欄位名稱相似邊數: {len(ns)}")

    # -------- 數值分布相似度（跨表） --------
    print('計算數值分布相似度（跨表）...')
    ds,dd=[],[]
    idxs=[i for i,h in enumerate(c_hist) if h is not None]
    if idxs:
        H=torch.stack([F.normalize(c_hist[i],p=2,dim=0) for i in idxs])
        M=H.size(0)
        for s in range(0,M,sim_chunk):
            e=min(s+sim_chunk,M)
            block=H[s:e]  # [B,16]
            sims=torch.matmul(block,H.T)  # [B,M]
            idx=torch.arange(s,e)
            sims[torch.arange(e-s),idx]=float('-inf')
            top=torch.topk(sims,k=min(topk_dist,M),largest=True)
            top_idx=top.indices.cpu().tolist()
            for row,k0 in enumerate(range(s,e)):
                i=idxs[k0]
                for jj in top_idx[row]:
                    j=idxs[jj]
                    if c_tbl_idx[i]!=c_tbl_idx[j]:
                        ds.append(i); dd.append(j)
            if ((s//sim_chunk) % log_every)==0 or e==M:
                print(f"數值分布相似度進度: {e}/{M} (chunk {s//sim_chunk + 1}/{math.ceil(M/sim_chunk)})")
    if ds:
        d['column','dist_sim','column'].edge_index=torch.tensor([ds,dd],dtype=torch.long)
        print(f"數值分布相似邊數: {len(ds)}")

    return d


def main():
    base=os.path.dirname(os.path.abspath(__file__))
    tj=os.path.join(base,'table','train','feta','table.jsonl')
    tk=int(os.environ.get('TOPK_TABLE_SIM','10'))
    kc=int(os.environ.get('TOPK_COL_SIM','3'))
    kn=int(os.environ.get('TOPK_COL_NAME_SIM','5'))
    kd=int(os.environ.get('TOPK_COL_DIST_SIM','5'))
    ch=int(os.environ.get('SIM_CHUNK','512'))
    le=int(os.environ.get('LOG_EVERY_CHUNK','1'))
    try:
        print('正在讀取表格資料...')
        g=build_graph(tj,tk,kc,kn,kd,sim_chunk=ch,log_every=le)
        out=os.path.join(base,'graph_cache.pt')
        torch.save(g,out)
        print(f'圖構建完成，保存至 {out}')
    except Exception as e:
        print('構圖失敗:',e)

if __name__=='__main__':
    main()