import os, sys, json, time, math
import torch
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

# 確保可匯入專案根目錄的模組
root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root not in sys.path:
    sys.path.insert(0, root)

from train_model import DiffusionRetrievalGNN

# Config via env
QUERY_PATH = os.environ.get('QUERY_PATH', os.path.join(os.path.dirname(__file__), '..', 'table', 'train', 'feta', 'query.jsonl'))
LIMIT = int(os.environ.get('N', '1000'))
FORCE_SEMANTIC = os.environ.get('FORCE_SEMANTIC', '1') == '1'
EMBED_MODEL = os.environ.get('EMBED_MODEL', 'distiluse-base-multilingual-cased-v2')
EMBED_DEVICE = os.environ.get('EMBED_DEVICE', 'cpu')
EMBED_BATCH = int(os.environ.get('EMBED_BATCH', '64'))
LOG_EVERY = int(os.environ.get('LOG_EVERY', '50'))
TOPK1 = 1
TOPK5 = 5
TOPK10 = 10


def load_graph():
    p = os.path.join(root, 'graph_cache.pt')
    print(f'[INFO] 載入圖: {p}')
    try:
        g = torch.load(p, map_location='cpu', weights_only=False)
        return g
    except Exception as e:
        print('[ERROR] 載入圖失敗:', e)
        sys.exit(1)


def load_model():
    ck = os.path.join(root, 'retrieval_model.pt')
    print(f'[INFO] 載入模型: {ck}')
    if not os.path.exists(ck):
        print('[ERROR] 找不到模型，請先執行 train_model.py')
        sys.exit(1)
    m = DiffusionRetrievalGNN(h=int(os.environ.get('HIDDEN_DIM', '128')), a=0.2, k=10)
    try:
        sd = torch.load(ck, map_location='cpu')
        m.load_state_dict(sd, strict=False)
    except Exception as e:
        print('[ERROR] 載入模型失敗:', e)
        sys.exit(1)
    m.eval()
    return m


class Embedder:
    def __init__(self):
        self.ok = False
        self.m = None
        if SentenceTransformer:
            try:
                self.m = SentenceTransformer(EMBED_MODEL)
                try:
                    self.m.to(EMBED_DEVICE)
                except Exception:
                    pass
                self.ok = True
                print(f'[INFO] 使用語義嵌入模型: {EMBED_MODEL} on {EMBED_DEVICE}')
            except Exception as e:
                print('[WARN] 語義模型載入失敗:', e)
                self.m = None
        if FORCE_SEMANTIC and not self.ok:
            print('[ERROR] FORCE_SEMANTIC=1 但無法載入語義嵌入模型，請安裝 sentence-transformers 或更換 EMBED_MODEL')
            sys.exit(1)

    def encode_list(self, texts):
        if not self.ok:
            print('[ERROR] 未載入語義模型；不允許使用 hash 嵌入')
            sys.exit(1)
        return self.m.encode(texts, batch_size=EMBED_BATCH)


def read_queries(path, limit):
    print(f'[INFO] 讀取查詢: {path}，限制 {limit} 筆')
    qs = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, 1):
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if 'id' in obj and 'question' in obj and obj['question']:
                qs.append({'id': int(obj['id']), 'question': obj['question']})
            if len(qs) >= limit:
                break
    print(f'[INFO] 讀取完成，共 {len(qs)} 筆')
    return qs


def build_id_to_idx(graph):
    t = graph['table']
    mapping = getattr(t, 'table_idx_to_id', None)
    if not mapping:
        print('[ERROR] 圖中缺少 table_idx_to_id 對應，無法比對正解 id')
        sys.exit(1)
    id_to_idx = {int(v): int(k) for k, v in mapping.items()}
    return id_to_idx


def evaluate(model, graph, embeddings, queries, id_to_idx):
    n = 0
    hits1 = hits5 = hits10 = 0
    mrr_sum = 0.0
    rank_sum = 0.0
    start = time.time()
    for i, (qv, q) in enumerate(zip(embeddings, queries), 1):
        gt_id = q['id']
        idx_gt = id_to_idx.get(gt_id)
        if idx_gt is None:
            continue
        q_tensor = torch.tensor(qv, dtype=torch.float)
        with torch.no_grad():
            s = model(graph, q_tensor)
        # rank
        sorted_scores, sorted_idx = torch.sort(s, descending=True)
        pos = (sorted_idx == idx_gt).nonzero(as_tuple=True)[0]
        if len(pos) == 0:
            continue
        r = int(pos.item()) + 1  # 1-based rank
        n += 1
        rank_sum += r
        mrr_sum += 1.0 / r
        if r <= TOPK1:
            hits1 += 1
        if r <= TOPK5:
            hits5 += 1
        if r <= TOPK10:
            hits10 += 1
        if LOG_EVERY and i % LOG_EVERY == 0:
            print(f'[PROG] {i} / {len(queries)} | 當前 MRR={mrr_sum/max(1,n):.4f} | Hit@1={hits1/max(1,n):.4f}')
    dur = time.time() - start
    if n == 0:
        print('[ERROR] 沒有可評估的查詢（id 未對上圖）')
        sys.exit(1)
    metrics = {
        'evaluated': n,
        'hit@1': hits1 / n,
        'hit@5': hits5 / n,
        'hit@10': hits10 / n,
        'mrr': mrr_sum / n,
        'mean_rank': rank_sum / n,
        'seconds': dur,
        'config': {
            'QUERY_PATH': QUERY_PATH,
            'LIMIT': LIMIT,
            'EMBED_MODEL': EMBED_MODEL,
            'EMBED_DEVICE': EMBED_DEVICE,
            'EMBED_BATCH': EMBED_BATCH,
            'FORCE_SEMANTIC': FORCE_SEMANTIC,
        },
    }
    return metrics


def main():
    graph = load_graph()
    print(f'[INFO] 圖表數量: {graph["table"].x.size(0)}')
    id_to_idx = build_id_to_idx(graph)
    model = load_model()
    queries = read_queries(QUERY_PATH, LIMIT)
    emb = Embedder()
    texts = [q['question'] for q in queries]
    print(f'[INFO] 開始批次嵌入查詢向量，共 {len(texts)} 筆 (batch={EMBED_BATCH})')
    qvecs = emb.encode_list(texts)
    print('[INFO] 開始評測...')
    metrics = evaluate(model, graph, qvecs, queries, id_to_idx)
    print('[RESULT] 評測指標:')
    print(f"- evaluated={metrics['evaluated']}")
    print(f"- hit@1={metrics['hit@1']:.4f}")
    print(f"- hit@5={metrics['hit@5']:.4f}")
    print(f"- hit@10={metrics['hit@10']:.4f}")
    print(f"- mrr={metrics['mrr']:.4f}")
    print(f"- mean_rank={metrics['mean_rank']:.2f}")
    print(f"- seconds={metrics['seconds']:.2f}")
    out = os.path.join(root, 'eval_results.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f'[INFO] 已保存評測結果到 {out}')


if __name__ == '__main__':
    main()