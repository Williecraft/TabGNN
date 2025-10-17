

import os
import io
import csv
import json
import random
import torch
import torch.nn.functional as F
from torch import nn
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv
from torch_geometric.nn import APPNP
from sentence_transformers import SentenceTransformer

def load_jsonl(path):
    with io.open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def parse_csv_line(line):
    return next(csv.reader([line]))


def build_graph_from_feta(table_jsonl_path, query_jsonl_path, max_tables=500, topk_table_sim=10, topk_col_sim=3):
    tables = []
    for obj in load_jsonl(table_jsonl_path):
        tables.append(obj)
        if len(tables) >= max_tables:
            break

    table_ids = [t["id"] for t in tables]
    table_id_to_idx = {tid: i for i, tid in enumerate(table_ids)}

    embedder = SentenceTransformer("distiluse-base-multilingual-cased-v2")

    table_feats = []
    columns = []
    table_to_cols = {}
    column_key_to_idx = {}
    col_feats = []

    for t in tables:
        tid = t["id"]
        sheet = str(t.get("sheet_name", ""))
        header_row = t.get("header", [])
        header_cols = parse_csv_line(header_row[0]) if header_row else []

        instances = [parse_csv_line(r) for r in t.get("instances", [])]

        table_text_parts = [sheet] + header_cols
        for r in instances[:10]:
            table_text_parts += r
        table_text = ", ".join([str(x) for x in table_text_parts if str(x).strip()])
        table_emb = embedder.encode(table_text)
        table_feats.append(torch.tensor(table_emb, dtype=torch.float))

        cols_this_table = []
        for j, col_name in enumerate(header_cols):
            values = []
            for r in instances[:10]:
                if j < len(r):
                    values.append(str(r[j]))
            text = ", ".join([str(col_name)] + values)
            c_emb = embedder.encode(text)
            c_key = f"{tid}::{j}::{col_name}"
            column_key_to_idx[c_key] = len(col_feats)
            col_feats.append(torch.tensor(c_emb, dtype=torch.float))
            cols_this_table.append(c_key)
        table_to_cols[tid] = cols_this_table
        columns.append((tid, header_cols))

    data = HeteroData()
    data["table"].x = torch.stack(table_feats) if table_feats else torch.empty((0, 512))
    data["column"].x = torch.stack(col_feats) if col_feats else torch.empty((0, 512))

    t_src, t_dst = [], []
    for tid, cols_keys in table_to_cols.items():
        for ck in cols_keys:
            t_src.append(table_id_to_idx[tid])
            t_dst.append(column_key_to_idx[ck])
    if t_src:
        edge_tc = torch.tensor([t_src, t_dst], dtype=torch.long)
        edge_ct = torch.tensor([t_dst, t_src], dtype=torch.long)
        data["table", "has_column", "column"].edge_index = edge_tc
        data["column", "rev_has_column", "table"].edge_index = edge_ct

    with torch.no_grad():
        x_table = F.normalize(data["table"].x, p=2, dim=1) if data["table"].x.size(0) > 0 else data["table"].x
        x_col = F.normalize(data["column"].x, p=2, dim=1) if data["column"].x.size(0) > 0 else data["column"].x

    sim_src, sim_dst = [], []
    for i in range(x_table.size(0)):
        sims = torch.mv(x_table, x_table[i])
        sims[i] = -1e9
        topk = torch.topk(sims, k=min(topk_table_sim, sims.numel()), largest=True).indices.tolist()
        for j in topk:
            sim_src.append(i)
            sim_dst.append(j)
    if sim_src:
        data["table", "sim", "table"].edge_index = torch.tensor([sim_src, sim_dst], dtype=torch.long)

    col_sim_src, col_sim_dst = [], []
    for tid, cols_keys in table_to_cols.items():
        idxs = [column_key_to_idx[ck] for ck in cols_keys]
        for a in idxs:
            for b in idxs:
                if a != b:
                    col_sim_src.append(a)
                    col_sim_dst.append(b)
    if x_col.size(0) > 0:
        for i in range(x_col.size(0)):
            sims = torch.mv(x_col, x_col[i])
            sims[i] = -1e9
            topk = torch.topk(sims, k=min(topk_col_sim, sims.numel()), largest=True).indices.tolist()
            for j in topk:
                col_sim_src.append(i)
                col_sim_dst.append(j)
    if col_sim_src:
        data["column", "col_sim", "column"].edge_index = torch.tensor([col_sim_src, col_sim_dst], dtype=torch.long)

    query_items = []
    for obj in load_jsonl(query_jsonl_path):
        tid = obj.get("table_id")
        if tid in table_id_to_idx:
            for q in obj.get("queries", []):
                query_items.append((q, table_id_to_idx[tid]))

    query_texts = [q for q, _ in query_items]
    query_embs = embedder.encode(query_texts)
    query_embs = [torch.tensor(e, dtype=torch.float) for e in query_embs]
    pos_indices = [pi for _, pi in query_items]

    return data, query_embs, pos_indices


class DiffusionRetrievalGNN(nn.Module):
    def __init__(self, hidden_dim, alpha=0.2, k=10):
        super().__init__()
        self.conv1 = HeteroConv(
            {
                ("table", "has_column", "column"): SAGEConv((-1, -1), hidden_dim),
                ("column", "rev_has_column", "table"): SAGEConv((-1, -1), hidden_dim),
                ("column", "col_sim", "column"): SAGEConv((-1, -1), hidden_dim),
                ("table", "sim", "table"): SAGEConv((-1, -1), hidden_dim),
            },
            aggr="sum",
        )
        self.conv2 = HeteroConv(
            {
                ("table", "has_column", "column"): SAGEConv((-1, -1), hidden_dim),
                ("column", "rev_has_column", "table"): SAGEConv((-1, -1), hidden_dim),
                ("column", "col_sim", "column"): SAGEConv((-1, -1), hidden_dim),
                ("table", "sim", "table"): SAGEConv((-1, -1), hidden_dim),
            },
            aggr="sum",
        )
        self.q_proj = nn.Linear(512, hidden_dim)
        self.appnp = APPNP(K=k, alpha=alpha)
        self.lin = nn.Linear(hidden_dim, 1)

    def forward(self, data, query_vec, start_idx=None):
        x_dict = self.conv1(data.x_dict, data.edge_index_dict)
        for k in x_dict:
            x_dict[k] = F.relu(x_dict[k])
        x_dict = self.conv2(x_dict, data.edge_index_dict)
        x_table = x_dict["table"]
        q = self.q_proj(query_vec)
        z0 = torch.zeros_like(x_table)
        if x_table.size(0) > 0:
            if start_idx is not None:
                # 訓練時可選擇在真值節點注入查詢（教師信號），但評估時不應使用。
                z0[start_idx] = q
            else:
                # 評估或不使用真值時，根據查詢與表格文本嵌入的相似度加權注入。
                # 使用原始 512 維表格嵌入與查詢向量做 cosine 相似度。
                x_raw = data["table"].x  # 512-d raw text embeddings
                x_raw = F.normalize(x_raw, p=2, dim=1)
                q_raw = F.normalize(query_vec, p=2, dim=0)
                weights = torch.mv(x_raw, q_raw)  # [num_tables]
                weights = torch.clamp(weights, min=0.0)
                z0 = weights.unsqueeze(1) * q.unsqueeze(0)
        edge_tt = data["table", "sim", "table"].edge_index
        z = self.appnp(x_table + z0, edge_tt)
        scores = self.lin(z).squeeze(-1)
        return scores


def sample_negative(num_neg, num_tables, pos_idx):
    neg = []
    while len(neg) < num_neg:
        r = random.randrange(num_tables)
        if r != pos_idx and r not in neg:
            neg.append(r)
    return neg


def train(
    table_jsonl_path,
    query_jsonl_path,
    epochs=20,
    max_tables=500,
    lr=1e-3,
    topk_table_sim=10,
    topk_col_sim=3,
    neg_k=8,
    report_every=5,
    save_path=None,
):
    data, query_embs, pos_indices = build_graph_from_feta(
        table_jsonl_path, query_jsonl_path, max_tables=max_tables, topk_table_sim=topk_table_sim, topk_col_sim=topk_col_sim
    )

    model = DiffusionRetrievalGNN(hidden_dim=128, alpha=0.2, k=10)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    def evaluate_k(k=5):
        model.eval()
        with torch.no_grad():
            recalls, rr = [], []
            for q, pos in zip(query_embs, pos_indices):
                scores = model(data, q, start_idx=None)
                topk = torch.topk(scores, k=min(k, scores.numel())).indices.tolist()
                recalls.append(1.0 if pos in topk else 0.0)
                ranks = torch.argsort(torch.argsort(scores, descending=True))
                r = ranks[pos].item() + 1
                rr.append(1.0 / r)
            return float(sum(recalls) / len(recalls)), float(sum(rr) / len(rr))

    for ep in range(1, epochs + 1):
        total_loss = 0.0
        for q, pos in zip(query_embs, pos_indices):
            optimizer.zero_grad()
            # 不使用真值注入，改為以相似度加權注入，避免學習與評估偏差。
            scores = model(data, q, start_idx=None)
            negs = sample_negative(neg_k, data["table"].x.size(0), pos)
            logits = torch.cat([scores[pos].unsqueeze(0), scores[negs]])
            loss = F.cross_entropy(logits.view(1, -1), torch.tensor([0]))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        if ep % report_every == 0:
            rec, mrr = evaluate_k(k=5)
            print(f"Epoch {ep} | Loss {total_loss:.4f} | Recall@5 {rec:.4f} | MRR {mrr:.4f}")
        else:
            print(f"Epoch {ep} | Loss {total_loss:.4f}")
    # 可選保存模型參數
    if save_path is not None:
        try:
            torch.save(model.state_dict(), save_path)
            print(f"Model saved to: {save_path}")
        except Exception as e:
            print(f"Failed to save model to {save_path}: {e}")
    return model


def main():
    base = os.path.dirname(os.path.abspath(__file__))
    table_jsonl_path = os.path.join(base, "table", "train", "feta", "table.jsonl")
    query_jsonl_path = os.path.join(base, "data", "train", "feta", "generate_query.jsonl")
    train(
        table_jsonl_path=table_jsonl_path,
        query_jsonl_path=query_jsonl_path,
        epochs=20,
        max_tables=500,
        lr=1e-3,
        topk_table_sim=10,
        topk_col_sim=3,
        neg_k=8,
        report_every=5,
        save_path=os.path.join(base, "diffusion_appnp.pt"),
    )


if __name__ == "__main__":
    import torch_geometric.nn as tg_nn
    print(hasattr(tg_nn, 'APPNP'))  # True表示存在

    main()
