import os
import torch
from diffusion import DiffusionRetrievalGNN, build_graph_from_feta

def run_infer(
    table_jsonl_path,
    totto_query_jsonl_path,
    model_paths,
    max_tables=500,
    topk_table_sim=10,
    topk_col_sim=3,
    sample_print=10,
):
    data, query_embs, pos_indices = build_graph_from_feta(
        table_jsonl_path,
        totto_query_jsonl_path,
        max_tables=max_tables,
        topk_table_sim=topk_table_sim,
        topk_col_sim=topk_col_sim,
        query_style="totto",
    )
    print(f"Loaded tables: {data['table'].x.size(0)}, queries: {len(query_embs)}")
    if len(query_embs) == 0:
        print("No queries matched table ids; please ensure query.jsonl corresponds to table set.")
        return

    # 選擇可用模型路徑
    load_path = None
    for p in model_paths:
        if os.path.exists(p):
            load_path = p
            break
    if load_path is None:
        print("No model checkpoint found among:")
        for p in model_paths:
            print("  - ", p)
        return
    print("Using checkpoint:", load_path)

    model = DiffusionRetrievalGNN(hidden_dim=128, alpha=0.2, k=10)
    state = torch.load(load_path, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    recalls, rr = [], []
    examples = []
    with torch.no_grad():
        for idx, (q, pos) in enumerate(zip(query_embs, pos_indices)):
            scores = model(data, q, start_idx=None)
            topk = torch.topk(scores, k=min(5, scores.numel())).indices.tolist()
            recalls.append(1.0 if pos in topk else 0.0)
            ranks = torch.argsort(torch.argsort(scores, descending=True))
            r = ranks[pos].item() + 1
            rr.append(1.0 / r)

            if idx < sample_print:
                pred_top1 = int(torch.argmax(scores).item())
                examples.append({
                    "pos_idx": int(pos),
                    "pred_idx": pred_top1,
                    "hit@5": pos in topk,
                    "rank": r,
                })

    print(f"Recall@5: {sum(recalls)/len(recalls):.4f} | MRR: {sum(rr)/len(rr):.4f}")
    print("Examples (first few):")
    for ex in examples:
        print(ex)

if __name__ == "__main__":
    base = os.path.dirname(os.path.abspath(__file__))
    table_jsonl_path = os.path.join(base, "table", "train", "feta", "table.jsonl")
    totto_query_jsonl_path = os.path.join(base, "table", "train", "feta", "query.jsonl")
    candidates = [
        os.path.join(base, "diffusion_appnp_40ep.pt"),
        os.path.join(base, "diffusion_appnp_15ep.pt"),
        os.path.join(base, "diffusion_appnp_quick.pt"),
        os.path.join(base, "diffusion_appnp.pt"),
    ]
    run_infer(
        table_jsonl_path,
        totto_query_jsonl_path,
        candidates,
        max_tables=500,
        topk_table_sim=10,
        topk_col_sim=3,
        sample_print=10,
    )