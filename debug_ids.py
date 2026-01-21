import torch
import json
import sys

def check_ids():
    # Load graph to get id_to_idx
    print("Loading graph...")
    try:
        data = torch.load("/user_data/TabGNN/data/processed/graph.pt", weights_only=False)
    except Exception as e:
        print(f"Error loading graph: {e}")
        return

    # Reconstruct id_to_idx from data['table'].node_id
    # Assuming node_id is a list of strings stored in the graph object
    # If not, we might need to look at how id_to_idx is constructed in train_model.py
    
    # Let's look at train_model.py again to see how id_to_idx is made.
    # It seems it's passed to load_training_data.
    # In main(), it does:
    # data = torch.load(BASE_CONFIG['GRAPH_FILE'])
    # id_to_idx = {id: idx for idx, id in enumerate(data['table'].node_id)}
    
    try:
        id_to_idx = data.metadata_maps['table_id_to_idx']
    except (AttributeError, KeyError):
        print("Graph data does not have 'metadata_maps['table_id_to_idx']'.")
        print("Available keys:", data.keys)
        return
    
    print(f"Loaded {len(id_to_idx)} table IDs from graph.")
    first_key = list(id_to_idx.keys())[0]
    print(f"Sample graph IDs: {list(id_to_idx.keys())[:5]}")
    print(f"Type of first key: {type(first_key)}")

    query_file = "/user_data/TabGNN/data/table/test/feta/query.jsonl"
    print(f"Checking query file: {query_file}")
    
    with open(query_file, 'r') as f:
        for i, line in enumerate(f):
            if i >= 5: break
            item = json.loads(line)
            pos_id = str(item.get('id'))
            print(f"Query {i}: id='{pos_id}' (type: {type(pos_id)})")
            
            if pos_id in id_to_idx:
                print(f"  -> Found in graph! Index: {id_to_idx[pos_id]}")
            else:
                print(f"  -> NOT FOUND in graph.")

if __name__ == "__main__":
    check_ids()
